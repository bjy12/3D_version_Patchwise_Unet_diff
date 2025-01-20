import os
import logging
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from accelerate import Accelerator
from dataclasses import dataclass
import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm
from typing import Optional, Union, Dict ,List
from omegaconf import OmegaConf
import yaml
#from model.Song3DUnet import Song_Unet3D
from model.Song3DUnetV4 import Song_Unet3D
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from datasets.geometry import Geometry
from pachify_and_projector import pachify3d_projected_points , init_projector
#from training_cfg_pcc import BaseTrainingConfig
from training_cfg_resnetbackbone import BaseTrainingConfig
from datasets.datautils3d import load_diffusion_condition
from metrics import calculate_metrics ,calculate_metrics_each_names
import pdb

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class InferenceOutput(BaseOutput):
    """Output class for inference results."""
    predictions: np.ndarray
    ground_truth: np.ndarray

def load_model_from_checkpoint(checkpoint_dir: str, use_ema: bool = True) -> Dict:
    """Load model and configuration from checkpoint directory."""
    logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
    model = Song_Unet3D.from_pretrained(checkpoint_dir, subfolder="unet" ,assign=True)
    
    ema_model = None
    if use_ema and os.path.exists(os.path.join(checkpoint_dir, "unet_ema")):
        logger.info("Loading EMA model")
        ema_model = EMAModel.from_pretrained(
            os.path.join(checkpoint_dir, "unet_ema"), 
            Song_Unet3D,
            assign=True
        )
    
    return {
        "model": model,
        "ema_model": ema_model
    }

class CheckpointInferencePipeline(DiffusionPipeline):
    def __init__(
        self,
        net: Song_Unet3D,
        scheduler: DDPMScheduler,
        use_acc: True,
    ):
        super().__init__()
        

        #self._device = None
        #pdb.set_trace()
        if use_acc:

             self.accelerator = Accelerator(mixed_precision=None)
             net = self.accelerator.prepare(net)
             #self._device = self.accelerator.device

             
        self.register_modules(
            net=net,
            scheduler=scheduler
        )
       
        self.weight_dtype = torch.float16 if self.accelerator.mixed_precision == "fp16" else torch.float32

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        config_path: str,
        use_ema: bool = True,
        **kwargs
    ):
        """创建pipeline的推荐方法"""
        # 加载配置
        data_dict = OmegaConf.load(config_path)
        cfg = BaseTrainingConfig(**data_dict)
        pdb.set_trace()
        # 加载模型
        loaded_models = load_model_from_checkpoint(checkpoint_dir, use_ema)
        model = loaded_models["model"] if not use_ema else loaded_models["ema_model"]
        
        # 初始化scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.ddpm_num_steps,
            beta_schedule=cfg.ddpm_beta_schedule,
            prediction_type=cfg.prediction_type,
        )

        # 创建pipeline实例
        pipeline = cls(
            net=model,
            scheduler=scheduler,
            use_acc=True
        )
        
        # 保存配置和其他必要的属性
        pipeline.cfg = cfg
        pipeline.projector = init_projector(cfg.geo_cfg_path)
        pdb.set_trace()
        pipeline.patch_size = cfg.pachify_size
        pipeline.start_pos = (cfg.img_resolution - cfg.pachify_size) // 2
        pipeline.positions = {
            'i': pipeline.start_pos,
            'j': pipeline.start_pos,
            'k': pipeline.start_pos,
        }
        
        logger.info(f"Pipeline initialized with patch_size: {pipeline.patch_size}, start_pos: {pipeline.start_pos}")
        
        return pipeline

    # @property
    # def device(self):
    #     if self._device is None:
    #         self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     return self._device

    # def to(self, device):
    #     self._device = torch.device(device)
    #     super().to(device)
    #     return self

    def _save_volume(self, image: np.ndarray, save_path: str , spacing: np.ndarray):
        """Helper method to save 3D volumes."""
        # if uint8:
        #     image = (image * 255).astype(np.uint8)
        if len(image.shape) == 5:  # Batch of volumes
            for i in range(image.shape[0]):
                volume = image[i, 0]  # Take first channel
                volume = volume.transpose(2, 1, 0)  # Adjust for SimpleITK
                sitk_image = sitk.GetImageFromArray(volume)
                if spacing is not None:
                    sitk_image.SetSpacing(spacing.astype(np.float64)) # unit: mm
                batch_save_path = save_path.format(batch_idx=i)
                os.makedirs(os.path.dirname(batch_save_path), exist_ok=True)
                sitk.WriteImage(sitk_image, batch_save_path)
        else:  # Single volume
            volume = image[0]  # Take first channel
            volume = volume.transpose(2, 1, 0)  # Adjust for SimpleITK
            sitk_image = sitk.GetImageFromArray(volume)
            if spacing is not None:
                sitk_image.SetSpacing(spacing.astype(np.float64)) # unit: mm
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sitk.WriteImage(sitk_image, save_path)

    def save_metrics_to_txt(self, all_metrics: List[Dict], output_dir: str):
        """保存metrics到txt文件"""
        metrics_path = os.path.join(output_dir, 'case_metrics.txt')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            # 写入标题行
            f.write(f"{'Batch':^6} {'Case Name':<20} {'PSNR':^10} {'SSIM':^10}\n")
            f.write("-" * 50 + "\n")  # 分隔线
            
            # 写入每个case的metrics
            for metric in all_metrics:
                f.write(f"{metric['batch_idx']:^6} {metric['name']:<20} "
                       f"{metric['psnr']:^10.4f} {metric['ssim']:^10.4f}\n")
            
            # 写入平均值
            f.write("-" * 50 + "\n")  # 分隔线
            avg_psnr = sum(m['psnr'] for m in all_metrics) / len(all_metrics)
            avg_ssim = sum(m['ssim'] for m in all_metrics) / len(all_metrics)
            f.write(f"{'Average':^26} {avg_psnr:^10.4f} {avg_ssim:^10.4f}\n")
        
        logger.info(f"Saved metrics to {metrics_path}")

    @torch.no_grad()
    def __call__(
        self,
        dataloader,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_dir: Optional[str] = None,
        save_intermediates: bool = False,
        save_interval: int = 100
    ):
        """
        Run inference on dataloader using loaded checkpoint.
        """
        # Use config value if not specified
        if num_inference_steps is None:
            num_inference_steps = self.cfg.ddpm_num_inference_steps
             
        self.net.eval()
        self.scheduler.set_timesteps(num_inference_steps)
        results = []
        all_metrics = []
        dataloader = self.accelerator.prepare(dataloader)

        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Move batch to device and process
            # projs = batch['projs'].to(device=self.device, dtype=self.weight_dtype)
            # clean_images = batch['gt_idensity'].to(device=self.device, dtype=self.weight_dtype)
            # angles = batch['angles'].to(device=self.device)

            projs = batch['projs'].to(dtype=self.weight_dtype)
            clean_images = batch['gt_idensity'].to(dtype=self.weight_dtype)
            angles = batch['angles']
            batch_names = batch['name']
            # Process data using pachify3d_projected_points
            patch_image_tensor, patch_pos_tensor, projs_points_tensor = pachify3d_projected_points(
                clean_images,
                self.patch_size,
                angles,
                self.projector,
                patch_pos=self.positions
            )
            
            projs_points_tensor = projs_points_tensor.to(self.weight_dtype)
            
            # Get shapes for initialization
            batch_size = patch_image_tensor.shape[0]
            image_shape = patch_image_tensor.shape

            # Initialize noise
            noise = randn_tensor(
                image_shape, 
                generator=generator, 
                device=self.accelerator.device, 
                dtype=self.weight_dtype
            )

            # Initialize noisy image with position information
            noisy_image = torch.cat([noise, patch_pos_tensor], dim=1)

            # Process through denoising steps
            for step_idx, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # Expand timestep for batch
                timesteps = t.expand(batch_size).to(self.device)

                # Get model prediction
                model_output = self.net(
                    noisy_image, 
                    timesteps, 
                    projs, 
                    projs_points_tensor
                )
                
                # Split position and noise components
                pos = noisy_image[:, 1:4, ...]
                noise_part = noisy_image[:, :1, ...]

                # Scheduler step
                scheduler_output = self.scheduler.step(
                    model_output, 
                    t, 
                    noise_part,
                    generator=generator
                )
                
                # Recombine noise and position information
                noisy_image = torch.cat([scheduler_output.prev_sample, pos], dim=1)
                
                
               # Save intermediates if requested
                if save_intermediates and step_idx % save_interval == 0 and output_dir:
                    intermediate_image = (scheduler_output.pred_original_sample + 1) / 2
                    intermediate_image = intermediate_image.clamp(0, 1).cpu().numpy()
                    for i, name in enumerate(batch_names):
                        save_path = os.path.join(
                            output_dir, 
                            f'intermediate/batch_{batch_idx:04d}_{name}_step_{step_idx:04d}.nii.gz'
                        )
                        if self.accelerator.is_main_process:
                            self._save_volume(intermediate_image[i:i+1], save_path)

            # Process final prediction
            final_pred = (scheduler_output.pred_original_sample + 1.) / 2.
            final_pred = final_pred.clamp(0, 1).cpu().numpy()
            gt_image   = patch_image_tensor.cpu().numpy()
            gt_image   = ( gt_image + 1. ) / 2. 
            
            
            final_pred = (final_pred * 255).astype(np.uint8)
            gt_image = (gt_image * 255).astype(np.uint8)
            # 计算每个case的metrics
            case_metrics = calculate_metrics_each_names(
                pred=final_pred,
                target=gt_image,
                names=batch_names,
                data_range=255
            )
            spacing_values = np.array([2.0, 2.0, 2.0], dtype=np.float64)
            # 添加batch信息并存储metrics
            batch_psnr = 0
            batch_ssim = 0
            for metric in case_metrics:
                metric['batch_idx'] = batch_idx
                all_metrics.append(metric)
                batch_psnr += metric['psnr']
                batch_ssim += metric['ssim']
                # 打印每个case的metrics
                logger.info(f"Case {metric['name']}: PSNR = {metric['psnr']:.4f}, SSIM = {metric['ssim']:.4f}")
            
            # 打印当前batch的平均metrics
            batch_avg_psnr = batch_psnr / len(case_metrics)
            batch_avg_ssim = batch_ssim / len(case_metrics)
            logger.info(f"\nBatch {batch_idx} Average - PSNR: {batch_avg_psnr:.4f}, SSIM: {batch_avg_ssim:.4f}\n")

            # Save final results if output directory provided
            if output_dir:
                for i, name in enumerate(batch_names):
                    pred_path = os.path.join(
                        output_dir, 
                        f'predictions/sample_{batch_idx:04d}_{name}.nii.gz'
                    )
                    gt_path = os.path.join(
                        output_dir, 
                        f'ground_truth/sample_{batch_idx:04d}_{name}.nii.gz'
                    )
                if self.accelerator.is_main_process:
                    self._save_volume(final_pred, pred_path ,spacing=spacing_values)
                    self._save_volume(gt_image, gt_path , spacing=spacing_values)


            results.append(InferenceOutput(
                predictions=final_pred,
                ground_truth=gt_image
            ))
            #pdb.set_trace()
            logger.info(f"Completed inference for batch {batch_idx}")
        if output_dir and self.accelerator.is_main_process:
            self.save_metrics_to_txt(all_metrics, output_dir)

        return results , all_metrics



if __name__ == "__main__":
    pipeline = CheckpointInferencePipeline.from_pretrained(
                checkpoint_dir='F:/Code_Space/diff_implict_xray_to_ct/res_net_backBoneV0/check_points_/res_net_backBone_epo_2k/checkpoint-50000/unet',
                config_path='./cfg_pcc.json',
                use_ema=False)
    #pipeline.to('cuda')
    pdb.set_trace()
    img_root = 'F:/Data_Space/Pelvic1K/processed_128x128_s2'
    files_list_path = './files_name/pelvic_coord_test_16.txt'
    dataset = load_diffusion_condition(
        img_root, 
        files_list_path
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=pipeline.cfg.valid_batch_size,
        shuffle=False, 
        num_workers=pipeline.cfg.dataloader_num_workers
    )
    pdb.set_trace()
    results , metrics = pipeline(dataloader=dataloader, num_inference_steps=1000,output_dir='./inference_res_net_back_',save_intermediates=False,save_interval=100)
    
    # 可以查看每个case的metrics
    for metric in metrics:
        print(f"Batch {metric['batch_idx']}, Case {metric['name']}: "
              f"PSNR = {metric['psnr']:.4f}, SSIM = {metric['ssim']:.4f}")



# Example usage:
"""
# Initialize pipeline with checkpoint
pipeline = CheckpointInferencePipeline(
    checkpoint_dir='path/to/checkpoint-XXXXX',
    config_path='path/to/config.yaml',
    use_ema=True
)
pipeline.to('cuda')

# Create dataloader
dataloader = torch.utils.data.DataLoader(...)

# Run inference
results = pipeline(
    dataloader=dataloader,
    output_dir='./inference_outputs',
    save_intermediates=True,
    save_interval=100
)
"""