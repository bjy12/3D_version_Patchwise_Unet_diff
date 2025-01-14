import torch
import torch.nn as nn
import os
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union ,Dict
from dataclasses import dataclass
import numpy as np
import pickle
import safetensors
from model.Song3DUnet import  Song_Unet3D
from diffusers import DDPMPipeline, DDPMScheduler
import pdb
import SimpleITK as sitk
from datasets.geometry import Geometry
from datasets.datautils3d import  load_diffusion_condition
from datasets.slicedataset import load_slice_dataset
from pachify_and_projector import pachify3d_projected_points , full_volume_inference_process
import matplotlib.pyplot as plt
import yaml
from metrics import calculate_metrics
@dataclass
class TrainingInferenceOutput(BaseOutput):
    """Output class for training inference results."""
    pred_samples: np.ndarray
    gt_samples: np.ndarray
    metrics_log: Dict[str, float]
    names: Optional[List[str]] = None




class TrainingInferencePipeline(DiffusionPipeline):
    """Pipeline for inference during training, using pre-processed data."""
    model_cpu_offload_seq = "net"

    def __init__(self, net, scheduler):
        super().__init__()
        self.register_modules(net=net, scheduler=scheduler)
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def set_device(self, device):
        self._device = device

    @torch.no_grad()
    def __call__(
        self,
        noisy_images,          # 已处理的噪声图像
        patch_pos_tensor,      # 位置信息
        projs,                 # 投影数据
        projs_points_tensor,   # 投影点信息
        patch_image_tensor,    # 原始patch图像（用于ground truth）
        names,
        generator=None,
        num_inference_steps=1000,
        output_type="npy",
        return_dict=True,
        save_path=None,
    ):
        """
        执行推理过程，使用已经预处理好的数据。

        Args:
            noisy_images: 已添加噪声的图像
            patch_pos_tensor: patch位置信息
            projs: 投影数据
            projs_points_tensor: 投影点信息
            patch_image_tensor: 原始patch图像
            generator: 随机数生成器
            num_inference_steps: 推理步数
            output_type: 输出类型
            return_dict: 是否返回字典格式
            save_path: 保存路径
        """
        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps)
        
        # 初始化当前图像（使用已处理的噪声图像和位置信息）
        current_image = torch.cat([noisy_images, patch_pos_tensor], dim=1)

        # 去噪循环
        for step_idx , t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # 扩展时间步到batch大小
            timesteps = t.expand(current_image.shape[0]).to(self.device)

            # 模型预测
            model_output = self.net(current_image, timesteps, projs, projs_points_tensor)

            # 分离位置信息和噪声部分
            pos = current_image[:, 1:4, ...]
            noise_part = current_image[:, :1, ...]

            # 调度器步骤
            noise_pred = self.scheduler.step(
                model_output, t, noise_part, generator=generator
            )
            
            # 更新当前图像
            current_image = torch.cat([noise_pred.prev_sample, pos], dim=1)

        # 处理最终预测结果
        final_pred = noise_pred.pred_original_sample
        final_pred = (final_pred + 1. ) / 2.
        final_pred = final_pred.clamp(0, 1).cpu().numpy()
        #pdb.set_trace()
        # 获取ground truth
        #patch_image_tensor = patch_image_tensor.astype(torch.float32)
        patch_image_tensor = ( patch_image_tensor + 1. ) / 2.
        gt_image = patch_image_tensor.clamp(0,1).cpu().numpy()
        
        metrics_dict = calculate_metrics(final_pred , gt_image , data_range=1.0)
        print(f"\nQuality Metrics:")
        print(f"PSNR: {metrics_dict['psnr']:.4f}")
        print(f"SSIM: {metrics_dict['ssim']:.4f}")        # 保存结果（如果指定了保存路径）
        # 保存结果（如果指定了保存路径）
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            # 如果提供了names，使用name保存每个样本
            if names is not None:
                for idx, name in enumerate(names):
                    # 为每个样本创建子目录
                    sample_dir = os.path.join(save_path, name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 保存预测结果和ground truth
                    sitk_save(os.path.join(sample_dir, 'pred.nii.gz'), 
                             final_pred[idx:idx+1], uint8=True)
                    sitk_save(os.path.join(sample_dir, 'gt.nii.gz'), 
                             gt_image[idx:idx+1], uint8=True)
                    # 保存metrics到文本文件
            else:
                # 如果没有提供names，使用原来的保存方式
                sitk_save(os.path.join(save_path, 'pred.nii.gz'), final_pred, uint8=True)
                sitk_save(os.path.join(save_path, 'gt.nii.gz'), gt_image, uint8=True)

        if not return_dict:
            return final_pred, gt_image

        return TrainingInferenceOutput(
            pred_samples=final_pred,
            gt_samples=gt_image,
            metrics_log=metrics_dict,
            names = names
        )


@dataclass
class MyCustom3DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    image : np.ndarray 


class ProjectedBaseDDPMPipline(DiffusionPipeline):
    model_cpu_offload_seq = "net"

    def __init__(self , net , scheduler ):
        super().__init__()
        #pdb.set_trace()
        self.register_modules(net=net, scheduler=scheduler)
        self.weight_dtype = torch.float32
        self.projector = None
        self._device = None
    
    @property
    def device(self):
        if self._device is None:
            # Default to CUDA if available, otherwise CPU
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def set_device(self, device):
        self._device = device

    def init_projector(self, geo_cfg):
        with open(geo_cfg , 'r' ) as f :
            dst_cfg = yaml.safe_load(f)
            out_res = np.array(dst_cfg['dataset']['resolution'])
            self.projector = Geometry(dst_cfg['projector'])    

    def sample_patch_wise(self , batch_step , projs , clean_images , angles , generator,patch_size,save_intermediate_dir,save_step_interval):
        print(f"\n=== Starting sample_patch_wise ===")
        print(f"Input clean_images shape: {clean_images.shape}, dtype: {clean_images.dtype}")
        print(f"Input projs shape: {projs.shape}, dtype: {projs.dtype}")
        start_pos = (128 - 64) // 2 
        positions = {
            'i': start_pos ,
            'j': start_pos ,
            'k': start_pos ,
        } 
        #pdb.set_trace()
        print(f"Starting positions: {positions}")
        patch_image_tensor , patch_pos_tensor , projs_points_tensor =  pachify3d_projected_points(clean_images , patch_size , angles , self.projector , patch_pos=positions)
        pdb.set_trace()
        print(f"\nAfter pachify3d:")
        print(f"patch_image_tensor - shape: {patch_image_tensor.shape}, range: [{patch_image_tensor.min():.3f}, {patch_image_tensor.max():.3f}]")
        print(f"patch_pos_tensor - shape: {patch_pos_tensor.shape}, range: [{patch_pos_tensor.min():.3f}, {patch_pos_tensor.max():.3f}]")
        print(f"projs_points_tensor - shape: {projs_points_tensor.shape}, range: [{projs_points_tensor.min():.3f}, {projs_points_tensor.max():.3f}]")
        projs_points_tensor = projs_points_tensor.to(self.weight_dtype)
        b , c , h , w, d  = patch_image_tensor.shape
        image_shape = ( b , c ,h , w, d )
        image = randn_tensor(image_shape , generator=generator , device=self.device)
        print(f"\nInitial noise image - shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
        current_image = torch.concat([image, patch_pos_tensor] , dim=1)
        print(f"Initial current_image - shape: {current_image.shape}, range: [{current_image.min():.3f}, {current_image.max():.3f}]")
        pdb.set_trace()
        for step_idx , t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            t_ = t
            t = t.expand(current_image.shape[0]).to(self.device)
            #pdb.set_trace()
            model_output = self.net(current_image, t , projs , projs_points_tensor)

            pos = current_image[:,1:4,...]
            noise_part = current_image[:,:1,...]

            updated_noise  = self.scheduler.step(model_output, t_, noise_part, return_dict=True , generator=generator)
            #pdb.set_trace()
                    
            if save_intermediate_dir is not None and step_idx % save_step_interval == 0:
                pred_x0 = updated_noise.pred_original_sample
                # 转换到[0,1]范围
                #intermediate_image = (pred_x0 / 2 + 0.5).clamp(0, 1)
                #intermediate_image = intermediate_image.cpu().numpy()

                intermediate_image = (pred_x0 + 1) / 2 
                intermediate_image = intermediate_image.clamp(0,1).cpu().numpy()
            
                # 为每个batch样本保存一个文件
                save_path = os.path.join(
                        save_intermediate_dir, 
                        f'step_{step_idx:04d}_sample_{batch_step}.nii.gz')
                sitk_save(save_path, intermediate_image, uint8=True)
            current_image = torch.cat([updated_noise.prev_sample , pos], dim=1)


        pdb.set_trace()
        print("\n=== Final Processing ===")
        # current_image = (current_image / 2 + 0.5 ).clamp(0,1)
        # print(f"After normalization - shape: {current_image.shape}, range: [{current_image.min():.3f}, {current_image.max():.3f}]")
        # current_image = current_image[:,:1,...].cpu().numpy()
        # print(f"Final output - shape: {current_image.shape}, range: [{current_image.min():.3f}, {current_image.max():.3f}]")
        pdb.set_trace()
        final_pred = updated_noise.pred_original_sample
        final_pred = (final_pred + 1) / 2
        final_pred = final_pred.clamp(0, 1).cpu().numpy()
        gt_image  = patch_image_tensor.cpu().numpy() 
        print(f"GT image - shape: {gt_image.shape}, range: [{gt_image.min():.3f}, {gt_image.max():.3f}]")
        # 添加额外的验证检查
        if np.isnan(final_pred).any():
            print("WARNING: NaN values detected in output!")
        if np.isinf(final_pred).any():
            print("WARNING: Inf values detected in output!")
        return MyCustom3DOutput(image = final_pred)  , gt_image
    def sample_full_volume(self, batch_step , block_size , projs , clean_images , angles , generator, save_intermediate_dir, save_step_interval):  
        pdb.set_trace()
        image_blocks , pos_blocks , proj_points_tensor , blocks_info= full_volume_inference_process(clean_images , patch_size= 128 , block_size=16 , angles=angles , projector=self.projector)
        proj_points_tensor = proj_points_tensor.to(self.weight_dtype)
        # Get original volume shape from blocks_info
        num_blocks = image_blocks.shape[0]
        B, C, block_d, block_h, block_w = image_blocks.shape[1:]
        
        # Initialize noise for all blocks
        blocks_shape = image_blocks.shape
        noise_blocks = randn_tensor(blocks_shape, generator=generator, device=self.device)

        processed_blocks = []

        for block_idx in range(num_blocks):
            current_noise = noise_blocks[block_idx]
            current_pos = pos_blocks[block_idx]
            current_proj_points = proj_points_tensor[block_idx * B:(block_idx + 1) * B]

            # Combine noise and position information
            current_image = torch.cat([current_noise, current_pos], dim=1)

            for step_idx , t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                t_ = t 
                t = t.expand(current_image.shape[0]).to(self.device)
                
                model_output = self.net(current_image , t , projs , current_proj_points)
                
                # Split position and noise information
                pos = current_image[:, 1:4, ...]
                noise_part = current_image[:, :1, ...]
                
                # Update noise using scheduler
                updated_noise = self.scheduler.step(
                    model_output, t_, noise_part, 
                    return_dict=True, generator=generator
                )
                
                # Combine updated noise with position information
                current_image = torch.cat([updated_noise.prev_sample, pos], dim=1)

                # Process final block result
            final_block = (current_image[:, :1, ...] / 2 + 0.5).clamp(0, 1)
            processed_blocks.append(final_block) 


                # Stack all processed blocks
        all_blocks = torch.stack(processed_blocks, dim=0)
        
        # Reconstruct full volume
        # Calculate final volume dimensions based on block arrangement
        D = blocks_info['num_blocks_d'] * block_size
        H = blocks_info['num_blocks_h'] * block_size
        W = blocks_info['num_blocks_w'] * block_size
        
        # Initialize final volume tensor
        final_volume = torch.zeros((B, 1, D, H, W), device=self.device)
        
        # Place blocks back in their original positions
        block_idx = 0
        for d in range(blocks_info['num_blocks_d']):
            for h in range(blocks_info['num_blocks_h']):
                for w in range(blocks_info['num_blocks_w']):
                    d_start = d * block_size
                    h_start = h * block_size
                    w_start = w * block_size
                    
                    d_end = min((d + 1) * block_size, D)
                    h_end = min((h + 1) * block_size, H)
                    w_end = min((w + 1) * block_size, W)
                    
                    final_volume[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = \
                        all_blocks[block_idx]
                    
                    block_idx += 1

        return MyCustom3DOutput(image = final_volume) ,  clean_images.cpu().numpy()             
        
   
    @torch.no_grad()
    def __call__(
        self,
        batch,
        batch_step,
        num_inference_steps: int = 1000,
        sample_type:str = 'patch',
        patch_size:int = 16,
        block_size:int = 16,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True ,
        save_intermediate_dir: Optional[str] = None,
        save_step_interval: int = 100,  # 每隔多少步保存一次
        **kwargs,
    ):  
        pdb.set_trace()

        projs = batch['projs'].to(device=self.device, dtype=self.weight_dtype)
        clean_images = batch['gt_idensity'].to(device=self.device, dtype=self.weight_dtype)
        angles = batch['angles'].to(device=self.device)  # Keep angles in default dtype
        pdb.set_trace()
        
        self.scheduler.set_timesteps(num_inference_steps)
        if sample_type == 'patch':
            pred_image , gt_image = self.sample_patch_wise(batch_step , projs , clean_images , angles , generator, patch_size , save_intermediate_dir,save_step_interval)
            results = pred_image['image']

            return results , gt_image
        elif sample_type == 'full_volume':
            full_pred_image , gt_image = self.sample_full_volume(batch_step , block_size , projs , clean_images , angles , generator , save_intermediate_dir , save_step_interval)
            results = full_pred_image['image']
            return results , gt_image



        
        
class PosBaseDDPMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "net"

    def __init__(self , net , scheduler ):
        super().__init__()
        #pdb.set_trace()
        self.register_modules(net=net, scheduler=scheduler)
    
    @property
    def device(self):
        return self._device

    def set_device(self, device):
        self._device = device
    
    def sample_patch_wise(self , img_resolution , patch_size , start_pos):
        x_pos = torch.arange(start_pos, start_pos+patch_size).view(1, 1, -1).repeat(patch_size, patch_size, 1)
        y_pos = torch.arange(start_pos, start_pos+patch_size).view(1, -1, 1).repeat(patch_size, 1, patch_size)
        z_pos = torch.arange(start_pos, start_pos+patch_size).view(-1, 1, 1).repeat(1, patch_size, patch_size)
        
        x_pos = (x_pos / (img_resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (img_resolution - 1) - 0.5) * 2.
        z_pos = (z_pos / (img_resolution - 1) - 0.5) * 2.

        pos = torch.stack([x_pos, y_pos, z_pos], dim=0).to('cuda')
        return pos
    

    @torch.no_grad()
    def __call__(self, 
        sample_type: str = "patch",
        batch_size: int = 4,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        img_resolution: int = 128,
        patch_size: int = 32 , 
        start_pos: int = 32 ,
    )-> Union[MyCustom3DOutput ,Tuple]:
        pdb.set_trace()
        if sample_type == 'patch':
            image_shape = (batch_size,1,patch_size,patch_size,patch_size)
            pos = self.sample_patch_wise(img_resolution , patch_size , start_pos)
        elif sample_type == 'full_volume':
            assert batch_size == 1 and start_pos == 0 
            image_shape = (batch_size,1,img_resolution,img_resolution,img_resolution)
            pos = self.sample_patch_wise(img_resolution , img_resolution , start_pos)    
        # 归一化到[-1, 1]范围
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)
        #pdb.set_trace()
        pos = pos.unsqueeze(0).repeat(batch_size,1,1,1,1)    
        current_image = torch.concat([image ,pos] , dim=1)
        #pdb.set_trace()
        self.scheduler.set_timesteps(num_inference_steps)

        for step_idx , t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # 1. predict noise model_output
            #pdb.set_trace()
            t_ = t
            t = t.expand(current_image.shape[0]).to(self.device)
            #pdb.set_trace()
            model_output = self.net(current_image, t)
            # 只更新noise部分
            coords = current_image[:,1:,...]  # 保存坐标信息
            noise_part = current_image[:,:1,...]  # 获取noise部分
            # 2. compute previous image: x_t -> x_t-1
            updated_noise  = self.scheduler.step(model_output, t_, noise_part, generator=generator).prev_sample
            #pdb.set_trace()
            current_image = torch.cat([updated_noise ,coords], dim=1)
        #pdb.set_trace()
        current_image = (current_image / 2 + 0.5 ).clamp(0,1)
        current_image = current_image[:,:1,...].cpu().numpy()
        #pdb.set_trace()
        # 保存完整的分析结果
        #final_image = np.concatenate( current_image , axis=0)
        return MyCustom3DOutput(image = current_image) 


def projected_condition_process():
    model_config = 'F:/Code_Space/diffusers_v2/serv_ck_norm_size_0/checkpoint-154000/unet/config.json'
    safetensors_path = 'F:/Code_Space/diffusers_v2/serv_ck_norm_size_0/checkpoint-154000/unet/diffusion_pytorch_model.safetensors'
    infernect_time_steps = 1000
    save_path = './projected_condition/'
    save_sample = os.path.join(save_path, 'sample')
    save_gt_path = os.path.join(save_path, 'gt')
    save_intermediate_dir = os.path.join(save_path , 'intermediate_log')
    os.makedirs(save_path , exist_ok=True)
    os.makedirs(save_sample,exist_ok=True)
    os.makedirs(save_gt_path , exist_ok=True)
    os.makedirs(save_intermediate_dir , exist_ok=True)

    model = Song_Unet3D.from_config(model_config)
    check_points = safetensors.torch.load_file(safetensors_path)
    model.load_state_dict(check_points)
    #load training weight 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )
    pipline = ProjectedBaseDDPMPipline(model, noise_scheduler)
    pipline.to('cuda')
    image_root = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/'
    file_list_path = './files_name/pelvic_coord_train_16.txt'
    geo_config_path =  './geo_cfg/config_2d_128_s2.5_3d_128_2.0_25.yaml'
    pipline.init_projector(geo_config_path)

    dataset = load_diffusion_condition(image_root , file_list_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    generator = torch.Generator(device=pipline.device).manual_seed(0)
    pdb.set_trace()
    for step , batch in enumerate(train_dataloader):
        pdb.set_trace()
        # sample type "patch" and "full_volume"
        results , gt_image = pipline(batch ,step , num_inference_steps=infernect_time_steps , generator=generator , patch_size=64 ,sample_type='patch' ,save_intermediate_dir=save_intermediate_dir , save_step_interval=100)
        #results = images['image']
        pdb.set_trace()
        sitk_save(save_sample + f'/{step}.nii.gz' , results , uint8=True)
        sitk_save(save_gt_path+ f'/{step}.nii.gz' , gt_image, uint8=True)








def save_slice_image(images, save_path):
    N = images.shape[0]
    for i in range(N):
        print(f"Saving image {i + 1}/{N}...")
        np_image = images[i, 0, ...]  # Get the image data
        
        # Check if the image is 2D or 3D
        if len(np_image.shape) == 2:
            # For 2D images, add a dimension to make it 3D (depth=1)
            np_image = np_image[..., np.newaxis]
            
        nii_save_path = save_path
        sitk_save(nii_save_path, np_image, uint8=True)



def save_batch_image(images, save_path, inference_step, ts):
    N = images.shape[0]
    for i in range(N):
        print(f"Saving image {i + 1}/{N}...")
        np_image = images[i, 0, ...]  # Get the image data
        
        # Check if the image is 2D or 3D
        if len(np_image.shape) == 2:
            # For 2D images, add a dimension to make it 3D (depth=1)
            np_image = np_image[..., np.newaxis]
            
        nii_save_path = os.path.join(save_path, f"inference_{inference_step}timesteps_{ts}_{i}_sample_.nii.gz")
        sitk_save(nii_save_path, np_image, uint8=True)

def sitk_save(path, image, spacing=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    
    # Handle both 2D and 3D cases
    if len(image.shape) == 3:
        image = image.transpose(2, 1, 0)  # 3D case
    elif len(image.shape) == 2:
        # For 2D images, add a dimension and transpose
        image = image[..., np.newaxis]
        image = image.transpose(2, 1, 0)

    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)


if __name__ == '__main__':
    projected_condition_process()

    # model_config = 'output_log/overlap_blocks_ddpm/checkpoint-240000/unet/config.json'
    # safetensors_path = 'output_log/overlap_blocks_ddpm/checkpoint-240000/unet/diffusion_pytorch_model.safetensors' 
    # inference_time_steps = 500
    # save_path_root = f'./overlap_blocks_sample_{inference_time_steps}'  # Path to save the generated image
    # blocks_coords_path = '/root/share/pcc_gan_demo/cnetrilize_overlap_blocks_64/blocks/blocks_coords.npy'
    # ddpm_process_overlap_blocks(model_config , safetensors_path , blocks_coords_path ,inference_time_steps , save_path_root)


    #device = torch.device("cuda")
    #crop_shape = 64
    #ct_image_range = 256 
    #coords_crop = generator_coord( batch_size=4, crop_shape=crop_shape , device=device , ct_image_range=ct_image_range)
