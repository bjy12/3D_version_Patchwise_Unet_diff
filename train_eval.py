from dataclasses import dataclass
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from typing import Dict, List, Optional ,Tuple
from pachify_and_projector import pachify3d_projected_points
from metrics import calculate_metrics_each_names
import torch
import numpy as np
from tqdm.auto import tqdm
from utils import sitk_save
import os
import shutil
from accelerate.logging import get_logger


import pdb
logger = get_logger(__name__)

@dataclass
class TestEvaluationOutput(BaseOutput):
    """Output class for test set evaluation results"""
    metrics: Dict[str, float]  # Average metrics across all test samples
    sample_metrics: List[Dict[str, float]]  # Individual sample metrics
    best_so_far: bool  # Whether this evaluation produced new best results
    improved_metric: Optional[str] = None  # Which metric improved (if any)

class TestEvaluationPipeline(DiffusionPipeline):
    """Pipeline for evaluating model performance on test set during training"""
    
    def __init__(
        self, 
        net,
        scheduler,
        accelerator,
        cfg,
        projector
    ):
        super().__init__()
        self.register_modules(net=net, scheduler=scheduler)
        self.accelerator = accelerator
        self.cfg = cfg
        self.projector = projector
        self.weight_dtype = (torch.float16 if accelerator.mixed_precision == "fp16" 
                           else torch.bfloat16 if accelerator.mixed_precision == "bf16"
                           else torch.float32)

    def save_best_checkpoint(
        self,
        metrics: Dict[str, float],
        best_metrics: Dict[str, float],
        epoch: int,
        global_step: int,
        ema_model=None
    ) -> Tuple[bool, Optional[str]]:
        """
        Save checkpoint if current metrics are better than best metrics
        Returns:
        - bool: Whether model improved
        - str or None: Which metric improved
        """
        improved = False
        improved_metric = None

        if metrics['psnr'] > best_metrics.get('psnr', 0):
            improved = True
            improved_metric = 'psnr'
            best_metrics['psnr'] = metrics['psnr']
            
            save_dir = os.path.join(self.cfg.output_dir, 'best_psnr_checkpoint')
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
            if ema_model is not None:
                ema_model.store(self.net.parameters())
                ema_model.copy_to(self.net.parameters())
                
            self.accelerator.save_state(save_dir)
            
            if ema_model is not None:
                ema_model.restore(self.net.parameters())
                
            logger.info(f"Saved new best PSNR checkpoint: {metrics['psnr']:.4f}")
            
        if metrics['ssim'] > best_metrics.get('ssim', 0):
            improved = True
            improved_metric = 'ssim' if improved_metric is None else 'both'
            best_metrics['ssim'] = metrics['ssim']
            
            save_dir = os.path.join(self.cfg.output_dir, 'best_ssim_checkpoint')
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
            if ema_model is not None:
                ema_model.store(self.net.parameters())
                ema_model.copy_to(self.net.parameters())
                
            self.accelerator.save_state(save_dir)
            
            if ema_model is not None:
                ema_model.restore(self.net.parameters())
                
            logger.info(f"Saved new best SSIM checkpoint: {metrics['ssim']:.4f}")

        if improved:
            self.save_metrics_history(best_metrics, epoch, global_step)
            
        return improved, improved_metric

    def save_metrics_history(
        self,
        metrics: Dict[str, float],
        epoch: int,
        global_step: int
    ):
        """Save metrics history to file"""
        metrics_file = os.path.join(self.cfg.output_dir, 'best_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Best PSNR: {metrics.get('psnr', 0):.4f}\n")
            f.write(f"Best SSIM: {metrics.get('ssim', 0):.4f}\n")
            f.write(f"Achieved at epoch {epoch}, step {global_step}\n")
    def save_test_images(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        names: List[str],
        epoch: int,
        save_dir: str
    ):
        """
        Save predicted and ground truth images from test set
        
        Args:
            pred: Predicted images (B, C, D, H, W)
            gt: Ground truth images (B, C, D, H, W)
            names: List of sample names
            epoch: Current epoch number
            save_dir: Base directory for saving images
        """
        # Create epoch-specific directory
        epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
        pred_dir = os.path.join(epoch_dir, 'predictions')
        gt_dir = os.path.join(epoch_dir, 'ground_truth')
        
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # Save each sample
        for idx, name in enumerate(names):
            # Get single sample volumes (removing batch and channel dims)
            pred_vol = pred[idx, 0]  # (D, H, W)
            gt_vol = gt[idx, 0]  # (D, H, W)

            # Save prediction
            pred_path = os.path.join(pred_dir, f'{name}.nii.gz')
            sitk_save(
                path=pred_path,
                image=pred_vol,
                uint8=True  # Convert to uint8 since input is in [0,1] range
            )

            # Save ground truth
            gt_path = os.path.join(gt_dir, f'{name}.nii.gz')
            sitk_save(
                path=gt_path,
                image=gt_vol,
                uint8=True  # Convert to uint8 since input is in [0,1] range
            )

        logger.info(f"Saved test images for epoch {epoch} to {epoch_dir}")
    @torch.no_grad()
    def __call__(
        self,
        test_dataloader,
        epoch: int,
        global_step: int,
        best_metrics: Dict[str, float],
        ema_model = None,
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
    ) -> TestEvaluationOutput:
        """
        Run evaluation on test dataset
        """
        logger.info("Starting test set evaluation...")
        self.net.eval()
        
        if ema_model is not None:
            ema_model.store(self.net.parameters())
            ema_model.copy_to(self.net.parameters())

        # Set up noise scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        sample_metrics = []
        # Use the same vis_dir path as in main script
        vis_dir = os.path.join(self.cfg.output_dir, 'vis_res')
        save_dir = os.path.join(vis_dir, f'epoch_{epoch}_step_{global_step}')
        os.makedirs(save_dir, exist_ok=True)
        # Initialize progress bar
        progress_bar = tqdm(total=len(test_dataloader), desc="Evaluating test set")
        
        for batch_idx, batch in enumerate(test_dataloader):
            # Process batch data
            projs = batch['projs'].to(device=self.accelerator.device, dtype=self.weight_dtype)
            clean_images = batch['gt_idensity'].to(device=self.accelerator.device, dtype=self.weight_dtype)
            angles = batch['angles'].to(device=self.accelerator.device)  # Move angles to device
            names = batch['name']

            # Get patch data
            patch_image_tensor, patch_pos_tensor, projs_points_tensor = pachify3d_projected_points(
                clean_images, 
                self.cfg.pachify_size, 
                angles, 
                self.projector,
                patch_pos={
                        'i': self.cfg.patch_start_i,
                        'j': self.cfg.patch_start_j,
                        'k': self.cfg.patch_start_k
                    }
            )
            
            # Move tensors to correct device and dtype
            patch_image_tensor = patch_image_tensor.to(device=self.accelerator.device, dtype=self.weight_dtype)
            patch_pos_tensor = patch_pos_tensor.to(device=self.accelerator.device, dtype=self.weight_dtype)
            projs_points_tensor = projs_points_tensor.to(device=self.accelerator.device, dtype=self.weight_dtype)
            
            # Initialize noise
            noise = torch.randn_like(patch_image_tensor)
            noisy_images = torch.cat([noise, patch_pos_tensor], dim=1)

            # Denoise
            for t in self.scheduler.timesteps:
                # Expand timestep and ensure it's on the correct device
                timesteps = t.expand(patch_image_tensor.shape[0]).to(device=self.accelerator.device)
                
                # Model prediction
                model_output = self.net(noisy_images, timesteps, projs, projs_points_tensor)
                
                # Split position and noise components
                pos = noisy_images[:, 1:4, ...]
                noise_part = noisy_images[:, :1, ...]
                
                # Scheduler step
                scheduler_output = self.scheduler.step(model_output, t, noise_part)
                noisy_images = torch.cat([scheduler_output.prev_sample, pos], dim=1)

            # Process results
            final_pred = (scheduler_output.pred_original_sample + 1.) / 2.
            final_pred = final_pred.clamp(0, 1).cpu().numpy()
            
            gt_image = patch_image_tensor.cpu().numpy()
            gt_image = (gt_image + 1.) / 2.

            # Convert to uint8 for metrics calculation
            final_pred = (final_pred * 255.).astype(np.uint8)
            gt_image = (gt_image * 255.).astype(np.uint8)

            # Calculate metrics for each sample
            batch_metrics = calculate_metrics_each_names(
                pred=final_pred,
                target=gt_image,
                names=names,
                data_range=255
            )
            sample_metrics.extend(batch_metrics)

            # Log individual sample metrics
            for metric in batch_metrics:
                logger.info(
                    f"Sample {metric['name']}: "
                    f"PSNR = {metric['psnr']:.4f}, "
                    f"SSIM = {metric['ssim']:.4f}"
                )
            # Save images
            pred_dir = os.path.join(save_dir, 'predictions')
            gt_dir = os.path.join(save_dir, 'ground_truth')
            
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            # Save each sample
            for idx, name in enumerate(names):
                pred_vol = final_pred[idx, 0]
                gt_vol = gt_image[idx, 0]

                # Save prediction
                pred_path = os.path.join(pred_dir, f'{name}.nii.gz')
                sitk_save(
                    path=pred_path,
                    image=pred_vol,
                    uint8=False
                )

                # Save ground truth
                gt_path = os.path.join(gt_dir, f'{name}.nii.gz')
                sitk_save(
                    path=gt_path,
                    image=gt_vol,
                    uint8=False
                )

            logger.info(f"Saved test images for batch {batch_idx} to {save_dir}")

            progress_bar.update(1)

        progress_bar.close()

        # Calculate average metrics
        avg_metrics = {
            'psnr': sum(m['psnr'] for m in sample_metrics) / len(sample_metrics),
            'ssim': sum(m['ssim'] for m in sample_metrics) / len(sample_metrics)
        }

        # Log average metrics
        logger.info(
            f"Test Set Evaluation Results - "
            f"Avg PSNR: {avg_metrics['psnr']:.4f}, "
            f"Avg SSIM: {avg_metrics['ssim']:.4f}"
        )

        # Save checkpoints if improved
        improved, improved_metric = self.save_best_checkpoint(
            avg_metrics, 
            best_metrics,
            epoch,
            global_step,
            ema_model
        )

        # Restore original model if using EMA
        if ema_model is not None:
            ema_model.restore(self.net.parameters())

        # Log to tensorboard if available
        if self.accelerator.is_main_process and hasattr(self, 'tracker'):
            self.tracker.add_scalar("test/avg_psnr", avg_metrics['psnr'], global_step)
            self.tracker.add_scalar("test/avg_ssim", avg_metrics['ssim'], global_step)

        return TestEvaluationOutput(
            metrics=avg_metrics,
            sample_metrics=sample_metrics,
            best_so_far=improved,
            improved_metric=improved_metric
        )