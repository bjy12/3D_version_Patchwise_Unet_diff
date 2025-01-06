import numpy as np
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
import torch
from typing import Dict, Union, Tuple
from utils import sitk_load
import logging
import pdb

logger = logging.getLogger(__name__)




def calculate_metrics(pred: Union[np.ndarray, torch.Tensor], 
                     target: Union[np.ndarray, torch.Tensor],
                     data_range: float = 1.0) -> Dict[str, float]:
    """
    Calculate PSNR and SSIM metrics between prediction and target images.
    
    Args:
        pred: Predicted image array/tensor (B, C, D, H, W) or (C, D, H, W)
        target: Ground truth image array/tensor (B, C, D, H, W) or (C, D, H, W)
        data_range: Data range of the input (default: 1.0 for normalized images)
        
    Returns:
        Dictionary containing PSNR and SSIM values
    """
    # Convert torch tensors to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    #pdb.set_trace()    
    # Ensure 5D format (B, C, D, H, W)
    if pred.ndim == 4:
        pred = pred[np.newaxis, ...]
    if target.ndim == 4:
        target = target[np.newaxis, ...]
        
    batch_size = pred.shape[0]
    total_psnr = 0
    total_ssim = 0
    
    # Calculate metrics for each sample in batch
    for i in range(batch_size):
        pred_sample = pred[i, 0]  # Take first channel only
        target_sample = target[i, 0]
        
        # Calculate PSNR
        psnr_val = ski_psnr(target_sample, 
                           pred_sample, 
                           data_range=data_range)
        
        # Calculate SSIM
        ssim_val = ski_ssim(target_sample,
                           pred_sample,
                           data_range=data_range,
                           channel_axis=None)  # No channel axis for 3D
        
        total_psnr += psnr_val
        total_ssim += ssim_val
        
    # Average over batch
    avg_psnr = total_psnr / batch_size
    avg_ssim = total_ssim / batch_size
    
    metrics = {
        'psnr': float(avg_psnr),
        'ssim': float(avg_ssim)
    }
    
    logger.info(f"Quality Metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    return metrics




if __name__ == '__main__':
    data_path = '/root/share/processed_128x128_s2/images/0103.nii.gz'
    img , space = sitk_load(data_path, uint8=True)
    pdb.set_trace()
    img_1 = img.copy()
    metrics = calculate_metrics(img , img_1)
    print("\nTest with identical images:")
    print(f"PSNR: {metrics['psnr']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    # 测试添加噪声的图像
    img_2 = img.copy()
    noise = np.random.normal(0, 0.1, img.shape)
    img_2 = img_2 + noise
    img_2 = np.clip(img_2, 0, 1)  # 确保值在[0,1]范围内
    
    metrics_noisy = calculate_metrics(img, img_2)
    print("\nTest with noisy images:")
    print(f"PSNR: {metrics_noisy['psnr']:.4f}")
    print(f"SSIM: {metrics_noisy['ssim']:.4f}")



