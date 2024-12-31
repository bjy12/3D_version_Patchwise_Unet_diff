import torch
import torch.nn as nn
import os
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pickle
import safetensors
from model.network import PCC_Net
from model.SongUnet import Slice_UNet
from diffusers import DDPMPipeline, DDPMScheduler
import pdb
import SimpleITK as sitk
from datasets.datautils3d import load_with_coord
from datasets.slicedataset import load_slice_dataset
import matplotlib.pyplot as plt
import json
from datetime import datetime
from einops import rearrange
@dataclass
class MyCustom3DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    image : np.ndarray 

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




def ddpm_process_overlap_blocks(model_config ,safetensors_path , blocks_coords_path ,inference_time_steps , save_path_root):
    save_sample_path = os.path.join(save_path_root,'sample')
    save_gt_path = os.path.join(save_path_root,'gt')
    os.makedirs(save_path_root , exist_ok=True)
    os.makedirs(save_sample_path,exist_ok=True)
    os.makedirs(save_gt_path , exist_ok=True)

    model =  PCC_Net.from_config(model_config)
    check_points = safetensors.torch.load_file(safetensors_path)
    model.load_state_dict(check_points)
    #load training weight 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )
    pipeline = CustomDDPMPipeline(model , noise_scheduler)
    pipeline.to('cuda')
    if blocks_coords_path is not None: 
        overlap_blocks = np.load(blocks_coords_path)
        overlap_blocks = overlap_blocks.astype(np.float32)
        
    generator = torch.Generator(device=pipeline.device).manual_seed(0)

    #pdb.set_trace()
    for i in range(overlap_blocks.shape[0]-1):
        coords = overlap_blocks[i]
        coords = torch.from_numpy(coords)
        coords = coords.to('cuda')
        images , intermediate_results  = pipeline(generator=generator,batch_size=1,
                                                num_inference_steps=inference_time_steps, use_coord=True ,
                                                inference_type='overlap',overlap_coords=coords ,slice_index=None,
                                                output_type="np")
        #pdb.set_trace()
        pred_save_path =  os.path.join(save_sample_path , f"blocks_{i}_sample.nii.gz")
        pred_image = images['image']
        pred_image = pred_image.squeeze(0).squeeze(0)
        #pdb.set_trace()
        sitk_save( pred_save_path , pred_image , uint8=True)


def ddpm_process():
    model_config = '/root/codespace/diffusers_pcc/output_log/slice_traning/checkpoint-148000/unet/config.json'
    #model_path = 'F:/Code_Space/DiffusersExample/diffuser_pcc/output_log/ddpm_0/checkpoint-12000/random_states_0.pkl'  # Update this path to your model.pkl file
    #scheduler_path = 'F:/Code_Space/DiffusersExample/diffuser_pcc/output_log/ddpm_0/checkpoint-12000/scheduler.bin'  # Update this path to your scheduler.bin file
    safetensors_path = '/root/codespace/diffusers_pcc/output_log/slice_traning/checkpoint-148000/unet/diffusion_pytorch_model.safetensors'  # Update to your safetensors file
    inference_time_steps = 500 
    save_path_root = f'./slice_ck_148000_{inference_time_steps}'  # Path to save the generated image
    save_sample_path = os.path.join(save_path_root,'sample')
    save_gt_path = os.path.join(save_path_root,'gt')
    os.makedirs(save_path_root , exist_ok=True)
    os.makedirs(save_sample_path,exist_ok=True)
    os.makedirs(save_gt_path , exist_ok=True)
    #image_shape = (1, 3, 64, 64, 64)  # Adjust the shape based on your model (e.g., 3D image shape)
    #ckpt_path = 'F:\Code_Space\DiffusersExample\diffuser_pcc\output_log\ddpm_0\checkpoint-12000'
    #scheduler_path = DDPMScheduler.from_config(scheduler_path)
    #model =  PCC_Net.from_config(model_config)
    # 
    model = Slice_UNet.from_config(model_config)
    check_points = safetensors.torch.load_file(safetensors_path)
    model.load_state_dict(check_points)
    #load training weight 
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon'
    )
    pipeline = CustomDDPMPipeline(model , noise_scheduler)
    pipeline.to('cuda')
    image_root: str = '/root/share/slice_aixs_dataset'
    coord_root: str = '/root/share/pcc_gan_demo/pcc_gan_demo_centriod/train/coords'
    files_list_path: str = './files_list/p_train_demo.txt'
    overlap_blocks_path : str = '/root/share/pcc_gan_demo/cnetrilize_overlap_blocks_64/blocks/blocks_coords.npy'
    if overlap_blocks_path is not None: 
        overlap_blocks = np.load(overlap_blocks_path)

    batch_size = 1
    #dataset = load_with_coord(img_root=image_root , coord_root= coord_root, files_list_path=files_list_path )
    dataset = load_slice_dataset(img_root= image_root , files_list=  files_list_path)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    # pdb.set_trace()
    for step , batch in enumerate(train_dataloader):
        if step >= 50:  # 当达到100步时停止
            break
        #pdb.set_trace()
        #input_data = batch[0].to('cuda')
        #gt_image = batch[1].cpu().numpy()
        #pdb.set_trace()
        coords = batch['coords'].to('cuda')
        slice_gt = batch['value'].cpu().numpy()
        name = batch['name'][0]
        axes = batch['axes'][0]
        slice_number = batch['slice_number'][0]
        save_name = f"{name}_{axes}_{slice_number}.nii.gz"
        #pdb.set_trace()
        save_slice_sample_path = os.path.join(save_sample_path , save_name)
        save_slice_gt_path = os.path.join(save_gt_path , save_name)

        #pdb.set_trace()
        images , intermediate_results  = pipeline(generator=generator,batch_size=batch_size,inputs_tensor=coords,
                                                  num_inference_steps=inference_time_steps, use_coord=True ,image_size=256,
                                                  inference_type='slice_unet',slice_index=None,
                                                  output_type="np")
        denoised_image = images['image']
        #pdb.set_trace()
        save_slice_image(slice_gt, save_slice_gt_path)
        save_slice_image(denoised_image ,save_slice_sample_path)
        #pdb.set_trace()
        # save_batch_image(denoised_image , save_sample_path , inference_step=step,ts=inference_time_steps)
        #pdb.set_trace()
        # save_batch_image(slice_gt ,save_gt_path ,  inference_step=step,ts=inference_time_steps )
            # 保存图像
        #intermediate_dir =  os.path.join(save_path_root, f'batch_{step}')
        #save_intermediate_results(intermediate_results, intermediate_dir)

        #save_pred_from_noise(images, save_path, inference_step=step, ts=inference_time_steps)
        
        # 保存和可视化统计信息
        #save_and_plot_stats(stats_logs, save_path_root, step)
    #pdb.set_trace()
    #images = images['image']
    #pdb.set_trace()
    #sitk_save(output_path , images , uint8 =True)

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
    ddpm_process()
    

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
