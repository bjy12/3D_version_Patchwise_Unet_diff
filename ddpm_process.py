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
class MyCustom2DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    image : np.ndarray 

class CustomDDPMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "net"

    def __init__(self , net , scheduler ):
        super().__init__()
        #pdb.set_trace()
        self.register_modules(net=net, scheduler=scheduler)


    def coord_maker(self , slice_index , image_size , batch_size):
        range_h =   np.arange(0, image_size, step=1) / ( image_size - 1.0)
        range_w =   np.arange(0, image_size, step=1) / ( image_size - 1.0)
        range_d =   np.arange(0, image_size, step=1) / ( image_size - 1.0)
        pdb.set_trace()
        coords = np.stack(np.meshgrid(range_h,range_w,range_d) , axis=-1)
        coords = coords.astype(np.float32)
        pdb.set_trace()
        coords = coords[:16,:,:,:]
        coords = np.transpose(coords , (3,0,1,2))
        coords = torch.from_numpy(coords).to(self.device)

        coords = coords - 0.5

        coords = coords.unsqueeze(0).repeat(batch_size,1,1,1,1)

        return coords
    
    def full_inference_coords(self , image_size , batch_size):
        pdb.set_trace()
        h =  image_size 
        # normalize to [-1 ,1]
        range_h = 2 * (np.arange(0, h, step=1) / (h - 1.0)) - 1
        range_w = 2 * (np.arange(0, h, step=1) / (h - 1.0)) - 1
        range_d = 2 * (np.arange(0, h, step=1) / (h - 1.0)) - 1
        #pdb.set_trace()
        coords = np.stack(np.meshgrid(range_h,range_w,range_d, indexing='ij') , axis=-1)
        coords = coords.astype(np.float32)
        pdb.set_trace()
        coords = torch.from_numpy(coords).to(self.device)
        coords = rearrange(coords ," h w d c -> c h w d ")
        coords = coords.unsqueeze(0).repeat(batch_size,1,1,1,1)      
        return coords
    def overlap_blocks_inference_coords(self, overlap_coords_path):
        blocks = np.load(overlap_coords_path)
        blocks = blocks.astype(np.float32)
        blocks = torch.from_numpy(blocks)
        blocks = blocks.to(self.device)
        blocks = rearrange(blocks , "b h w d c -> b c h w d ")
        pdb.set_trace()
        return blocks
    def slice_axis_flow_coords(self , image_size , axis):
        pdb.set_trace()
        range_ct = np.arange(image_size)
        coords = np.stack(np.meshgrid(range_ct, range_ct, range_ct, indexing='ij'), axis=0)  # 3 x 128 x 128 x 128
        slices = []
        for slice_idx in range(image_size):
            if axis == 'axial':
                slice_coord = coords[:,slice_idx,:,:]
            elif axis == 'sagittal':
                slice_coord = coords[:,:,slice_idx,:]
            else:
                slice_coord = coords[:,:,:,slice_idx]
            slices.append(slice_coord)
        pdb.set_trace()
        slices = np.stack(slices, axis=0)  # 3 x 128 x 128 x 128
        slices = slices / (image_size - 1)
        slices = slices.astype(np.float32)
        slices = (slices - 0.5) * 2
        pdb.set_trace()
        return slices

    @torch.no_grad()
    def __call__(self, 
        batch_size: int = 4,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        inputs_tensor: Optional[torch.Tensor] = None,
        use_coord: bool = False,
        image_size: int = 256 , 
        inference_type : str = 'full', 
        overlap_coords: Optional[torch.Tensor] = None,
        slice_index: int = 128  ,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    )-> Union[MyCustom2DOutput ,Tuple]:
        #if isinstance(self.net.config.sample_size , int):
        #pdb.set_trace()
        if inference_type == 'slice' :
            coords = self.coord_maker(slice_index, image_size, batch_size)
            #pdb.set_trace()
            image_shape = (batch_size , 1 , coords.shape[2] , coords.shape[3], coords.shape[4])
        elif inference_type == 'full' :
            coords = self.full_inference_coords(image_size , batch_size)
            image_shape = (batch_size , 1 , image_size , image_size , image_size)
        elif inference_type == 'random' :
            coords = inputs_tensor[:,:3,...]
            image_shape = (
                batch_size,
                1,
                self.net.config.sample_size,
                self.net.config.sample_size,
                self.net.config.sample_size,
            )
        elif inference_type == 'overlap' : 
            coords = overlap_coords
            #pdb.set_trace()
            coords = coords.unsqueeze(0)
            coords = rearrange(coords , 'b h w d c -> b c h w d')
            h = coords.shape[2]
            batch_size = coords.shape[0]
            image_shape =  (batch_size , 1 , h ,h,h)
            #pdb.set_trace()
        elif inference_type == 'slice_unet':
            #pdb.set_trace()
            b , _ ,  h ,w  = inputs_tensor.shape
            coords = inputs_tensor
            image_shape = ( b, 1 , h , w )
        elif inference_type == 'slices_flow_axis':
            #pdb.set_trace()
            _ , _ , h , w = inputs_tensor.shape
            image_size = h 
            coords = self.slice_axis_flow_coords(image_size ,axis='sagittal')
            b , _ , _ , _ = coords.shape
            coords = torch.from_numpy(coords).to(self.device)
            image_shape = (b , 1 , h , w)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)
        #pdb.set_trace()
        if use_coord:
              #coords = inputs_tensor[:,:3,...]
              #pdb.set_trace()
              image = torch.concat([coords , image] , dim=1)
        if save_intermediate_results:
                intermediate_results = []
        #pdb.set_trace()
        self.scheduler.set_timesteps(num_inference_steps)
        all_images = []
        total_batches = image.shape[0]
        samples_per_batch = 10
        num_rounds = (total_batches + samples_per_batch - 1 ) // samples_per_batch

        for round_idx in range(num_rounds): 
            start_idx = round_idx * samples_per_batch
            end_idx = min(start_idx + samples_per_batch, total_batches)
            
            # 获取当前批次的图像和坐标
            current_image = image[start_idx:end_idx].clone()
            for step_idx , t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # 1. predict noise model_output
                #pdb.set_trace()
                t_ = t
                t = t.expand(current_image.shape[0]).to(self.device)
                #pdb.set_trace()
                model_output = self.net(current_image, t)
                # 只更新noise部分
                coords = current_image[:,:3,...]  # 保存坐标信息
                noise_part = current_image[:,3:,...]  # 获取noise部分
                #pdb.set_trace()
                # 2. compute previous image: x_t -> x_t-1
                # 每隔一定步数或在关键时间点保存中间结果
                updated_noise  = self.scheduler.step(model_output, t_, noise_part, generator=generator).prev_sample
                #pdb.set_trace()
                current_image = torch.cat([coords, updated_noise], dim=1)
            #pdb.set_trace()
            current_image = (current_image / 2 + 0.5 ).clamp(0,1)
            current_image = current_image[:,3:4,...].cpu().numpy()
            all_images.append(current_image)
        #pdb.set_trace()
        # 保存完整的分析结果
        final_image = np.concatenate( all_images , axis=0)
        return MyCustom2DOutput(image = final_image) , intermediate_results

def save_intermediate_results(intermediate_results, save_dir):
    """保存中间生成结果
    
    Args:
        intermediate_results (list): 包含中间结果的列表
        save_dir (str): 保存目录的路径
    """
    intermediate_dir = os.path.join(save_dir, 'intermediate_results')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    for result in intermediate_results:
        step_idx = result['step_idx']
        timestep = result['timestep']
        
        # 为当前时间步创建目录
        step_dir = os.path.join(intermediate_dir, f'step_{step_idx:04d}_t{timestep:04d}')
        os.makedirs(step_dir, exist_ok=True)
        
        # 保存噪声图像和预测噪声
        noisy_images = result['noisy_image']
        pred_noises = result['predicted_noise']
        
        for b in range(noisy_images.shape[0]):
            # 保存噪声图像
            noisy_path = os.path.join(step_dir, f'noisy_sample_{b}.nii.gz')
            sitk_save(noisy_path, noisy_images[b, 0], uint8=True)
            
            # 保存预测的噪声
            pred_path = os.path.join(step_dir, f'pred_noise_{b}.nii.gz')
            sitk_save(pred_path, pred_noises[b, 0], uint8=False)
        
        # 保存统计信息
        stats_path = os.path.join(step_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(result['stats'], f, indent=4)

def generator_coord(batch_size , crop_shape , device , ct_image_range):
    range_ = np.arange(0, ct_image_range , step=1)
    grid = np.meshgrid(range_ , range_ , range_)
    coords = np.stack(grid,axis=0)
    pdb.set_trace()
    coord_index = coords.reshape(3,-1)
    coords_norm = coord_index / (ct_image_range - 1)
    coords_norm = coords_norm - 0.5
    coords_norm = coords_norm.astype(np.float32)
    coords_norm = coords_norm.reshape(3,ct_image_range , ct_image_range , ct_image_range)
    coords_list = []
    for b in range(batch_size):
        d = np.random.randint(0, ct_image_range - crop_shape)
        h = np.random.randint(0, ct_image_range - crop_shape)
        w = np.random.randint(0, ct_image_range - crop_shape)
        crop_coords = coords_norm[:,d:d+crop_shape,h:h+crop_shape,w:w+crop_shape]
        crop_coords = torch.from_numpy(crop_coords).to(device)
        coords_list.append(crop_coords)
    coords_tensor = torch.stack(coords_list , dim=0)    
    #pdb.set_trace()
    return coords_tensor





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
