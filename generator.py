import safetensors.torch
import torch 
import numpy as np
import os 
import sys
import safetensors

from patch_wise_generation import PosBaseDDPMPipeline
from model.Song3DUnet import Song_Unet3D
from diffusers import DDPMPipeline, DDPMScheduler
from utils import save_pred_to_local

import pdb



def load_model_config_and_checkpoints(config_path , ckpt_path):
    model = Song_Unet3D.from_config(config_path)
    ckpt = safetensors.torch.load_file(ckpt_path)
    model.load_state_dict(ckpt)

    return model 




def setting_DDPMPipline(model , num_train_timesteps , beta_schedule='linear' , prediction_type='epsilon'):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule = beta_schedule,
        prediction_type = prediction_type
    )
    # 检查模型
    print(f"Model device: {next(model.parameters()).device}")
    
    # 创建pipeline
    pipeline = PosBaseDDPMPipeline(model, noise_scheduler)
    print(f"Pipeline created: {pipeline}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    # 直接设置设备而不是使用to方法
    pipeline.net = pipeline.net.to(device)
    pipeline.set_device(device)
    print(f"Pipeline after device set: {pipeline}")
    
    # 创建生成器
    generator = torch.Generator(device=device).manual_seed(0)

    return pipeline , generator




def uncondition_gennerator( batch_size ,net_config_path , ckpt_path , num_train_timesteps ,save_path  ,img_res , patch_size , start_pos ,sample_type='patch'):
    model = load_model_config_and_checkpoints(net_config_path , ckpt_path)
    num_train_timesteps = 1000
    pipeline , generator = setting_DDPMPipline(model , num_train_timesteps)
    sample_pred = pipeline(
        sample_type=sample_type,
        generator=generator,
        batch_size=batch_size,
        num_inference_steps=num_train_timesteps,
        img_resolution= img_res,
        patch_size =patch_size,
        start_pos = start_pos
    )
    image_pred = sample_pred['image']
    save_pred_to_local(images=image_pred ,save_path = save_path)

if __name__ == "__main__":
    net_cfg = 'F:/Code_Space/diffusers_v2/checkpoint-78000/unet/config.json'
    ckpt_path = 'F:/Code_Space/diffusers_v2/checkpoint-78000/unet/diffusion_pytorch_model.safetensors'
    save_path = './ckpt_results'
    uncondition_gennerator(batch_size=5 ,
                           net_config_path= net_cfg,
                           ckpt_path=ckpt_path ,
                           num_train_timesteps=1000 ,
                           save_path= save_path,
                           img_res = 128 ,
                           patch_size = 64 ,
                           start_pos = 32 ,
                           sample_type='patch')

    # uncondition_gennerator(batch_size=1 ,
    #                        net_config_path= net_cfg,
    #                        ckpt_path=ckpt_path ,
    #                        num_train_timesteps=1000 ,
    #                        save_path= save_path,
    #                        img_res = 128 ,
    #                        patch_size= 32 ,
    #                        start_pos= 0 ,
    #                        sample_type='full_volume')    


