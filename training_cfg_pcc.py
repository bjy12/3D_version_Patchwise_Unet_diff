from dataclasses import dataclass , field
from typing import Any, Dict, List, Optional


@dataclass
class BaseTrainingConfig:
    # Dir
    logging_dir: str
    output_dir: str

    # Logger and checkpoint
    logger: str = 'tensorboard'
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 20
    # vis val
    valid_epochs: int = 1
    valid_batch_size: int = 1
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None
    img_resolution: int = 128 
    patch_size: int = 32
    start_pos: int = 64

    # Diffuion Models
    model_config: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 500

    #unet_setting 
    unet_config: Dict = field(default_factory=lambda:{
        'embedding_type': 'positional',
        'encoder_type': 'standard',
        'decoder_type':'standard',
        'channel_mult_noise': 1,
        'resample_filter':[1,1],
        'model_channels': 128, 
        'channel_mult':[1,2,2],
        'channel_mult_emb': 2,
        'num_blocks': 2,
        'attn_resolutions': [16],
        'img_resolution':128,
        'in_channels': 4, 
        'out_channels': 1, # 1 + 3 
        'condition_mixer_out_channels': 64,
        'implict_condition_dim' : 128,
        #image_enocer_setting
        'image_encoder_output': 64,
        'bilinear': False,
        #implict_func_model setting 
        'pos_dim': 63 ,
        'local_f_dim':64 , 
        'num_layer': 4 ,
        'hidden_dim': 256 ,
        'output_dim': 128 ,
        'skips': [2] ,
        'last_activation': 'relu',
        'use_silu':False , 
        'no_activation':False, 
    })  

    # Training
    seed: int = None
    num_epochs: int = 200
    train_batch_size: int = 2
    dataloader_num_workers: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False
    eval_vis: bool = True
    # Dataset
    dataset_name: str = None
    #image_root: str = 'F:/Data_Space/Pelvic1K/pcc_gan_demo/pcc_gan_demo_coords_test/train/img'
    image_root: str = 'F:/Data_Space/Pelvic1K/processed_128x128_s2/'
    coord_root: str = 'F:/Data_Space/Pelvic1K/pcc_gan_demo/pcc_gan_demo_coords_test/train/coords'
    files_list_path: str = './files_name/pelvic_coord_train_16.txt'
    geo_cfg_path: str = './geo_cfg/config_2d_128_s2.5_3d_128_2.0_25.yaml'
    
    # 
    use_multi_patch_size: bool = False  
    pachify_size: int = 64

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # EMA
    use_ema: bool = False
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4

    # Hub
    push_to_hub: bool = False
    hub_model_id: str = ''
