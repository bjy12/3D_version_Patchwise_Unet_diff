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
    valid_epochs: int = 100
    valid_batch_size: int = 1
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None

    # Diffuion Models
    model_config: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100

    #pcc_net models
    layers: list = field(default_factory=lambda: [1, 1, 1, 1])  # 使用 default_factory
    embed_dims: list = field(default_factory=lambda: [64, 128, 256, 512])  # 64, 128, 256, 512
    mlp_ratios: list = field(default_factory=lambda: [8, 8, 4, 4])
    heads: list = field(default_factory=lambda: [4, 4, 8, 8])
    head_dim: list = field(default_factory=lambda: [24, 24, 24, 24])
    with_coord: bool = True
    time_embed_dims: list = field(default_factory=lambda: [16])
    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 1


    unet_config: Dict = field(default_factory=lambda:{
        'embedding_type': 'positional',
        'encoder_type': 'standard',
        'decoder_type':'standard',
        'channel_mult_noise': 1,
        'resample_filter':[1,1],
        'model_channels': 128, 
        'channel_mult':[2,2,2],
        'img_resolution':256,
        'in_channels': 4, 
        'out_channels': 1, # 1 + 3 
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
    eval_vis: bool = False
    # Dataset
    dataset_name: str = None
    #image_root: str = 'F:/Data_Space/Pelvic1K/pcc_gan_demo/pcc_gan_demo_coords_test/train/img'
    image_root: str = '/root/share/slice_aixs_dataset'
    coord_root: str = 'F:/Data_Space/Pelvic1K/pcc_gan_demo/pcc_gan_demo_coords_test/train/coords'
    files_list_path: str = './files_list/p_train_demo.txt'


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
