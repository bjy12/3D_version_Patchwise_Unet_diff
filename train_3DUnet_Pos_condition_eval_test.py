import argparse
import inspect
import logging
import math
import os
import shutil
from pathlib import Path
import sys
import accelerate
import datasets
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets.datautils3d import  load_diffusion_condition
#from huggingface_hub import create_repo, upload_folder
from packaging import version
#from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf
#from model.Song3DUnet import Song_Unet3D
#from model.Song3DUnetV3 import Song_Unet3D
from model.Song3DUnetV4 import Song_Unet3D
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from patch_wise_generation import TrainingInferencePipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

#from ddpm_process import save_pred_from_noise
from pachify_and_projector import pachify3D , init_projector ,pachify3d_projected_points
#from training_cfg_pcc import BaseTrainingConfig
from training_cfg_resnetbackbone import BaseTrainingConfig
from utils import save_pred_to_local
from train_eval import TestEvaluationPipeline
import pdb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    data_dict = OmegaConf.load(args.cfg)
    cfg = BaseTrainingConfig(**data_dict)
    #pdb.set_trace()
    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    vis_dir = os.path.join(cfg.output_dir , 'vis_res')
    os.makedirs(vis_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.logger,
        project_config=accelerator_project_config
    )

    if cfg.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training.")

    elif cfg.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if cfg.use_ema:
                    ema_model.save_pretrained(
                        os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), Song_Unet3D)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = Song_Unet3D.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        #datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    #pdb.set_trace()
    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

        # if cfg.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=cfg.hub_model_id or Path(cfg.output_dir).name, exist_ok=True, token=cfg.hub_token
        #     ).repo_id
    #   
    # Initialize the model
    #pdb.set_trace()
    # model = PCC_Net(
    #     layers = cfg.layers,  # 2, 2, 2, 2
    #     embed_dims= cfg.embed_dims ,  # 64, 128, 256, 512
    #     mlp_ratios= cfg.mlp_ratios,
    #     heads = cfg.heads,
    #     head_dim = cfg.head_dim,
    #     with_coord = cfg.with_coord,
    #     time_embed_dims= cfg.time_embed_dims,
    #     sample_size=cfg.sample_size,
    #     in_channels=cfg.in_channels,
    #     out_channels=cfg.out_channels
    # )
    #pdb.set_trace()
    model = Song_Unet3D(
        **cfg.unet_config,
    )


    # Create EMA for the model.
    if cfg.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=cfg.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.ema_inv_gamma,
            power=cfg.ema_power,
            model_cls=Song_Unet3D,
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision

    # if cfg.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warning(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         model.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError(
    #             "xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(
        inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.ddpm_num_steps,
            beta_schedule=cfg.ddpm_beta_schedule,
            prediction_type=cfg.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.ddpm_num_steps, beta_schedule=cfg.ddpm_beta_schedule)

    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps *
            cfg.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # pdb.set_trace()
    
    if cfg.dataset_name is not None:
        # dataset = load_with_coord(
        #     img_root= cfg.image_root,
        #     coord_root= cfg.coord_root,
        #     files_list_path= cfg.files_list_path
        # )
        #dataset = load_diffusion_random_blocks(cfg.image_root , cfg.files_list_path)
        #dataset = load_diffusion_overlap_blocks(cfg.image_root , cfg.files_list_path)
        # dataset = load_slice_dataset(cfg.image_root ,  cfg.files_list_path )
        dataset = load_diffusion_condition(cfg.image_root , cfg.files_list_path)
        test_dataset = load_diffusion_condition(cfg.image_root , cfg.test_list_path)
    #else:
        # dataset = load_with_coord(
        #     "imagefolder", data_dir=cfg.train_data_dir, cache_dir=cfg.cache_dir, split="train")
        # # See more about loading custom images at
        # # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    # init_projector will use on from 3d points to 2d image 
    projector = init_projector(cfg.geo_cfg_path)
   
    logger.info(f"Dataset size: {len(dataset)}")

    #dataset.set_transform(transform_images)
    #pdb.set_trace()
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.dataloader_num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * cfg.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader,test_dataloader ,lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader,test_dataloader  , lr_scheduler
    )

    if cfg.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    test_pipeline = TestEvaluationPipeline(
        net=model,
        scheduler=noise_scheduler,
        accelerator=accelerator,
        cfg=cfg,
        projector=projector
    )
    best_metrics = {'psnr':0 , 'ssim':0}
    
    total_batch_size = cfg.train_batch_size * \
        accelerator.num_processes * cfg.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.num_epochs * num_update_steps_per_epoch
    #pdb.set_trace()
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * cfg.gradient_accumulation_steps)

    # Train!
    # Train progressive patch size     
    if cfg.use_multi_patch_size:
        batch_mul_dict = {64:2 , 32:2 , 16:2 }
        real_p = 0.5 # 选择每个patch size的概率
        p_list = np.array([(1-real_p)*2/5, (1-real_p)*3/5, real_p])
        img_resolution = 128
        #patch_list = np.array([img_resolution//4, img_resolution//2, img_resolution])  # 32  64 128 
        patch_list = np.array([img_resolution//8, img_resolution//4,img_resolution//2]) # 32 64 
        #batch_mul_avg = np.sum(np.array(p_list) * np.array([4, 2, 1]))  # 2 
        progressive = True
    else:
        patch_size = cfg.pachify_size
    # patch pos 
    #start_pos = (128 - patch_size) // 2 
    # 使用配置中的位置
    positions = {
        'i': cfg.patch_start_i,
        'j': cfg.patch_start_j,
        'k': cfg.patch_start_k
    }  
    
    for epoch in range(first_epoch, cfg.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            if cfg.use_multi_patch_size == True:
                if progressive:
                    p_cumsum = p_list.cumsum()
                    p_cumsum[-1] = 10.
                    prog_mask = (global_step / max_train_steps) <= p_cumsum
                    patch_size = int(patch_list[prog_mask][0])
                    #batch_mul = batch_mul_dict[patch_size]
                else:
                    patch_size = int(np.random.choice(patch_list, p=p_list))
            else:
                patch_size = patch_size
            #pdb.set_trace()
            projs = batch['projs'].to(weight_dtype)
            clean_images = batch['gt_idensity'].to(weight_dtype)
            angles = batch['angles']
            #pdb.set_trace()
            patch_image_tensor , patch_pos_tensor , projs_points_tensor =  pachify3d_projected_points(clean_images , patch_size , angles , projector, patch_pos=positions)
            projs_points_tensor = projs_points_tensor.to(weight_dtype)
            #clean_images , images_pos =  pachify3D(clean_images , patch_size)
            #pdb.set_trace()
            # coords = batch['coords'].to(weight_dtype)
            # Sample noise that we'll add to the images
            noise = torch.randn(patch_image_tensor.shape,
                    dtype=weight_dtype, device=clean_images.device)

            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=patch_image_tensor.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            # noisy_idensity = noise_scheduler.add_noise(
            #     clean_images[:,3:4,...], noise, timesteps)

            noisy_idensity = noise_scheduler.add_noise(
                patch_image_tensor, noise, timesteps)
            noisy_images = torch.cat([noisy_idensity ,patch_pos_tensor ] , dim=1)
            #pdb.set_trace()
            with accelerator.accumulate(model):
                # Predict the noise residual
                #pdb.set_trace()
                model_output = model(noisy_images, timesteps , projs , projs_points_tensor)
                if cfg.prediction_type == "epsilon":
                    # this could have different weights!
                    loss = F.mse_loss(model_output.float(), noise.float())
                elif cfg.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (
                            clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * \
                        F.mse_loss(model_output.float(),
                                   clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {cfg.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        #pdb.set_trace()
                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
                0], "step": global_step}
            if cfg.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if cfg.eval_vis :
            if accelerator.is_main_process:
                if epoch % cfg.valid_epochs == 0 or epoch == cfg.num_epochs - 1:
                    logger.info(f"Running test set evaluation at epoch {epoch}")  
                    eval_output = test_pipeline(
                        test_dataloader=test_dataloader,
                        epoch=epoch,
                        global_step=global_step,
                        best_metrics=best_metrics,
                        ema_model=ema_model if cfg.use_ema else None,
                        num_inference_steps=cfg.ddpm_num_inference_steps,
                        generator=torch.Generator(device=accelerator.device).manual_seed(0)
                    )
                    # Update best metrics
                    if eval_output.best_so_far:
                        best_metrics = eval_output.metrics
                        if eval_output.improved_metric:
                            logger.info(
                                f"New best {eval_output.improved_metric} achieved - "
                                f"PSNR: {best_metrics['psnr']:.4f}, "
                                f"SSIM: {best_metrics['ssim']:.4f}"
                            )       

                    # Save latest evaluation checkpoint
                    last_eval_dir = os.path.join(cfg.output_dir, 'last_eval')
                    if os.path.exists(last_eval_dir):
                        shutil.rmtree(last_eval_dir)
                    os.makedirs(last_eval_dir)
                    accelerator.save_state(last_eval_dir)         

                            
    accelerator.end_training()




if __name__ == "__main__":
    print("start train!")
    main()
