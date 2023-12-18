"""
use the distilled one-step model, train only the last step T,
input the groundtruth factors as conditions to condition generator,
freeze the diffusion model and update only the condition generator
"""
import os, torch
import argparse
from edm import dist_util, logger
from edm.image_datasets import load_data
from edm.resample import create_named_schedule_sampler
from edm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from edm.train_util import TrainLoop
import torch.distributed as dist
from cmfnet.content_mask_generator import ContentMaskGenerator
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = create_argparser().parse_args()
    if args.encoder_input_resize is not None:
        assert isinstance(args.encoder_input_resize, str)
        args.encoder_input_resize = [int(e) for e in args.encoder_input_resize.split(',')]
    if args.available_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.available_gpus

    setup_seed(args.seed)
    torch.backends.cudnn.benchmark = args.use_cudnn

    args.condition_idx = [int(e) for e in args.condition_idx.split(',')]

    dist_util.setup_dist(args)
    log_suffix = args.data_dir.split('/')[-1]+'_'
    log_suffix += args.log_suffix
    logger.configure(dir=args.logs_dir, log_suffix=log_suffix, args=args)

    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_and_diffusion_kwargs['loss_norm'] = args.loss_norm
    model_and_diffusion_kwargs['additional_cond_map_layer_dim'] = args.additional_cond_map_layer_dim
    model_and_diffusion_kwargs['sigma_weight'] = args.sigma_weight
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.to(dist_util.dev())

    # conditional generator
    if args.image_cond:
        condition_generator = ContentMaskGenerator(semantic_group_num=args.semantic_group_num,
                                                   semantic_code_dim=args.semantic_code_dim,
                                                   mask_code_dim=args.mask_code_dim,
                                                   semantic_code_adjust_dim=args.semantic_code_adjust_dim,
                                                   use_fp16=args.use_fp16,
                                                   encoder_type=args.encoder_type)
        condition_generator.to(dist_util.dev())
    else:
        condition_generator = None

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        if args.debug_mode:
            world_size = 1
        else:
            world_size = dist.get_world_size()
        batch_size = args.global_batch_size // world_size
        if args.global_batch_size % world_size != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {world_size*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        factor_cond=args.factor_cond,
        image_cond=args.image_cond,
        image_cond_path=args.image_cond_path,
        num_workers=args.num_workers,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        dataset_name=os.path.basename(args.data_dir),
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        resume_checkpoint=args.resume_checkpoint,
        eval_only=args.eval_only,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        condition_generator=condition_generator,
        update_condition_generator_only=args.update_condition_generator_only,
        debug_mode=args.debug_mode,
        max_step=args.max_step,
        class_cond=args.class_cond,
        condition_idx=args.condition_idx,
        content_decorrelation_weight=args.content_decorrelation_weight,
        mask_entropy_weight=args.mask_entropy_weight,
        train_args=args,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        content_decorrelation_weight=-1.0,
        mask_entropy_weight=-1.0,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        log_interval=10,
        save_interval=10000,
        eval_interval=100000,
        resume_checkpoint="",
        eval_only=False,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        available_gpus='-1',
        debug_mode=False,
        log_suffix='',
        max_step=10000,
        factor_cond=False,
        image_cond=False,
        image_cond_path='original image',
        update_condition_generator_only=False,
        semantic_group_num=6,
        semantic_code_dim=8,
        mask_code_dim=8,
        condition_idx='0',
        semantic_code_adjust_dim=80,
        logs_dir=None,
        loss_norm='l2',
        semantic_dim_reduce='joint',
        use_seperate_mask=False,
        encoder_type='resnet18',
        encoder_input_resize=None,
        use_cudnn=True,
        sigma_weight=1.0,
        additional_cond_map_layer_dim=-1,
        seed=0,
        semantic_code_str='semantic_code',
        num_workers=8,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
