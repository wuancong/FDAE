"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch as th
import torch.distributed as dist
import torch
from edm import dist_util, logger
from edm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from edm.random_util import get_generator
from edm.karras_diffusion import karras_sample
import json
from PIL import Image
import PIL


def main():
    args = create_argparser().parse_args()
    if args.distributed_enable:
        dist_util.setup_dist()
    logger.configure(dir='logs/tmp', args=args)

    if args.condition_generator_path == '':
        args.condition_generator_path = args.model_path
    train_args_path = os.path.join(os.path.dirname(args.condition_generator_path), 'args.json')  # load train args
    with open(os.path.join(train_args_path), "r") as fp:
        train_args = json.load(fp)

    logger.log("creating model and diffusion...")
    additional_cond_map_layer_dim = train_args.get('additional_cond_map_layer_dim')
    if additional_cond_map_layer_dim is None:
        additional_cond_map_layer_dim = -1
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        additional_cond_map_layer_dim=additional_cond_map_layer_dim,
        distillation=False,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )

    if dist.is_initialized():
        model.to(dist_util.dev())
    else:
        model.to(th.device('cuda'))

    from cmfnet.content_mask_generator import ContentMaskGenerator
    images = []
    data_list = os.listdir(args.data_dir)
    for i, file in enumerate(data_list):
        if i == args.num_samples:
            break
        if not file.endswith('jpg') and not file.endswith('png'):
            continue
        img = Image.open(os.path.join(args.data_dir, file))
        arr = np.array(img)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])
        images.append(torch.from_numpy(arr))
    factors = torch.stack(images, dim=0)
    factors.requires_grad_(False)

    if train_args.get('encoder_type') is None:
        train_args['encoder_type'] = 'resnet18'
    condition_generator = ContentMaskGenerator(semantic_group_num=train_args['semantic_group_num'],
                                               semantic_code_dim=train_args['semantic_code_dim'],
                                               mask_code_dim=train_args['mask_code_dim'],
                                               use_fp16=train_args['use_fp16'],
                                               encoder_type=train_args['encoder_type'],
                                               )
    condition_idx = train_args['condition_idx']
    state_dict = torch.load(args.condition_generator_path)
    condition_generator.load_state_dict(state_dict, strict=False)
    condition_generator.to(dist_util.dev())
    condition_generator.half()

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    condition_generator.eval()
    condition_generator.requires_grad_(False)

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    model_kwargs = {}
    model_kwargs['condition_generator'] = condition_generator
    model_kwargs['condition_idx'] = condition_idx
    while len(all_images) * args.batch_size < args.num_samples:
        len_all_images = len(all_images)
        if factors is not None:
            model_kwargs['factors'] = factors[len_all_images:min(len_all_images+args.batch_size, args.num_samples)].cuda()
        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.tensor(classes, device=dist_util.dev())
            model_kwargs["y"] = classes
            # extract features and mask for condition generator

        image_input = None

        if args.use_image_input:
            image_input = model_kwargs['factors']

        if args.save_mask:
            # {'mask_code': mask_code, 'mask': mask_output, 'semantic_code_list': semantic_code_list, 'condition_map': [condition_map]}
            from visualization.visualize_feature_maps import visualize_feature_maps, concat_images
            out_dict = condition_generator.forward(model_kwargs['factors'])
            mask_output = out_dict['mask']

            for i in range(mask_output.size(0)):
                filename = f"{i + 1:02d}.jpg"
                path_str = Path(args.condition_generator_path)
                output_dir = os.path.join(path_str.parent, f'{path_str.stem}_imgs_each_masks')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_path = os.path.join(output_dir, filename)
                imgs = [mask_output[i, j, :, :].cpu().detach().numpy() for j in range(mask_output.size(1))]
                new_imgs = concat_images(imgs)
                new_imgs.save(save_path)

        swap_flag = args.swap_flag
        if swap_flag:
            c_g, m_g = args.factor_dim.split('-')
            assert c_g[0] == 'c' and m_g[0] == 'm'
            semantic_group, mask_group = [], []
            if c_g[1] != '0':
                semantic_group = [int(e)-1 for e in c_g[1:].split(',')]
            if m_g[1] != '0':
                mask_group = [int(e)-1 for e in m_g[1:].split(',')]

            model_kwargs['swap_info'] = {
                'source_ind': np.arange(args.num_samples),  # use content
                'target_ind': np.arange(args.num_samples),  # use mask
                'semantic_group': semantic_group,
                'mask_group': mask_group,
            }
            args.batch_size = len(model_kwargs['swap_info']['source_ind']) * \
                              len(model_kwargs['swap_info']['target_ind'])
            args.num_samples = args.batch_size

        sample = karras_sample(
            diffusion,
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            steps=args.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=args.clip_denoised,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            s_churn=args.s_churn,
            s_tmin=args.s_tmin,
            s_tmax=args.s_tmax,
            s_noise=args.s_noise,
            generator=generator,
            ts=ts,
            image_input=image_input,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [sample]
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # save output images
    if not isinstance(all_images[0], np.ndarray):
        return
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    path_str = Path(args.condition_generator_path)
    output_dir = os.path.join(path_str.parent, f'{path_str.stem}_imgs_factor_{args.factor_dim}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img in enumerate(arr):
        img = PIL.Image.fromarray(img)
        filename = f"{i + 1:06d}.jpg"
        img.save(os.path.join(output_dir, filename))

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ-indiv",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        condition_generator_path="",
        seed=42,
        ts="",
        distributed_enable=False,
        debug_mode=True,
        factor_dim='',
        factor_type='original image',
        data_dir='',
        save_mask=False,
        save_feature=False,
        traversal_flag=False,
        swap_flag=False,
        use_image_input=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
