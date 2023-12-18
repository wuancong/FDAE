import math
import os.path
import random
from pathlib import Path
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    factor_cond=False,
    image_cond=False,
    image_cond_path='original image',
    deterministic=False,
    random_crop=False,
    random_flip=False,
    num_workers=1,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param factor_cond: if True, use the ground truth factor in the disentanglement datasets as condition
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    factors = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # classes = [sorted_classes[x] for x in class_names]

        # use the id in cars3d as class_cond
        fnames = [bf.basename(path) for path in all_files]
        with open(bf.join(data_dir, "factors.json"), "r") as fp:
            factors_dict = json.load(fp)
        classes = [factors_dict[fname][2] - 1 for fname in fnames] # the ids
    if factor_cond:
        fnames = [bf.basename(path) for path in all_files]
        with open(bf.join(data_dir, "factors.json"), "r") as fp:
            factors_dict = json.load(fp)
        factors = [factors_dict[fname] for fname in fnames]
        factors = np.array(factors, dtype=np.float32)
        factors = _normalize_columns(factors)
    if image_cond:
        if 'original image' in image_cond_path:
            factors = image_cond_path
        elif Path(image_cond_path).exists():
            fnames = [bf.basename(path) for path in all_files]
            factors = []
            for fname in fnames:
                img = Image.open(os.path.join(image_cond_path, fname))
                factors.append(np.array(img, dtype=np.uint8))
            factors = np.stack(factors)
            factors = factors.reshape((factors.shape[0], 1, factors.shape[1], factors.shape[2]))
            np.save(os.path.join(image_cond_path, 'image_cond.npy'), factors)
            # cached by numpy
            # factors = np.load(os.path.join(image_cond_path, 'image_cond.npy'))
            # factors = np.array(factors, dtype=np.float32) / 255.0
        else:
            raise ValueError('image_cond_path is not a valid option or directory')

    if dist.is_initialized():
        shard = dist.get_rank()
        num_shards = dist.get_world_size()
    else:
        shard = 0
        num_shards = 1
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        factors=factors,
        shard=shard,
        num_shards=num_shards, #MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def _normalize_columns(arr):
    # 计算每列的最大值和最小值
    col_max = arr.max(axis=0)
    col_min = arr.min(axis=0)
    # 计算每列的范围
    col_range = col_max - col_min
    # 避免除以零
    col_range[col_range == 0] = 1
    # 对每列进行归一化
    norm_arr = 2 * (arr - col_min) / col_range - 1
    return norm_arr


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        factors=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.factor_transform = None
        if 'original image' in factors:
            self.local_factors = factors
            if 'original image dino' in factors:
                # from lightly.transforms.dino_transform import DINOTransform
                from lightly.transforms.gaussian_blur import GaussianBlur
                from lightly.transforms.solarize import RandomSolarization
                from lightly.transforms.utils import IMAGENET_NORMALIZE
                import torchvision.transforms as T
                hf_prob: float = 0.5
                cj_strength: float = 0.5
                cj_bright: float = 0.8
                cj_contrast: float = 0.8
                cj_sat: float = 0.4
                cj_hue: float = 0.2
                cj_prob: float = 0.8
                random_gray_scale: float = 0.2
                gaussian_blur = (1.0, 0.1, 0.5)
                kernel_size = None,
                kernel_scale = None,
                sigmas = (0.1, 2)
                solarization_prob: float = 0.2
                if 'flip' in factors:
                    transform = [T.RandomHorizontalFlip(p=hf_prob)]
                else:
                    transform = []
                transform += [
                    T.RandomApply(
                        [
                            T.ColorJitter(
                                brightness=cj_strength * cj_bright,
                                contrast=cj_strength * cj_contrast,
                                saturation=cj_strength * cj_sat,
                                hue=cj_strength * cj_hue,
                            )
                        ],
                        p=cj_prob,
                    ),
                    T.RandomGrayscale(p=random_gray_scale),
                    GaussianBlur(
                        kernel_size=kernel_size,
                        scale=kernel_scale,
                        sigmas=sigmas,
                        prob=gaussian_blur[0],
                    ),
                    RandomSolarization(prob=solarization_prob),
                    T.ToTensor(),
                ]
                transform += [T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"])]
                self.factor_transform = T.Compose(transform) #DINOTransform()
        else:
            self.local_factors = None if factors is None else factors[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = pil_image.resize((self.resolution, self.resolution))
            arr = random_crop_arr(arr, self.resolution)
        else:
            arr = pil_image.resize((self.resolution, self.resolution))
            arr = center_crop_arr(arr, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.local_factors is not None:
            if self.local_factors == 'original image':
                out_dict["factors"] = arr
            elif self.local_factors == 'original image market norm':
                arr_factor = np.array(pil_image)
                mean = np.array([105.6896057223271, 99.13061989599639, 97.91248239907647])
                std = np.array([50.219032666884374, 48.612007061689525, 48.226633015928826])
                arr_factor = (arr_factor - mean) / std
                arr_factor = arr_factor.astype(np.float32)
                arr_factor = np.transpose(arr_factor, [2, 0, 1])
                out_dict["factors"] = arr_factor
            elif 'original image dino' in self.local_factors:
                out_dict["factors"] = self.factor_transform(pil_image).numpy()
            else:
                out_dict["factors"] = self.local_factors[idx]
        return arr, out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
