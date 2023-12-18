from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def split_and_concat_image(image, n, padding=10):
    # 计算每张子图的宽度
    width, height = image.size
    sub_width = width // n

    # 切分图像
    sub_images = [image.crop((i * sub_width, 0, (i + 1) * sub_width, height)) for i in range(n)]

    # 计算拼接图像的大小
    total_width = width + padding * (n - 1)
    max_height = height

    # 创建一个新的白色图像
    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    # 拼接图像
    x_offset = 0
    for im in sub_images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + padding

    return new_im


def normalize_array(arr: np.ndarray) -> np.ndarray:
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


def concat_images(arrays, padding=10, ind_list=None):
    if ind_list is None:
        ind_list = np.arange(len(arrays))
    # arrays = [arr for i, arr in enumerate(arrays) if i in ind_list]
    # height, width = arrays[0].shape[0], arrays[0].shape[1]
    # arrays_large = np.zeros((height, width * len(arrays)))
    # for i in range(len(arrays)):
    #     arrays_large[:, i * width : (i+1) * width] = arrays[i]
    # image_large = Image.fromarray(np.uint8(plt.cm.jet(normalize_array(arrays_large)) * 255))
    # new_im = split_and_concat_image(image_large, len(arrays), padding=padding)
    images = [Image.fromarray(np.uint8(plt.cm.jet(normalize_array(arr)) * 255)) for i, arr in enumerate(arrays)
              if i in ind_list]
    # 计算拼接图像的大小
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)

    # 创建一个新的白色图像
    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    # 拼接图像
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + padding
    return new_im


def visualize_feature_maps(feature_map, save_path=None):
    C, H, W = feature_map.shape
    for i in range(C):
        plt.subplot(C // 8 + 1, 8, i + 1)
        plt.imshow(feature_map[i], cmap='jet')
        plt.axis('off')
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path, format='pdf')
    plt.show()
