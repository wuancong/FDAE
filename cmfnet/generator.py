'''
follow implementation in https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
'''
import torch.nn as nn
import torch


class ConditionMapGenerator(nn.Module):
    def __init__(self, latent_dim, use_fp16=False, img_size=64, channel_list=[192, 192, 384, 576], num_groups=32):
        super(ConditionMapGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.init_size = img_size // 2**4 # 4 times of x2 upsample
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks1 = nn.Sequential(
            nn.GroupNorm(num_groups, 128), # 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channel_list[-1], 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, channel_list[-1]),
            nn.LeakyReLU(0.2, inplace=True), # 8
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_list[-1], channel_list[-2], 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, channel_list[-2]),
            nn.LeakyReLU(0.2, inplace=True), # 16
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_list[-2], channel_list[-3], 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, channel_list[-3]),
            nn.LeakyReLU(0.2, inplace=True), # 32
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_list[-3], channel_list[-4], 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, channel_list[-4]),
            nn.LeakyReLU(0.2, inplace=True), # 64
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out_list = []
        out_list.append(self.conv_blocks1(out))
        out_list.append(self.conv_blocks2(out_list[-1]))
        out_list.append(self.conv_blocks3(out_list[-1]))
        out_list.append(self.conv_blocks4(out_list[-1]))
        out_list1 = [e.type(self.dtype) for e in out_list]
        return out_list1


class SeperateMaskGenerator(nn.Module):
    def __init__(self, latent_dim, num_masks, img_size=64, use_fp16=False, channel_list=[384, 256, 128, 64],
                 num_groups=32):
        super(SeperateMaskGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.init_size = img_size // 2**4 # 4 times of x2 upsample
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 384 * self.init_size ** 2))
        self.num_masks = num_masks
        self.latent_dim = latent_dim
        self.conv_blocks = nn.ModuleList()
        in_dim = 384
        for out_dim in channel_list:
            conv_block = nn.Sequential(
                nn.GroupNorm(num_groups, in_dim),  # 4
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, out_dim),
                nn.SiLU(),  # 8
            )
            in_dim = out_dim
            self.conv_blocks.append(conv_block)
        self.conv_blocks_img = nn.Conv2d(in_dim, 1, 3, stride=1, padding=1)
        self.mask_normalize_block = nn.Softmax(dim=1)  # mask

    def forward(self, z):
        # size of input z: N x num_masks x mask_code
        # convert z from N x num_masks x mask_code to (N x num_masks) x mask_code
        N, num_masks, mask_code_dim = z.size()
        assert self.num_masks == num_masks
        assert self.latent_dim == mask_code_dim
        z = z.view(N * num_masks, mask_code_dim)
        out = self.l1(z)
        out = out.view(out.shape[0], 384, self.init_size, self.init_size)
        for block in self.conv_blocks:
            out = block(out)
        out = self.conv_blocks_img(out)
        _, _, H, W = out.size()
        out = out.view(N, num_masks, H, W)
        out = self.mask_normalize_block(out)
        return out
