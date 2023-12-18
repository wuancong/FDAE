import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18
from cmfnet.generator import SeperateMaskGenerator


def get_last_layer_output_channels(model):
    # Get the last layer of the model
    last_layer = list(model.modules())[-1]
    # Check if the last layer is a linear or convolutional layer
    if isinstance(last_layer, nn.Linear):
        # For a linear layer, return the number of output features
        return last_layer.in_features
    elif isinstance(last_layer, nn.Conv2d):
        # For a convolutional layer, return the number of output channels
        return last_layer.out_channels
    elif isinstance(last_layer, nn.BatchNorm2d):
        return last_layer.num_features
    elif isinstance(last_layer, nn.AdaptiveAvgPool2d):
        return get_last_layer_output_channels(torch.nn.Sequential(*list(model.children())[:-1]))
    else:
        raise ValueError('last_layer value not supported')


class ContentMaskGenerator(nn.Module):
    def __init__(self, semantic_group_num=2, semantic_code_dim=80, mask_code_dim=80,
                 semantic_code_adjust_dim=80,
                 use_fp16=False, encoder_type='resnet18'):
        '''
        semantic_group_num: concept number N
        semantic_code_dim: dimensionality of content codes
        mask_code_dim: dimensionality of mask codes
        semantic_code_adjust_dim: dimensionality of content codes after dimension adjustment
        '''
        super(ContentMaskGenerator, self).__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.semantic_group_num = semantic_group_num
        self.semantic_code_adjust_dim = semantic_code_adjust_dim
        self.semantic_code_dim = semantic_code_dim
        self.mask_code_dim = mask_code_dim
        if encoder_type == 'resnet18':
            self.encoder = resnet18(weights=None)
        elif encoder_type == 'resnet50':
            self.encoder = resnet50(weights=None)
        else:
            raise(ValueError('Unsupported encoder type'))
        encoder_out_dim = get_last_layer_output_channels(self.encoder)
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])
        self.gn_dim = 32

        self.semantic_decoder1 = nn.Linear(encoder_out_dim, self.semantic_code_dim * self.semantic_group_num)
        self.semantic_code_dim = semantic_code_dim

        self.semantic_decoder2 = nn.ModuleList()
        for i in range(self.semantic_group_num):
            self.semantic_decoder2.append(nn.Linear(self.semantic_code_dim, semantic_code_adjust_dim))

        self.mask_decoder = nn.Linear(encoder_out_dim, mask_code_dim * semantic_group_num)
        self.mask_generator = SeperateMaskGenerator(latent_dim=mask_code_dim, num_masks=semantic_group_num)

    def forward(self, x, model_kwargs=None):
        swap_info = None  # swap_info is used for image editing by swapping latent codes
        if model_kwargs is not None:
            swap_info = model_kwargs.get('swap_info')
        if swap_info is not None:
            source_ind = swap_info['source_ind']
            target_ind = swap_info['target_ind']
            semantic_group = swap_info['semantic_group']
            mask_group = swap_info['mask_group']

        x = x.type(self.dtype)

        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        semantic_code = self.semantic_decoder1(features)
        semantic_code_list = semantic_code.chunk(self.semantic_group_num, dim=1)
        layer_semantic_code0_list = [net(code)  # code nxd_group
                                     for code, net in zip(semantic_code_list, self.semantic_decoder2)]
        layer_semantic_code0 = torch.stack(layer_semantic_code0_list, dim=2).unsqueeze(3)  # nxdxgroup_numx1
        semantic_code_out = semantic_code

        if swap_info is not None:
            # layer_semantic_code0: nxdxgroup_numx1
            layer_semantic_code0_new = torch.zeros((len(source_ind), len(target_ind), layer_semantic_code0.shape[1],
                                                    layer_semantic_code0.shape[2], layer_semantic_code0.shape[3]),
                                                   dtype=layer_semantic_code0.dtype).cuda()
            swap_array = swap_info.get('cluster_center')
            for i_s, s_ind in enumerate(source_ind):
                for i_t, t_ind in enumerate(target_ind):
                    layer_semantic_code0_new[i_s, i_t] = layer_semantic_code0[s_ind]
                    #swap
                    for g in semantic_group:
                        if swap_array is None:
                            layer_semantic_code0_new[i_s, i_t, :, g, :] = layer_semantic_code0[t_ind, :, g, :]
                        else:
                            layer_semantic_code0_new[i_s, i_t, :, g, :] = torch.from_numpy(swap_array['semantic'][g][:, i_t]).unsqueeze(1).cuda().half()
            layer_semantic_code0 = layer_semantic_code0_new.reshape(len(source_ind) * len(target_ind), layer_semantic_code0.shape[1],
                                               layer_semantic_code0.shape[2], layer_semantic_code0.shape[3])

        layer_semantic_code_list = [layer_semantic_code0]

        mask_code = self.mask_decoder(features)
        mask_code = mask_code.view(mask_code.size(0), self.semantic_group_num, self.mask_code_dim)
        mask_code_out = [mask_code.view(mask_code.size(0), -1)]

        if self.semantic_group_num == 1:
            mask_code = None
            mask_code_out = []

        if swap_info is not None:
            # mask_code nxgroup_numxd
            mask_code_new = torch.zeros((len(source_ind), len(target_ind), self.semantic_group_num, self.mask_code_dim),
                                        dtype=mask_code.dtype).cuda()
            swap_array = swap_info.get('cluster_center')
            for i_s, s_ind in enumerate(source_ind):
                for i_t, t_ind in enumerate(target_ind):
                    mask_code_new[i_s, i_t] = mask_code[s_ind]
                    #swap
                    for g in mask_group:
                        if swap_array is None:
                            mask_code_new[i_s, i_t, g, :] = mask_code[t_ind, g, :]
                        else:
                            mask_code_new[i_s, i_t, g, :] = torch.from_numpy(swap_array['mask'][g][:, i_t]).unsqueeze(1).cuda().half()
            mask_code = mask_code_new.reshape(len(source_ind) * len(target_ind), self.semantic_group_num, self.mask_code_dim)

        if self.semantic_group_num == 1:
            mask_list = [torch.ones(x.shape[0], self.semantic_group_num, x.shape[2], x.shape[3]).cuda()]
            mask_output = mask_list[-1]
        else:
            mask_list = [self.mask_generator(mask_code)]
            mask_output = mask_list[-1]

        condition_map_list = []
        for semantic_code_map, mask in zip(layer_semantic_code_list, mask_list):
            N, _, H, W = mask.size()
            # expand maps
            semantic_code_map = semantic_code_map.unsqueeze(-1).expand(
                N, self.semantic_code_adjust_dim, self.semantic_group_num, H, W)
            mask = mask.unsqueeze(1).expand_as(semantic_code_map)
            condition_map = torch.sum(semantic_code_map * mask, dim=2)
            condition_map_list.append(condition_map)
        return {'mask_code': mask_code, 'mask': mask_output,
                'semantic_code': semantic_code_list,
                'condition_map': condition_map_list,
                'feature': torch.cat([semantic_code_out] + mask_code_out, dim=1), 'feature_avg_pool': features
                }
