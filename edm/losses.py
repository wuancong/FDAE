import torch
import torch.nn.functional as F


def mask_entropy_loss(p):
    log_p = torch.log(p + 1e-20)
    entropy = -torch.sum(p * log_p, dim=1)
    return torch.mean(entropy)


def content_decorrelation_loss(features_in, group_num):
    n = features_in.shape[0]
    d = features_in.shape[1]
    d_group = d // group_num
    # decenter
    features_mean = torch.mean(features_in, dim=0, keepdim=True)
    features_de = features_in - features_mean
    features_re = features_de.reshape(n, group_num, d_group).permute(1, 2, 0)
    # group_num x d_group x n
    features_norm = F.normalize(features_re, dim=1) / group_num
    features_view = features_norm.contiguous().view(group_num, -1)
    inner_product = features_view.mm(features_view.T)
    mask = torch.ones(group_num, group_num) - torch.eye(group_num)
    off_diag = torch.sum(torch.abs(inner_product) * mask.cuda()) / torch.sum(mask)
    return off_diag
