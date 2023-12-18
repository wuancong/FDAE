"""
Helpers for distributed training.
"""

import torch as th
import torch.distributed as dist
import platform


def setup_dist(args):
    """
    Setup a distributed process group.
    """
    if args.debug_mode:
        return
    if dist.is_initialized():
        return
    backend = "gloo" if not th.cuda.is_available() or platform.system() == "Windows" else "nccl"
    dist.init_process_group(backend=backend)


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        if dist.is_initialized():
            return dist.get_rank()
        else:
            return th.device('cuda')
    else:
        return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    pass
    return
