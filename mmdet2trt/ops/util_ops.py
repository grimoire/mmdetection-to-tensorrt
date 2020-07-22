import torch
from torch import nn
import torch.nn.functional as F


def adaptive_max_pool2d_by_input(x, shape_warper):
    gather_size = shape_warper.size()
    return F.adaptive_max_pool2d(
                    x, output_size=gather_size)


def arange_gridmesh(x, starts=[0,0], strides=[1,1]):
    h, w = x.shape[2:]
    dtype = x.dtype
    x_range = torch.arange(w, dtype=dtype, device=x.device)
    y_range = torch.arange(h, dtype=dtype, device=x.device)
    y, x = torch.meshgrid(y_range, x_range)

    return y, x
