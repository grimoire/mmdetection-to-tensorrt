import torch
from torch import nn
import torch.nn.functional as F


def adaptive_max_pool2d_by_input(x, shape_warper):
    gather_size = shape_warper.size()
    return F.adaptive_max_pool2d(
                    x, output_size=gather_size)