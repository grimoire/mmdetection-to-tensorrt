import torch
import torch.nn.functional as F


def adaptive_max_pool2d_by_input(x, shape_wraper):
    gather_size = shape_wraper.size()
    return F.adaptive_max_pool2d(x, output_size=gather_size)


def arange_gridmesh(x, starts=[0, 0], strides=[1, 1]):
    h, w = x.shape[2:]
    dtype = x.dtype
    x_range = torch.arange(w, dtype=dtype, device=x.device) * strides[0]
    y_range = torch.arange(h, dtype=dtype, device=x.device) * strides[1]
    x_range += starts[0]
    y_range += starts[1]
    y, x = torch.meshgrid(y_range, x_range)

    return y, x


def arange_by_input(x, dim, start=0, stride=1):
    lin = torch.linspace(
        start, start + stride * (x.size(dim) - 1), x.size(dim),
        dtype=x.dtype).to(x.device)
    return lin


def pad_with_value(x, pad_dim, pad_size, pad_value=None):
    num_dims = len(x.shape)
    pad_slice = (slice(None, None, None), ) * num_dims
    pad_slice = pad_slice[:pad_dim] + (slice(0, 1,
                                             1), ) + pad_slice[pad_dim + 1:]
    repeat_size = [1] * num_dims
    repeat_size[pad_dim] = pad_size

    x_pad = x.__getitem__(pad_slice)
    if pad_value is not None:
        x_pad = x_pad * 0 + pad_value

    x_pad = x_pad.repeat(*repeat_size)
    x = torch.cat([x, x_pad], dim=pad_dim)
    return x


def gather_topk(x, dim, index):
    num_dims = len(x.shape)
    num_index = len(index.shape)

    for i in range(num_dims - num_index):
        index = index.unsqueeze(-1)

    if num_index != num_dims:
        repeat_size = (1, ) * num_index
        for i in range(num_index, num_dims):
            repeat_size += (x.shape[i], )
        index = index.repeat(*repeat_size)

    return x.gather(dim, index)
