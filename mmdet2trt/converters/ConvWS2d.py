import torch
from torch2trt_dynamic.converters.Conv2d import convert_Conv2d
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter


@tensorrt_converter('mmcv.cnn.ConvWS2d.forward')
def convert_ConvWS2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    dilation = module.dilation
    groups = module.groups
    eps = module.eps

    weight = module.weight
    in_channels = weight.size()[1]
    out_channels = weight.size()[0]
    bias = module.bias
    need_bias = True if bias is not None else False
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    weight = torch.nn.Parameter(weight)

    new_module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=need_bias)
    new_module.weight = weight
    new_module.bias = bias

    ctx.method_args = (new_module, input)
    convert_Conv2d(ctx)
