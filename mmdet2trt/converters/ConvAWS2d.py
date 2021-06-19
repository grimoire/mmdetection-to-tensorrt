import torch
from torch2trt_dynamic.converters.Conv2d import convert_Conv2d
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('mmcv.cnn.ConvAWS2d.forward')
def convert_ConvAWS2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    kernel_size = module.kernel_size
    stride = module.stride
    padding = module.padding
    dilation = module.dilation
    groups = module.groups

    weight = module._get_weight(module.weight)
    weight = torch.nn.Parameter(weight)
    in_channels = weight.size()[1]
    out_channels = weight.size()[0]
    bias = module.bias
    need_bias = True if bias is not None else False

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


@tensorrt_converter('mmcv.cnn.ConvAWS2d._get_weight')
def convert_convws_get_weight(ctx):
    output = ctx.method_return

    output._trt = trt_(ctx.network, output)
