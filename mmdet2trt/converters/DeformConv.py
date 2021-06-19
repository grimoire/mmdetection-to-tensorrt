from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)

from .plugins import create_dcn_plugin, create_dcnv2_plugin


@tensorrt_converter('mmcv.ops.deform_conv.deform_conv2d')
def convert_DeformConv(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    offset = get_arg(ctx, 'offset', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=1)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    dilation = get_arg(ctx, 'dilation', pos=5, default=1)
    groups = get_arg(ctx, 'groups', pos=6, default=1)
    deform_groups = get_arg(ctx, 'deform_groups', pos=7, default=1)

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    offset_trt = trt_(ctx.network, offset)
    weight_trt = trt_(ctx.network, weight)

    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    plugin = create_dcn_plugin(
        'dcn_' + str(id(input)),
        stride=stride,
        padding=padding,
        dilation=dilation,
        deformable_group=deform_groups,
        group=groups)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, offset_trt, weight_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


@tensorrt_converter('mmcv.ops.modulated_deform_conv.modulated_deform_conv2d')
def convert_ModulatedDeformConv(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    offset = get_arg(ctx, 'offset', pos=1, default=None)
    mask = get_arg(ctx, 'mask', pos=2, default=None)
    weight = get_arg(ctx, 'weight', pos=3, default=None)
    bias = get_arg(ctx, 'bias', pos=4, default=None)
    stride = get_arg(ctx, 'stride', pos=5, default=1)
    padding = get_arg(ctx, 'padding', pos=6, default=0)
    dilation = get_arg(ctx, 'dilation', pos=7, default=1)
    groups = get_arg(ctx, 'groups', pos=8, default=1)
    deform_groups = get_arg(ctx, 'deform_groups', pos=9, default=1)

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    offset_trt = trt_(ctx.network, offset)
    mask_trt = trt_(ctx.network, mask)
    weight_trt = trt_(ctx.network, weight)
    if bias is not None:
        bias_trt = trt_(ctx.network, bias)

    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    plugin = create_dcnv2_plugin(
        'dcn_' + str(id(input)),
        stride=stride,
        padding=padding,
        dilation=dilation,
        deformable_group=deform_groups,
        group=groups)

    layer_inputs = [input_trt, offset_trt, mask_trt, weight_trt]
    if bias is not None:
        layer_inputs += [bias_trt]
    custom_layer = ctx.network.add_plugin_v2(
        inputs=layer_inputs, plugin=plugin)

    output._trt = custom_layer.get_output(0)
