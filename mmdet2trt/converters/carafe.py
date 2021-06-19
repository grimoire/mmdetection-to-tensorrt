import tensorrt as trt
import torch
from torch2trt_dynamic.converters.pixel_shuffle import convert_pixel_shuffle
from torch2trt_dynamic.converters.softmax import convert_softmax
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)

from .plugins import create_carafefeaturereassemble_plugin


@tensorrt_converter('mmcv.ops.CARAFEPack.feature_reassemble')
def convert_carafe_feature_reassemble(ctx):
    module = ctx.method_args[0]
    x = ctx.method_args[1]
    mask = ctx.method_args[2]

    scale_factor = module.scale_factor
    up_kernel = module.up_kernel
    up_group = module.up_group

    x_trt = trt_(ctx.network, x)
    mask_trt = trt_(ctx.network, mask)
    output = ctx.method_return

    plugin = create_carafefeaturereassemble_plugin(
        'carafefeaturereassemble_' + str(id(module)), scale_factor, up_kernel,
        up_group)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[x_trt, mask_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


@tensorrt_converter('mmcv.ops.CARAFEPack.kernel_normalizer')
def convert_carafe_kernel_normalizer(ctx):
    import torch
    from torch.nn import functional as F
    module = ctx.method_args[0]
    mask = get_arg(ctx, 'mask', pos=1, default=None)

    scale_factor = module.scale_factor
    up_kernel = module.up_kernel

    output = ctx.method_return

    # pixel shuffle
    ps_mask = F.pixel_shuffle(mask, scale_factor)
    ctx.method_args = [mask, scale_factor]
    ctx.method_return = ps_mask
    convert_pixel_shuffle(ctx)

    # view0
    n, mask_c, h, w = ps_mask.size()
    mask_channel = int(mask_c / (up_kernel * up_kernel))
    view_ps_mask = ps_mask.view(n, mask_channel, -1, h, w)
    ps_mask_trt = trt_(ctx.network, ps_mask)
    ps_mask_shape_trt = ctx.network.add_shape(ps_mask_trt).get_output(0)
    ps_mask_batch_trt = ctx.network.add_slice(ps_mask_shape_trt, [0], [1],
                                              [1]).get_output(0)
    ps_mask_channel_trt = ctx.network.add_slice(ps_mask_shape_trt, [1], [1],
                                                [1]).get_output(0)
    ps_mask_hw_trt = ctx.network.add_slice(ps_mask_shape_trt, [2], [2],
                                           [1]).get_output(0)
    kernel_v2_trt = trt_(
        ctx.network,
        torch.tensor([up_kernel * up_kernel],
                     dtype=torch.int32).to(mask.device))
    ps_mask_new_channel_trt = ctx.network.add_elementwise(
        ps_mask_channel_trt, kernel_v2_trt,
        trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
    ps_mask_new_shape_trt = ctx.network.add_concatenation([
        ps_mask_batch_trt, ps_mask_new_channel_trt, kernel_v2_trt,
        ps_mask_hw_trt
    ]).get_output(0)

    layer = ctx.network.add_shuffle(ps_mask_trt)
    layer.set_input(1, ps_mask_new_shape_trt)
    view_ps_mask._trt = layer.get_output(0)

    # softmax
    softmax_mask = F.softmax(view_ps_mask, dim=2)
    ctx.method_args = [view_ps_mask, 2]
    ctx.method_return = softmax_mask
    convert_softmax(ctx)

    # view1
    softmax_mask_trt = trt_(ctx.network, softmax_mask)
    layer = ctx.network.add_shuffle(softmax_mask_trt)
    layer.set_input(1, ps_mask_shape_trt)
    output._trt = layer.get_output(0)
    ctx.method_return = output


@tensorrt_converter('mmdet.models.necks.fpn_carafe.FPN_CARAFE.tensor_add')
def convert_carafe_tensor_add(ctx):
    a = get_arg(ctx, 'a', pos=1, default=None)
    b = get_arg(ctx, 'b', pos=2, default=None)

    a_trt = trt_(ctx.network, a)
    b_trt = trt_(ctx.network, b)
    output = ctx.method_return

    a_shape_trt = ctx.network.add_shape(a_trt).get_output(0)

    layer = ctx.network.add_slice(b_trt, [0] * len(a.shape),
                                  [2] * len(a.shape), [1] * len(a.shape))
    layer.set_input(
        1,
        trt_(ctx.network,
             torch.tensor([0] * len(a.shape), dtype=torch.int32).to(a.device)))
    layer.set_input(2, a_shape_trt)
    layer.set_input(
        3,
        trt_(ctx.network,
             torch.tensor([1] * len(a.shape), dtype=torch.int32).to(a.device)))
    new_b_trt = layer.get_output(0)

    layer = ctx.network.add_elementwise(a_trt, new_b_trt,
                                        trt.ElementWiseOperation.SUM)

    output._trt = layer.get_output(0)
