import torch
from torch2trt_dynamic.converters.roi_align import *
import mmcv.ops


@tensorrt_converter("mmcv.ops.RoIAlign.forward")
def convert_mmcv_RoIAlign(ctx):
    module = ctx.method_args[0]
    input = get_arg(ctx, 'input', pos=1, default=None)
    boxes = get_arg(ctx, 'boxes', pos=2, default=None)

    output_size = module.output_size
    spatial_scale = module.spatial_scale
    sampling_ratio = module.sampling_ratio
    aligned = module.aligned

    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs
    new_method_args = [
        input, boxes, output_size, spatial_scale, sampling_ratio, aligned
    ]
    new_method_kwargs = {}
    ctx.method_args = new_method_args
    ctx.method_kwargs = new_method_kwargs
    convert_roi_align(ctx)
    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
