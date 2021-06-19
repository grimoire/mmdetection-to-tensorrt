from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)

from .plugins import create_deformable_pool_plugin


@tensorrt_converter('mmdet2trt.models.roi_heads.roi_extractors.pooling_layers'
                    '.deform_roi_pool_extractor.deformable_roi_pool_wrap')
# @tensorrt_converter('mmcv.ops.deform_roi_pool')
def convert_DeformPool(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    rois = get_arg(ctx, 'rois', pos=1, default=None)
    offset = get_arg(ctx, 'offset', pos=2, default=None)
    out_size = get_arg(ctx, 'output_size', pos=3, default=(7, 7))
    spatial_scale = get_arg(ctx, 'spatial_scale', pos=4, default=1.)
    sampling_ratio = get_arg(ctx, 'sampling_ratio', pos=5, default=0)
    gamma = get_arg(ctx, 'gamma', pos=6, default=0.1)

    output = ctx.method_return

    input_trt = trt_(ctx.network, input)
    rois_trt = trt_(ctx.network, rois)
    offset_trt = None
    if offset is not None and len(offset.shape) > 1:
        offset_trt = trt_(ctx.network, offset)

    plugin = create_deformable_pool_plugin('deform_roi_pool_' + str(id(input)),
                                           out_size, spatial_scale,
                                           sampling_ratio, gamma)

    if offset_trt is None:
        custom_layer = ctx.network.add_plugin_v2(
            inputs=[input_trt, rois_trt], plugin=plugin)
    else:
        custom_layer = ctx.network.add_plugin_v2(
            inputs=[input_trt, rois_trt, offset_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
