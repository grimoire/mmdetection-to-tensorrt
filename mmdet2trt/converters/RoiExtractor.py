from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)

from .plugins import create_roiextractor_plugin


@tensorrt_converter(
    'mmdet2trt.models.roi_heads.roi_extractors.'
    'pooling_layers.roi_align_extractor.RoiAlignExtractor.forward')
def convert_roiextractor(ctx):
    module = ctx.method_args[0]
    feats = get_arg(ctx, 'feats', pos=1, default=None)
    rois = get_arg(ctx, 'rois', pos=2, default=None)
    roi_scale_factor = get_arg(ctx, 'roi_scale_factor', pos=3, default=None)
    if not roi_scale_factor:
        roi_scale_factor = -1.0

    out_size = module.roi_layers[0].output_size[0]
    sample_num = module.roi_layers[0].sampling_ratio
    featmap_strides = module.featmap_strides
    finest_scale = module.finest_scale

    feats_trt = [trt_(ctx.network, f) for f in feats]
    rois_trt = trt_(ctx.network, rois)
    output = ctx.method_return

    plugin = create_roiextractor_plugin(
        'roiextractor_' + str(id(module)),
        out_size,
        sample_num,
        featmap_strides,
        roi_scale_factor,
        finest_scale,
        aligned=1)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[rois_trt] + feats_trt, plugin=plugin)

    output._trt = custom_layer.get_output(0)
