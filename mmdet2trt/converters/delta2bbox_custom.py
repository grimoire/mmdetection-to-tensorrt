import torch
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)

from .plugins import create_delta2bbox_custom_plugin


@tensorrt_converter(
    'mmdet2trt.core.bbox.coder.delta_xywh_bbox_coder.delta2bbox_custom_func')
def convert_delta2bbox(ctx):
    cls_scores = get_arg(ctx, 'cls_scores', pos=0, default=None)
    bbox_preds = get_arg(ctx, 'bbox_preds', pos=1, default=None)
    anchors = get_arg(ctx, 'anchors', pos=2, default=None)
    min_num_bboxes = get_arg(ctx, 'min_num_bboxes', pos=3, default=1000)
    num_classes = get_arg(ctx, 'num_classes', pos=4, default=1)
    use_sigmoid_cls = get_arg(ctx, 'use_sigmoid_cls', pos=5, default=True)
    target_mean = get_arg(ctx, 'target_mean', pos=6, default=[0, 0, 0, 0])
    target_std = get_arg(ctx, 'target_std', pos=7, default=[1, 1, 1, 1])
    input_x = get_arg(ctx, 'input_x', pos=8, default=None)

    scores_trt = trt_(ctx.network, cls_scores)
    preds_trt = trt_(ctx.network, bbox_preds)
    anchors_trt = trt_(ctx.network, anchors)
    if input_x is not None:
        input_x_trt = trt_(ctx.network, input_x)
        input_x_shape_trt = ctx.network.add_shape(input_x_trt).get_output(0)
    output = ctx.method_return

    plugin = create_delta2bbox_custom_plugin(
        'delta2bbox_custom_' + str(id(cls_scores)),
        use_sigmoid_cls=use_sigmoid_cls,
        min_num_bbox=min_num_bboxes,
        num_classes=num_classes,
        target_means=target_mean,
        target_stds=target_std)

    layer_input = [scores_trt, preds_trt, anchors_trt]
    if input_x is not None:
        layer_input.append(input_x_shape_trt)
    custom_layer = ctx.network.add_plugin_v2(inputs=layer_input, plugin=plugin)

    if isinstance(output, torch.Tensor):
        output._trt = custom_layer.get_output(0)

    else:
        for i in range(len(output)):
            output[i]._trt = custom_layer.get_output(i)
