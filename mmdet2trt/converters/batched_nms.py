import torch
from torch2trt_dynamic import tensorrt_converter, trt_

from .plugins import create_batchednms_plugin


@tensorrt_converter(
    'mmdet2trt.core.post_processing.batched_nms.BatchedNMS.forward')
def convert_batchednms(ctx):
    module = ctx.method_args[0]
    scores = ctx.method_args[1]
    bboxes = ctx.method_args[2]
    topK = ctx.method_args[3]
    keepTopK = ctx.method_args[4]

    topK = min(scores.shape[1], topK)
    keepTopK = min(scores.shape[1], keepTopK)

    scoreThreshold = module.scoreThreshold
    iouThreshold = module.iouThreshold
    backgroundLabelId = module.backgroundLabelId
    numClasses = scores.shape[2]
    shareLocation = (bboxes.shape[2] == 1)

    scores_trt = trt_(ctx.network, scores)
    bboxes_trt = trt_(ctx.network, bboxes)
    output = ctx.method_return

    plugin = create_batchednms_plugin(
        'batchednms_' + str(id(module)),
        scoreThreshold,
        iouThreshold,
        topK,
        keepTopK,
        numClasses,
        backgroundLabelId,
        shareLocation=shareLocation,
        isNormalized=False,
        clipBoxes=False)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[bboxes_trt, scores_trt], plugin=plugin)

    if isinstance(output, torch.Tensor):
        output._trt = custom_layer.get_output(0)

    else:
        for i in range(len(output)):
            output[i]._trt = custom_layer.get_output(i)
