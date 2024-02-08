import mmdet2trt.ops as mmdet2trt_ops
import torch.nn.functional as F
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter


@tensorrt_converter('mmdet.models.necks.BFP.forward', is_real=False)
def convert_BFP(ctx):
    module = ctx.method_args[0]
    inputs = ctx.method_args[1]
    outputs = ctx.method_return

    # step 1: gather multi-level features by resize and average
    feats = []
    gather_size = inputs[module.refine_level].size()[2:]
    gather_shapewarper = inputs[module.refine_level][0, 0]
    for i in range(module.num_levels):
        if i < module.refine_level:
            gathered = mmdet2trt_ops.adaptive_max_pool2d_by_input(
                inputs[i], gather_shapewarper)
        else:
            gathered = F.interpolate(
                inputs[i], size=gather_size, mode='nearest')
        feats.append(gathered)

    bsf = sum(feats) / len(feats)

    # step 2: refine gathered features
    if module.refine_type is not None:
        bsf = module.refine(bsf)

    # step 3: scatter refined features to multi-levels by a residual path
    outs = []
    for i in range(module.num_levels):
        out_size = inputs[i].size()[2:]
        out_shapewarper = inputs[i][0, 0]
        if i < module.refine_level:
            residual = F.interpolate(bsf, size=out_size, mode='nearest')
        else:
            residual = mmdet2trt_ops.adaptive_max_pool2d_by_input(
                bsf, out_shapewarper)
        outs.append(residual + inputs[i])

    for out_real, out_fake in zip(outputs, outs):
        out_real._trt = out_fake._trt

    ctx.method_return = outputs
