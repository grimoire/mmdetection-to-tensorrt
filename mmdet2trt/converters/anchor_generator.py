from torch2trt import tensorrt_converter, trt_
import torch
from torch import nn
import numpy as np

from .plugins import create_gridanchordynamic_plugin

@tensorrt_converter("mmdet2trt.core.anchor.anchor_generator.AnchorGeneratorSingle.forward")
def convert_AnchorGeneratorDynamic(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    ag = module
    base_size = ag.base_size
    if len(ctx.method_args)>2:
        stride = ctx.method_args[2]
    else:
        stride = base_size
    scales = ag.scales.detach().cpu().numpy().astype(np.float32)
    ratios = ag.ratios.detach().cpu().numpy().astype(np.float32)
    scale_major = ag.scale_major
    ctr = ag.ctr
    if ctr is None:
        # center_x = -1
        # center_y = -1
        center_x = 0
        center_y = 0
    else:
        center_x, center_y = ag.ctr

    plugin = create_gridanchordynamic_plugin("ag_" + str(id(module)),
                                    base_size=base_size,
                                    stride=stride,
                                    scales=scales,
                                    ratios=ratios,
                                    scale_major=scale_major,
                                    center_x=center_x,
                                    center_y=center_y)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
