import tensorrt as trt
from torch2trt_dynamic.plugins import create_adaptivepool_plugin
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('mmdet2trt.ops.adaptive_max_pool2d_by_input')
def convert_adaptive_max_pool2d_by_input(ctx):
    input = get_arg(ctx, 'x', pos=0, default=None)
    shape_wraper = get_arg(ctx, 'shape_wraper', pos=1, default=None)
    output = ctx.method_return

    output_size = shape_wraper.shape
    input_trt = trt_(ctx.network, input)
    wrapshape_trt = trt_(ctx.network, shape_wraper)

    plugin = create_adaptivepool_plugin(
        'adaptive_max_pool2d_by_input_' + str(id(input)),
        output_size=output_size,
        pooling_type=trt.PoolingType.MAX)

    layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, wrapshape_trt], plugin=plugin)

    output._trt = layer.get_output(0)
