import tensorrt as trt
from torch2trt_dynamic.plugins import (create_adaptivepool_plugin,
                                       create_meshgrid_plugin)
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


@tensorrt_converter('mmdet2trt.ops.arange_gridmesh')
def convert_arange_gridmesh(ctx):
    input = get_arg(ctx, 'x', pos=0, default=None)
    starts = get_arg(ctx, 'starts', pos=1, default=[0, 0])
    strides = get_arg(ctx, 'strides', pos=2, default=[1, 1])
    outputs = ctx.method_return
    input_trt = trt_(ctx.network, input)

    num_inputs = 0
    input_shape = input.shape
    slice_length = len(starts)
    slice_dims = list(range(len(input_shape) - slice_length, len(input_shape)))

    plugin = create_meshgrid_plugin(
        'arange_gridmesh_' + str(id(input)),
        num_inputs,
        slice_dims=slice_dims,
        starts=starts,
        strides=strides)

    layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    for idx, output in enumerate(outputs):
        output._trt = layer.get_output(idx)


@tensorrt_converter('mmdet2trt.ops.util_ops.arange_by_input')
def convert_arange_by_input(ctx):
    input = get_arg(ctx, 'x', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=0)
    start = get_arg(ctx, 'start', pos=2, default=0)
    stride = get_arg(ctx, 'stride', pos=3, default=1)

    output = ctx.method_return
    input_trt = trt_(ctx.network, input)
    num_inputs = 0
    slice_dims = [dim]
    starts = [start]
    strides = [stride]

    plugin = create_meshgrid_plugin(
        'arange_by_input_' + str(id(input)),
        num_inputs,
        slice_dims=slice_dims,
        starts=starts,
        strides=strides)

    layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output._trt = layer.get_output(0)
