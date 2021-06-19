from torch2trt_dynamic import get_arg, tensorrt_converter, trt_

from .plugins import create_gridanchordynamic_plugin


@tensorrt_converter(
    'mmdet2trt.core.anchor.anchor_generator.AnchorGeneratorSingle.forward')
def convert_AnchorGeneratorDynamic(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]

    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    ag = module
    index = ag.index
    base_size = ag.base_size
    stride = get_arg(ctx, 'stride', pos=2, default=base_size)
    if hasattr(ag.generator, 'base_anchors'):
        base_anchors = ag.generator.base_anchors[index]
        # base_anchors = base_anchors.view(-1).cpu().numpy()
        base_anchors_trt = trt_(ctx.network, base_anchors.float())

        plugin = create_gridanchordynamic_plugin(
            'ag_' + str(id(module)), stride=stride)
    else:
        print('no base_anchors in {}'.format(ag.generator))
        # scales = ag.scales.detach().cpu().numpy().astype(np.float32)
        # ratios = ag.ratios.detach().cpu().numpy().astype(np.float32)
        # scale_major = ag.scale_major
        # ctr = ag.ctr
        # if ctr is None:
        #     # center_x = -1
        #     # center_y = -1
        #     center_x = 0
        #     center_y = 0
        # else:
        #     center_x, center_y = ag.ctr

        # plugin = create_gridanchordynamic_plugin("ag_" + str(id(module)),
        #                                          base_size=base_size,
        #                                          stride=stride,
        #                                          scales=scales,
        #                                          ratios=ratios,
        #                                          scale_major=scale_major,
        #                                          center_x=center_x,
        #                                          center_y=center_y)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, base_anchors_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)
