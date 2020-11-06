from torch2trt_dynamic.torch2trt_dynamic import *


@tensorrt_converter('mmdet.models.VFNetHead.star_dcn_offset', is_real=False)
def convert_vfnet_star_dcn_offset(ctx):
    self = ctx.method_args[0]
    bbox_pred = get_arg(ctx, 'bbox_pred', pos=1, default=None)
    gradient_mul = get_arg(ctx, 'gradient_mul', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=None)

    output = ctx.method_return

    dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)

    bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
        gradient_mul * bbox_pred
    # map to the feature map scale
    bbox_pred_grad_mul = bbox_pred_grad_mul / stride
    N, C, H, W = bbox_pred.size()

    x1 = bbox_pred_grad_mul[:, 0, :, :]
    y1 = bbox_pred_grad_mul[:, 1, :, :]
    x2 = bbox_pred_grad_mul[:, 2, :, :]
    y2 = bbox_pred_grad_mul[:, 3, :, :]

    bbox_pred_grad_mul_offset_zero = y1 * 0.
    bbox_pred_grad_mul_offset0 = -1.0 * y1  # -y1
    bbox_pred_grad_mul_offset1 = -1.0 * x1  # -x1
    bbox_pred_grad_mul_offset2 = -1.0 * y1  # -y1
    bbox_pred_grad_mul_offset3 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset4 = -1.0 * y1  # -y1
    bbox_pred_grad_mul_offset5 = x2  # x2
    bbox_pred_grad_mul_offset6 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset7 = -1.0 * x1  # -x1
    bbox_pred_grad_mul_offset8 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset9 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset10 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset11 = x2  # x2
    bbox_pred_grad_mul_offset12 = y2  # y2
    bbox_pred_grad_mul_offset13 = -1.0 * x1  # -x1
    bbox_pred_grad_mul_offset14 = y2  # y2
    bbox_pred_grad_mul_offset15 = bbox_pred_grad_mul_offset_zero
    bbox_pred_grad_mul_offset16 = y2  # y2
    bbox_pred_grad_mul_offset17 = x2  # x2

    bbox_pred_grad_mul_offset = [
        bbox_pred_grad_mul_offset0, bbox_pred_grad_mul_offset1,
        bbox_pred_grad_mul_offset2, bbox_pred_grad_mul_offset3,
        bbox_pred_grad_mul_offset4, bbox_pred_grad_mul_offset5,
        bbox_pred_grad_mul_offset6, bbox_pred_grad_mul_offset7,
        bbox_pred_grad_mul_offset8, bbox_pred_grad_mul_offset9,
        bbox_pred_grad_mul_offset10, bbox_pred_grad_mul_offset11,
        bbox_pred_grad_mul_offset12, bbox_pred_grad_mul_offset13,
        bbox_pred_grad_mul_offset14, bbox_pred_grad_mul_offset15,
        bbox_pred_grad_mul_offset16, bbox_pred_grad_mul_offset17
    ]

    bbox_pred_grad_mul_offset = bbox_pred_grad_mul_offset + [
        bbox_pred_grad_mul_offset_zero
    ] * (2 * self.num_dconv_points - len(bbox_pred_grad_mul_offset))
    bbox_pred_grad_mul_offset = torch.stack(bbox_pred_grad_mul_offset, dim=1)
    dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

    output._trt = dcn_offset._trt
    ctx.method_return = output