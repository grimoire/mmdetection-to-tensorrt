from torch2trt_dynamic.torch2trt_dynamic import *
from torch2trt_dynamic.converters import convert_float, convert_Conv2d, convert_unsqueeze, convert_pad, convert_mul, convert_tensor_getitem

import mmcv.ops


@tensorrt_converter('mmcv.ops.masked_conv.masked_conv2d')
def convert_MaskedConv(ctx):

    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs

    input = get_arg(ctx, 'features', pos=0, default=None)
    mask = get_arg(ctx, 'mask', pos=1, default=None)
    weight = get_arg(ctx, 'weight', pos=2, default=None)
    bias = get_arg(ctx, 'bias', pos=3, default=None)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    stride = get_arg(ctx, 'stride', pos=5, default=1)

    output = ctx.method_return

    ## convert conv
    conv = torch.nn.Conv2d(input.shape[1], output.shape[1], weight.shape[2:],
                           stride, padding)
    conv.weight = weight
    conv.bias = bias
    conv_input = conv(input)
    ctx.method_args = [conv, input]
    ctx.method_kwargs = {}
    ctx.method_return = conv_input
    convert_Conv2d(ctx)

    ## mask to float
    float_mask = mask.float()
    ctx.method_args = [mask]
    ctx.method_return = float_mask
    convert_float(ctx)

    ## unsqueeze mask
    unsqueeze_mask = float_mask.unsqueeze(1)
    ctx.method_args = [float_mask, 1]
    ctx.method_return = unsqueeze_mask
    convert_unsqueeze(ctx)

    ## mask pad or slice
    pad_size_h = (output.shape[2] - unsqueeze_mask.shape[2])
    pad_size_w = (output.shape[3] - unsqueeze_mask.shape[3])

    if pad_size_h == 0 and pad_size_w == 0:
        ## output shape == mask shape
        final_mask = unsqueeze_mask
    elif pad_size_h * pad_size_w >= 0 and pad_size_h >= 0 and pad_size_h >= 0:
        ## pad
        final_mask = torch.nn.functional.pad(unsqueeze_mask,
                                             (0, pad_size_w, 0, pad_size_h))
        ctx.method_args = [
            unsqueeze_mask, (pad_size_w, pad_size_w, pad_size_h, pad_size_h)
        ]
        ctx.method_return = final_mask
        convert_pad(ctx)

    elif pad_size_h * pad_size_w >= 0 and pad_size_h <= 0 and pad_size_h <= 0:
        ## slice
        # final_mask = unsqueeze_mask[:, :, -pad_size_h:pad_size_h, -pad_size_w:pad_size_w]
        # ctx.method_args = [unsqueeze_mask, (slice(None), slice(None), slice(-pad_size_h, pad_size_h, 1), slice(-pad_size_w, pad_size_w, 1))]
        final_mask = unsqueeze_mask[:, :, :pad_size_h, :pad_size_w]
        ctx.method_args = [
            unsqueeze_mask,
            (slice(None), slice(None), slice(0, pad_size_h,
                                             1), slice(0, pad_size_w, 1))
        ]
        ctx.method_return = final_mask
        convert_tensor_getitem(ctx)

    ## mul
    ctx.method_args = [conv_input, final_mask]
    ctx.method_return = output
    convert_mul(ctx)

    ## recovery
    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
