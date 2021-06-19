import torch
import torch.nn.functional as F
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter

import mmcv.cnn
import mmcv.ops


@tensorrt_converter('mmcv.ops.saconv.SAConv2d.forward', is_real=False)
def convert_SAConv2d(ctx):
    self = ctx.method_args[0]
    x = ctx.method_args[1]
    output = ctx.method_return

    avg_x = F.adaptive_avg_pool2d(x, output_size=1)
    avg_x = self.pre_context(avg_x)
    avg_x = avg_x.expand_as(x)
    x = x + avg_x
    # switch
    # avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
    avg_x_w = torch.cat(
        [x[:, :, :, 1:3].flip(-1), x, x[:, :, :, -3:-1].flip(-1)], dim=3)
    avg_x = torch.cat([
        avg_x_w[:, :, 1:3, :].flip(-2), avg_x_w, avg_x_w[:, :,
                                                         -3:-1, :].flip(-2)
    ],
                      dim=2)

    avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
    switch = self.switch(avg_x)
    # sac
    weight = self._get_weight(self.weight)
    if self.use_deform:
        offset = self.offset_s(avg_x)
        out_s = mmcv.ops.deform_conv.deform_conv2d(x, offset, weight,
                                                   self.stride, self.padding,
                                                   self.dilation, self.groups,
                                                   1)
    else:
        out_s = self.super().conv2d_forward(x, weight)
    ori_p = self.padding
    ori_d = self.dilation
    self.padding = tuple(3 * p for p in self.padding)
    self.dilation = tuple(3 * d for d in self.dilation)
    weight = weight + self.weight_diff
    if self.use_deform:
        offset = self.offset_l(avg_x)
        out_l = mmcv.ops.deform_conv.deform_conv2d(x, offset, weight,
                                                   self.stride, self.padding,
                                                   self.dilation, self.groups,
                                                   1)
    else:
        out_l = self.super().conv2d_forward(x, weight)
    out = switch * out_s + (1 - switch) * out_l
    self.padding = ori_p
    self.dilation = ori_d
    # post-context
    avg_x = F.adaptive_avg_pool2d(out, output_size=1)
    avg_x = self.post_context(avg_x)
    avg_x = avg_x.expand_as(out)
    out = out + avg_x

    output._trt = out._trt
    ctx.method_return = output
