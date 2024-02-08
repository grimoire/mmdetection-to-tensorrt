import torch.nn as nn
from mmdet2trt.models.builder import register_wrapper


@register_wrapper('mmdet.models.backbones.CSPDarknet')
@register_wrapper('mmdet.models.backbones.MobileNetV2')
@register_wrapper('mmdet.models.backbones.ResNet')
@register_wrapper('mmdet.models.backbones.SSDVGG')
@register_wrapper('mmdet.models.backbones.HRNet')
@register_wrapper('mmdet.models.backbones.Darknet')
@register_wrapper('mmdet.models.backbones.DetectoRS_ResNet')
@register_wrapper('mmdet.models.backbones.HourglassNet')
@register_wrapper('mmdet.models.backbones.resnext.ResNeXt')
class BaseBackboneWraper(nn.Module):

    def __init__(self, module):
        super(BaseBackboneWraper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
