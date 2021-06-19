import torch.nn as nn
from mmdet2trt.models.builder import register_wraper


@register_wraper('mmdet.models.backbones.ResNet')
@register_wraper('mmdet.models.backbones.SSDVGG')
@register_wraper('mmdet.models.backbones.HRNet')
@register_wraper('mmdet.models.backbones.Darknet')
@register_wraper('mmdet.models.backbones.DetectoRS_ResNet')
@register_wraper('mmdet.models.backbones.HourglassNet')
@register_wraper('mmdet.models.backbones.resnext.ResNeXt')
class BaseBackboneWraper(nn.Module):

    def __init__(self, module):
        super(BaseBackboneWraper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
