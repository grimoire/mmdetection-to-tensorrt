import torch.nn as nn
from mmdet2trt.models.builder import register_wraper


@register_wraper('mmdet.models.necks.FPN')
@register_wraper('mmdet.models.necks.BFP')
@register_wraper('mmdet.models.necks.FPN_CARAFE')
@register_wraper('mmdet.models.necks.NASFPN')
@register_wraper('mmdet.models.necks.RFP')
@register_wraper('mmdet.models.necks.YOLOV3Neck')
class BaseNeckWraper(nn.Module):

    def __init__(self, module):
        super(BaseNeckWraper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
