import torch.nn as nn
from mmdet2trt.models.builder import register_wrapper


@register_wrapper('mmdet.models.necks.YOLOXPAFPN')
@register_wrapper('mmdet.models.necks.FPN')
@register_wrapper('mmdet.models.necks.BFP')
@register_wrapper('mmdet.models.necks.FPN_CARAFE')
@register_wrapper('mmdet.models.necks.NASFPN')
@register_wrapper('mmdet.models.necks.RFP')
@register_wrapper('mmdet.models.necks.YOLOV3Neck')
@register_wrapper('mmdet.models.necks.SSDNeck')
class BaseNeckWraper(nn.Module):

    def __init__(self, module):
        super(BaseNeckWraper, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
