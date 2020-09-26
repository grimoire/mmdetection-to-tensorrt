from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn
import torch.nn.functional as F
from .bbox_head import BBoxHeadWraper


@register_wraper("mmdet.models.roi_heads.bbox_heads.double_bbox_head.DoubleConvFCBBoxHead")
class DoubleConvFCBBoxHeadWraper(BBoxHeadWraper):
    def __init__(self, module, test_cfg):
        super(DoubleConvFCBBoxHeadWraper, self).__init__(module, test_cfg)
    

    def forward(self, x_cls, x_reg):
        return self.module(x_cls, x_reg)
    



