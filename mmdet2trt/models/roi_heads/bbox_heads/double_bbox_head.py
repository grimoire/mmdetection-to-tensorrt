from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn
import torch.nn.functional as F
from .bbox_head import BBoxHeadWarper


@register_warper("mmdet.models.roi_heads.bbox_heads.double_bbox_head.DoubleConvFCBBoxHead")
class DoubleConvFCBBoxHeadWarper(BBoxHeadWarper):
    def __init__(self, module, test_cfg):
        super(DoubleConvFCBBoxHeadWarper, self).__init__(module, test_cfg)
    

    def forward(self, x_cls, x_reg):
        return self.module(x_cls, x_reg)
    



