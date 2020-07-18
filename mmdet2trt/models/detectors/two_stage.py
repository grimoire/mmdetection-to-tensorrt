import torch
from mmdet2trt.models.builder import register_warper, build_warper
from mmdet2trt.models.dense_heads import RPNHeadWarper
from mmdet2trt.models.roi_heads import StandardRoIHeadWarper
import torch
from torch import nn

@register_warper("mmdet.models.CascadeRCNN")
@register_warper("mmdet.models.FasterRCNN")
@register_warper("mmdet.models.TwoStageDetector")
class TwoStageDetectorWarper(nn.Module):
    def __init__(self, model):
        super(TwoStageDetectorWarper, self).__init__()
        self.model = model
        
        mmdet_rpn_head = self.model.rpn_head
        self.rpn_head_warper = build_warper(mmdet_rpn_head, RPNHeadWarper)

        mmdet_roi_head = self.model.roi_head
        self.roi_head_warper = build_warper(mmdet_roi_head, StandardRoIHeadWarper)

    def forward(self, x):
        model = self.model
        rpn_head = self.rpn_head_warper

        # backbone
        feat = model.extract_feat(x)
        rois = rpn_head(feat, x)

        result = self.roi_head_warper(feat, rois)
        return result

