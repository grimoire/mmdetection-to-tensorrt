import torch
from mmdet2trt.models.builder import register_wraper, build_wraper
from mmdet2trt.models.dense_heads import RPNHeadWraper
from mmdet2trt.models.roi_heads import StandardRoIHeadWraper
import torch
from torch import nn

@register_wraper("mmdet.models.HybridTaskCascade")
@register_wraper("mmdet.models.MaskRCNN")
@register_wraper("mmdet.models.CascadeRCNN")
@register_wraper("mmdet.models.FasterRCNN")
@register_wraper("mmdet.models.TwoStageDetector")
class TwoStageDetectorWraper(nn.Module):
    def __init__(self, model):
        super(TwoStageDetectorWraper, self).__init__()
        self.model = model
        
        mmdet_rpn_head = self.model.rpn_head
        self.rpn_head_wraper = build_wraper(mmdet_rpn_head, RPNHeadWraper)

        mmdet_roi_head = self.model.roi_head
        self.roi_head_wraper = build_wraper(mmdet_roi_head, StandardRoIHeadWraper)

    def forward(self, x):
        model = self.model
        rpn_head = self.rpn_head_wraper

        # backbone
        feat = model.extract_feat(x)
        rois = rpn_head(feat, x)

        result = self.roi_head_wraper(feat, rois, x.shape[2:])
        return result

