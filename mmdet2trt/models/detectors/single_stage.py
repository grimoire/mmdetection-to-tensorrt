import torch
from mmdet2trt.models.builder import register_wraper, build_wraper
from mmdet2trt.models.dense_heads import RPNHeadWraper
import torch
from torch import nn


@register_wraper("mmdet.models.CornerNet")
@register_wraper("mmdet.models.PAA")
@register_wraper("mmdet.models.YOLOV3")
@register_wraper("mmdet.models.FSAF")
@register_wraper("mmdet.models.ATSS")
@register_wraper("mmdet.models.RepPointsDetector")
@register_wraper("mmdet.models.FOVEA")
@register_wraper("mmdet.models.FCOS")
@register_wraper("mmdet.models.RetinaNet")
@register_wraper("mmdet.models.SingleStageDetector")
class SingleStageDetectorWraper(nn.Module):
    def __init__(self, model):
        super(SingleStageDetectorWraper, self).__init__()
        self.model = model
        
        mmdet_bbox_head = self.model.bbox_head
        self.bbox_head_wraper = build_wraper(mmdet_bbox_head)

    def forward(self, x):
        model = self.model
        bbox_head = self.bbox_head_wraper

        # backbone
        feat = model.extract_feat(x)
        result = bbox_head(feat, x)

        return result

