import torch
from mmdet2trt.models.builder import register_warper, build_warper
from mmdet2trt.models.dense_heads import RPNHeadWarper
import torch
from torch import nn

@register_warper("mmdet.models.RetinaNet")
@register_warper("mmdet.models.SingleStageDetector")
class SingleStageDetectorWarper(nn.Module):
    def __init__(self, model):
        super(SingleStageDetectorWarper, self).__init__()
        self.model = model
        
        mmdet_bbox_head = self.model.bbox_head
        self.bbox_head_warper = build_warper(mmdet_bbox_head)

    def forward(self, x):
        model = self.model
        bbox_head = self.bbox_head_warper

        # backbone
        feat = model.extract_feat(x)
        result = bbox_head(feat, x)

        return result

