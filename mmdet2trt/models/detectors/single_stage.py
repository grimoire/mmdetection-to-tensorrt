from mmdet2trt.models.backbones import BaseBackboneWraper
from mmdet2trt.models.builder import build_wraper, register_wraper
from mmdet2trt.models.necks import BaseNeckWraper
from torch import nn


@register_wraper('mmdet.models.GFL')
@register_wraper('mmdet.models.CornerNet')
@register_wraper('mmdet.models.PAA')
@register_wraper('mmdet.models.YOLOV3')
@register_wraper('mmdet.models.FSAF')
@register_wraper('mmdet.models.ATSS')
@register_wraper('mmdet.models.RepPointsDetector')
@register_wraper('mmdet.models.FOVEA')
@register_wraper('mmdet.models.FCOS')
@register_wraper('mmdet.models.RetinaNet')
@register_wraper('mmdet.models.SingleStageDetector')
@register_wraper('mmdet.models.VFNet')
@register_wraper('mmdet.models.DETR')
class SingleStageDetectorWraper(nn.Module):

    def __init__(self, model, wrap_config={}):
        super(SingleStageDetectorWraper, self).__init__()
        self.model = model

        mmdet_backbone = self.model.backbone
        self.backbone_wraper = build_wraper(mmdet_backbone, BaseBackboneWraper)

        if self.model.with_neck:
            mmdet_neck = self.model.neck
            self.neck_wraper = build_wraper(mmdet_neck, BaseNeckWraper)

        mmdet_bbox_head = self.model.bbox_head
        self.bbox_head_wraper = build_wraper(mmdet_bbox_head)

    def extract_feat(self, img):
        x = self.backbone_wraper(img)
        if self.model.with_neck:
            x = self.neck_wraper(x)
        return x

    def forward(self, x):
        bbox_head = self.bbox_head_wraper

        # backbone
        feat = self.extract_feat(x)
        result = bbox_head(feat, x)

        return result
