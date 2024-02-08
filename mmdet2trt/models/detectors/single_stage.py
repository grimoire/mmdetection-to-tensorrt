from mmdet2trt.models.backbones import BaseBackboneWraper
from mmdet2trt.models.builder import build_wrapper, register_wrapper
from mmdet2trt.models.necks import BaseNeckWraper
from torch import nn


@register_wrapper('mmdet.models.YOLOX')
@register_wrapper('mmdet.models.GFL')
@register_wrapper('mmdet.models.CornerNet')
@register_wrapper('mmdet.models.PAA')
@register_wrapper('mmdet.models.YOLOV3')
@register_wrapper('mmdet.models.FSAF')
@register_wrapper('mmdet.models.ATSS')
@register_wrapper('mmdet.models.RepPointsDetector')
@register_wrapper('mmdet.models.FOVEA')
@register_wrapper('mmdet.models.FCOS')
@register_wrapper('mmdet.models.RetinaNet')
@register_wrapper('mmdet.models.SingleStageDetector')
@register_wrapper('mmdet.models.VFNet')
@register_wrapper('mmdet.models.DETR')
class SingleStageDetectorWraper(nn.Module):

    def __init__(self, model, wrap_config={}):
        super(SingleStageDetectorWraper, self).__init__()
        self.model = model

        mmdet_backbone = self.model.backbone
        self.backbone = build_wrapper(mmdet_backbone, BaseBackboneWraper)

        if self.model.with_neck:
            mmdet_neck = self.model.neck
            self.neck = build_wrapper(mmdet_neck, BaseNeckWraper)

        mmdet_bbox_head = self.model.bbox_head
        self.bbox_head = build_wrapper(mmdet_bbox_head)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.model.with_neck:
            x = self.neck(x)
        return x

    def forward(self, x):
        # backbone
        feat = self.extract_feat(x)
        result = self.bbox_head(feat, x)

        return result
