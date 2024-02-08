from mmdet2trt.models.backbones import BaseBackboneWraper
from mmdet2trt.models.builder import build_wrapper, register_wrapper
from mmdet2trt.models.dense_heads import RPNHeadWraper
from mmdet2trt.models.necks import BaseNeckWraper
from mmdet2trt.models.roi_heads import StandardRoIHeadWraper
from torch import nn


@register_wrapper('mmdet.models.MaskScoringRCNN')
@register_wrapper('mmdet.models.GridRCNN')
@register_wrapper('mmdet.models.HybridTaskCascade')
@register_wrapper('mmdet.models.MaskRCNN')
@register_wrapper('mmdet.models.CascadeRCNN')
@register_wrapper('mmdet.models.FasterRCNN')
@register_wrapper('mmdet.models.TwoStageDetector')
class TwoStageDetectorWraper(nn.Module):

    def __init__(self, model, wrap_config={}):
        super(TwoStageDetectorWraper, self).__init__()
        self.model = model

        mmdet_backbone = self.model.backbone
        self.backbone_wraper = build_wrapper(mmdet_backbone,
                                             BaseBackboneWraper)

        if self.model.with_neck:
            mmdet_neck = self.model.neck
            self.neck_wraper = build_wrapper(mmdet_neck, BaseNeckWraper)

        mmdet_rpn_head = self.model.rpn_head
        self.rpn_head_wraper = build_wrapper(mmdet_rpn_head, RPNHeadWraper)

        mmdet_roi_head = self.model.roi_head
        self.roi_head_wraper = build_wrapper(
            mmdet_roi_head, StandardRoIHeadWraper, wrap_config=wrap_config)

    def extract_feat(self, img):
        x = self.backbone_wraper(img)
        if self.model.with_neck:
            x = self.neck_wraper(x)
        return x

    def forward(self, x):
        rpn_head = self.rpn_head_wraper

        # backbone
        feat = self.extract_feat(x)
        rois = rpn_head(feat, x)

        result = self.roi_head_wraper(feat, rois, x.shape[2:])
        return result
