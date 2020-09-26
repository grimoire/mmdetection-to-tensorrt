from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
import mmdet2trt.ops.util_ops as mm2trt_util
from .standard_roi_head import StandardRoIHeadWraper

@register_wraper("mmdet.models.roi_heads.double_roi_head.DoubleHeadRoIHead")
class DoubleHeadRoIHeadWraper(StandardRoIHeadWraper):
    def __init__(self, module):
        super(DoubleHeadRoIHeadWraper, self).__init__(module)

        self.reg_roi_scale_factor = self.module.reg_roi_scale_factor
        
    def _bbox_forward(self, x, rois):

        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.shared_head is not None:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)

        # rcnn
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_cls_feats)
        return bbox_results


    def forward(self, feat ,proposals, img_shape):
        batch_size = proposals.shape[0]
        num_proposals = proposals.shape[1]
        rois_pad = mm2trt_util.arange_by_input(proposals, 0).unsqueeze(1)
        rois_pad = rois_pad.repeat(1, num_proposals).view(-1, 1)
        proposals = proposals.view(-1, 4)
        rois = torch.cat([rois_pad, proposals], dim=1)

        bbox_results = self._bbox_forward(feat, rois)
        
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_detections, det_boxes, det_scores, det_classes = self.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            batch_size,
            num_proposals,
            self.test_cfg
        )

        return num_detections, det_boxes, det_scores, det_classes
