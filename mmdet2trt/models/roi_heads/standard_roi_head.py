from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
import mmdet2trt.ops.util_ops as mm2trt_util

@register_wraper("mmdet.models.roi_heads.dynamic_roi_head.DynamicRoIHead")
@register_wraper("mmdet.models.roi_heads.standard_roi_head.StandardRoIHead")
class StandardRoIHeadWraper(nn.Module):
    def __init__(self, module):
        super(StandardRoIHeadWraper, self).__init__()
        self.module = module

        self.bbox_roi_extractor = build_wraper(module.bbox_roi_extractor)
        
        self.bbox_head = build_wraper(module.bbox_head, test_cfg=module.test_cfg)
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        self.test_cfg = module.test_cfg

    def _bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.shared_head is not None:
            bbox_feats = self.shared_head(bbox_feats)
        # rcnn
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    

    def forward(self, feat, proposals, img_shape):
        batch_size = proposals.shape[0]
        num_proposals = proposals.shape[1]
        rois_pad = mm2trt_util.arange_by_input(proposals, 0).unsqueeze(1)
        rois_pad = rois_pad.repeat(1, num_proposals).view(-1, 1)
        proposals = proposals.view(-1, 4)
        rois = torch.cat([rois_pad, proposals], dim=1)

        # rcnn
        bbox_results = self._bbox_forward(feat, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        xx = self.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            batch_size,
            num_proposals,
            self.test_cfg
        )
        return xx
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
