from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
import mmdet2trt.ops.util_ops as mm2trt_util


@register_wraper("mmdet.models.roi_heads.HybridTaskCascadeRoIHead")
@register_wraper("mmdet.models.roi_heads.CascadeRoIHead")
class CascadeRoIHeadWraper(nn.Module):
    def __init__(self, module):
        super(CascadeRoIHeadWraper, self).__init__()
        self.module = module

        self.bbox_roi_extractor = [
            build_wraper(extractor) for extractor in module.bbox_roi_extractor
        ]
        self.bbox_head = [
            build_wraper(bb_head, test_cfg=module.test_cfg)
            for bb_head in module.bbox_head
        ]
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        self.test_cfg = module.test_cfg

        self.num_stages = module.num_stages

    def _bbox_forward(self, stage, x, rois):

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        if rois.shape[1] == 4:
            zeros = rois.new_zeros([rois.shape[0], 1])
            rois = torch.cat([zeros, rois], dim=1)

        roi_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = bbox_head(roi_feats)

        bbox_results = dict(cls_score=cls_score,
                            bbox_pred=bbox_pred,
                            bbox_feats=roi_feats)
        return bbox_results

    def regress_by_class(self, stage, rois, label, bbox_pred):
        bbox_head = self.bbox_head[stage]
        reg_class_agnostic = bbox_head.reg_class_agnostic

        if not reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)

        means = bbox_head.bbox_coder.means
        stds = bbox_head.bbox_coder.stds

        new_rois = delta2bbox(rois, bbox_pred, means, stds)
        return new_rois

    def forward(self, feat, proposals, img_shape):
        ms_scores = []
        batch_size = proposals.shape[0]
        num_proposals = proposals.shape[1]
        rois_pad = mm2trt_util.arange_by_input(proposals, 0).unsqueeze(1)
        rois_pad = rois_pad.repeat(1, num_proposals).view(-1, 1)
        proposals = proposals.view(-1, 4)
        rois = proposals

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i, feat, torch.cat([rois_pad, rois], dim=1))
            ms_scores.append(bbox_results['cls_score'])
            bbox_pred = bbox_results['bbox_pred']

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_pred, img_shape)

        rois = torch.cat([rois_pad, rois], dim=1)

        ### bbox_head.get_boxes
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_detections, det_boxes, det_scores, det_classes = self.bbox_head[
            -1].get_bboxes(rois, cls_score, bbox_pred, img_shape, batch_size,
                           num_proposals, self.test_cfg)
        return num_detections, det_boxes, det_scores, det_classes
