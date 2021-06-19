import mmdet2trt.ops.util_ops as mm2trt_util
import torch
import torch.nn.functional as F
from mmdet2trt.core.post_processing import merge_aug_masks
from mmdet2trt.models.builder import build_wraper, register_wraper
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox

from .cascade_roi_head import CascadeRoIHeadWraper


@register_wraper('mmdet.models.roi_heads.HybridTaskCascadeRoIHead')
class HybridTaskCascadeRoIHeadWraper(CascadeRoIHeadWraper):

    def __init__(self, module, wrap_config):
        super(HybridTaskCascadeRoIHeadWraper,
              self).__init__(module, wrap_config)

        module = self.module
        self.semantic_head = None
        if module.semantic_head is not None:
            self.semantic_roi_extractor = build_wraper(
                module.semantic_roi_extractor)
            self.semantic_head = module.semantic_head

    def _bbox_forward(self, stage, x, rois, semantic_feat=None):

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        if rois.shape[1] == 4:
            zeros = rois.new_zeros([rois.shape[0], 1])
            rois = torch.cat([zeros, rois], dim=1)

        roi_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)

        if self.module.with_semantic and 'box' in self.module.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != roi_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, roi_feats.shape[-2:])
        cls_score, bbox_pred = bbox_head(roi_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=roi_feats)
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

        if self.module.with_semantic:
            _, semantic_feat = self.semantic_head(feat)
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(
                i,
                feat,
                torch.cat([rois_pad, rois], dim=1),
                semantic_feat=semantic_feat)
            ms_scores.append(bbox_results['cls_score'])
            bbox_pred = bbox_results['bbox_pred']

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = self.bbox_head[i].regress_by_class(
                    rois, bbox_label, bbox_pred, img_shape)

        rois = torch.cat([rois_pad, rois], dim=1)

        # bbox_head.get_boxes
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_detections, det_boxes, det_scores, det_classes = self.bbox_head[
            -1].get_bboxes(rois, cls_score, bbox_pred, img_shape, batch_size,
                           num_proposals, self.test_cfg)

        result = [num_detections, det_boxes, det_scores, det_classes]

        if self.enable_mask:
            # mask roi input
            num_mask_proposals = det_boxes.size(1)
            rois_pad = mm2trt_util.arange_by_input(det_boxes, 0).unsqueeze(1)
            rois_pad = rois_pad.repeat(1, num_mask_proposals).view(-1, 1)
            mask_proposals = det_boxes.view(-1, 4)
            mask_rois = torch.cat([rois_pad, mask_proposals], dim=1)

            mask_roi_extractor = self.mask_roi_extractor[-1]

            mask_feats = mask_roi_extractor(
                feat[:mask_roi_extractor.num_inputs], mask_rois)
            if self.module.with_semantic and ('mask'
                                              in self.module.semantic_fusion):
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None

            aug_masks = []
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.module.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                mask_pred = mask_pred.sigmoid()
                aug_masks.append(mask_pred)

            mask_pred = merge_aug_masks(aug_masks, self.test_cfg)

            mc, mh, mw = mask_pred.shape[1:]
            mask_pred = mask_pred.reshape(batch_size, -1, mc, mh, mw)
            if not self.module.mask_head[-1].class_agnostic:
                det_index = det_classes.unsqueeze(-1).long()
                det_index = det_index + 1
                mask_pad = mask_pred[:, :, 0:1, ...] * 0
                mask_pred = torch.cat([mask_pad, mask_pred], dim=2)
                mask_pred = mm2trt_util.gather_topk(
                    mask_pred, dim=2, index=det_index)
                mask_pred = mask_pred.squeeze(2)

            result += [mask_pred]

        return result
