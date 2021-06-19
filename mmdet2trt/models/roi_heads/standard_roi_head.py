import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper(
    'mmdet.models.roi_heads.mask_scoring_roi_head.MaskScoringRoIHead')
@register_wraper('mmdet.models.roi_heads.dynamic_roi_head.DynamicRoIHead')
@register_wraper('mmdet.models.roi_heads.standard_roi_head.StandardRoIHead')
class StandardRoIHeadWraper(nn.Module):

    def __init__(self, module, wrap_config={}):
        super(StandardRoIHeadWraper, self).__init__()
        self.module = module
        self.wrap_config = wrap_config

        self.bbox_roi_extractor = build_wraper(module.bbox_roi_extractor)

        self.bbox_head = build_wraper(
            module.bbox_head, test_cfg=module.test_cfg)
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        # init mask if exist
        self.enable_mask = False
        if 'enable_mask' in wrap_config and wrap_config[
                'enable_mask'] and module.with_mask:
            self.enable_mask = True
            self.init_mask_head(module.mask_roi_extractor, module.mask_head)

        self.test_cfg = module.test_cfg

    def init_mask_head(self, mask_roi_extractor, mask_head):
        self.mask_roi_extractor = build_wraper(mask_roi_extractor)
        self.mask_head = build_wraper(mask_head, test_cfg=self.module.test_cfg)

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

    def _mask_forward(self, x, rois):
        mask_feats = self.mask_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.shared_head is not None:
            mask_feats = self.shared_head(mask_feats)

        # mask forward
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

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

        bbox_head_outputs = self.bbox_head.get_bboxes(rois, cls_score,
                                                      bbox_pred, img_shape,
                                                      batch_size,
                                                      num_proposals,
                                                      self.test_cfg)

        num_detections, det_boxes, det_scores, det_classes = bbox_head_outputs
        result = [num_detections, det_boxes, det_scores, det_classes]

        if self.enable_mask:
            # mask roi input
            num_mask_proposals = det_boxes.size(1)
            rois_pad = mm2trt_util.arange_by_input(det_boxes, 0).unsqueeze(1)
            rois_pad = rois_pad.repeat(1, num_mask_proposals).view(-1, 1)
            mask_proposals = det_boxes.view(-1, 4)
            mask_rois = torch.cat([rois_pad, mask_proposals], dim=1)

            mask_results = self._mask_forward(feat, mask_rois)
            mask_pred = mask_results['mask_pred']

            mc, mh, mw = mask_pred.shape[1:]
            mask_pred = mask_pred.reshape(batch_size, -1, mc, mh, mw).sigmoid()
            if not self.module.mask_head.class_agnostic:
                det_index = det_classes.unsqueeze(-1).long()
                det_index = det_index + 1
                mask_pad = mask_pred[:, :, 0:1, ...] * 0
                mask_pred = torch.cat([mask_pad, mask_pred], dim=2)
                mask_pred = mm2trt_util.gather_topk(
                    mask_pred, dim=2, index=det_index)
                mask_pred = mask_pred.squeeze(2)

            result += [mask_pred]

        return result
