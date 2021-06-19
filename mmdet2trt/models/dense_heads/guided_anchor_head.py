import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper('mmdet.models.GARetinaHead')
class GuidedAnchorHeadWraper(nn.Module):

    def __init__(self, module):
        super(GuidedAnchorHeadWraper, self).__init__()
        self.module = module
        self.loc_filter_thr = module.loc_filter_thr
        self.num_anchors = module.num_anchors
        self.square_anchor_generator = build_wraper(
            self.module.square_anchor_generator)
        self.anchor_coder = build_wraper(self.module.anchor_coder)
        self.bbox_coder = build_wraper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.use_sigmoid_cls = self.module.use_sigmoid_cls
        if ('score_thr' in module.test_cfg) and (
                'nms' in module.test_cfg) and ('iou_threshold'
                                               in module.test_cfg.nms):
            self.rcnn_nms = BatchedNMS(
                module.test_cfg.score_thr,
                module.test_cfg.nms.iou_threshold,
                backgroundLabelId=self.num_classes)

    def get_anchors(self,
                    cls_scores,
                    shape_preds,
                    loc_preds,
                    use_loc_filter=False):

        multi_level_squares = self.square_anchor_generator(
            cls_scores, device=cls_scores[0].device)
        num_levels = len(cls_scores)

        guided_anchors_list = []
        loc_mask_list = []
        for i in range(num_levels):
            squares = multi_level_squares[i]
            shape_pred = shape_preds[i]
            loc_pred = loc_preds[i]
            guided_anchors, loc_mask = self._get_guided_anchors(
                squares, shape_pred, loc_pred, use_loc_filter)
            guided_anchors_list.append(guided_anchors)
            loc_mask_list.append(loc_mask)

        return multi_level_squares, guided_anchors_list, loc_mask_list

    def _get_guided_anchors(self,
                            squares,
                            shape_pred,
                            loc_pred,
                            use_loc_filter=False):

        loc_pred = loc_pred.sigmoid()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(0, 2, 3,
                                1).float().expand(-1, -1, -1, self.num_anchors)
        mask = mask.view(mask.shape[0], -1)
        # calculate guided anchors
        squares = squares.unsqueeze(0)
        anchor_deltas = shape_pred.permute(0, 2, 3, 1).contiguous().view(
            shape_pred.shape[0], -1, 2)
        zeros = anchor_deltas[:, :, :2] * 0.
        bbox_deltas = torch.cat([zeros, anchor_deltas], dim=2)
        guided_anchors = self.anchor_coder.decode(
            squares, bbox_deltas, wh_ratio_clip=1e-6)

        return guided_anchors, mask

    def forward(self, feat, x):
        img_shape = x.shape[2:]
        module = self.module
        cfg = self.test_cfg

        cls_scores, bbox_preds, shape_preds, loc_preds = module(feat)

        _, mlvl_anchors, mlvl_masks = self.get_anchors(
            cls_scores, shape_preds, loc_preds, use_loc_filter=True)

        mlvl_scores = []
        mlvl_proposals = []
        nms_pre = cfg.get('nms_pre', -1)
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):

            scores = cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.shape[0], -1, module.cls_out_channels).sigmoid()
            if module.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(-1)

            scores = scores * mask.unsqueeze(2)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(bbox_pred.shape[0], -1, 4)

            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                anchors = mm2trt_util.pad_with_value(anchors, 1, nms_pre)

                # do topk
                max_scores, _ = (scores).max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                anchors = mm2trt_util.gather_topk(anchors, 1, topk_inds)

            proposals = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_proposals = torch.cat(mlvl_proposals, dim=1)
        mlvl_proposals = mlvl_proposals.unsqueeze(2)

        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)

        num_bboxes = mlvl_proposals.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
