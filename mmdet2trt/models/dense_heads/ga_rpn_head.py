import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import register_wraper

from .guided_anchor_head import GuidedAnchorHeadWraper


@register_wraper('mmdet.models.GARPNHead')
class GARPNHeadWraper(GuidedAnchorHeadWraper):

    def __init__(self, module):
        super(GARPNHeadWraper, self).__init__(module)

        self.test_cfg = module.test_cfg
        if 'nms' in self.test_cfg:
            self.test_cfg.nms_thr = self.test_cfg.nms['iou_threshold']
        if 'max_per_img' in self.test_cfg:
            self.test_cfg.nms_post = self.test_cfg.max_per_img
            self.test_cfg.max_num = self.test_cfg.max_per_img
        self.rpn_nms = BatchedNMS(0.0, self.test_cfg.nms_thr, -1)

    def forward(self, feat, x):
        img_shape = x.shape[2:]
        module = self.module

        cls_scores, bbox_preds, shape_preds, loc_preds = module(feat)

        _, guided_anchors, loc_masks = self.get_anchors(
            cls_scores, shape_preds, loc_preds, use_loc_filter=True)

        mlvl_scores = []
        mlvl_proposals = []
        nms_pre = self.test_cfg.get('nms_pre', -1)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = guided_anchors[idx]
            mask = loc_masks[idx]

            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(rpn_cls_score.shape[0],
                                                      -1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(rpn_cls_score.shape[0],
                                                      -1, 2)
                scores = rpn_cls_score.softmax(dim=2)[:, :, :-1]
            scores = scores * mask

            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).reshape(
                rpn_bbox_pred.size(0), -1, 4)
            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(rpn_bbox_pred, 1,
                                                       nms_pre)
                anchors = mm2trt_util.pad_with_value(anchors, 1, nms_pre)

                # do topk
                # max_scores, _ = scores.max(dim=2)
                max_scores = scores
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                anchors = mm2trt_util.gather_topk(anchors, 1, topk_inds)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)

            proposals = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)

            scores = scores.unsqueeze(-1)
            proposals = proposals.unsqueeze(2)
            _, proposals, scores, _ = self.rpn_nms(scores, proposals,
                                                   self.test_cfg.nms_pre,
                                                   self.test_cfg.nms_post)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        scores = torch.cat(mlvl_scores, dim=1)
        proposals = torch.cat(mlvl_proposals, dim=1)

        _, topk_inds = scores.topk(self.test_cfg.max_num, dim=1)
        proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)

        return proposals
