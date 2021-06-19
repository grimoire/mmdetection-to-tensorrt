import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper('mmdet.models.dense_heads.FSAFHead')
@register_wraper('mmdet.models.RetinaSepBNHead')
@register_wraper('mmdet.models.FreeAnchorRetinaHead')
@register_wraper('mmdet.models.RetinaHead')
@register_wraper('mmdet.models.SSDHead')
@register_wraper('mmdet.models.AnchorHead')
class AnchorHeadWraper(nn.Module):

    def __init__(self, module):
        super(AnchorHeadWraper, self).__init__()
        self.module = module
        self.anchor_generator = build_wraper(self.module.anchor_generator)
        self.bbox_coder = build_wraper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.use_sigmoid_cls = self.module.use_sigmoid_cls
        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr,
            module.test_cfg.nms.iou_threshold,
            backgroundLabelId=self.num_classes)

    def forward(self, feat, x):
        module = self.module

        cls_scores, bbox_preds = module(feat)

        mlvl_anchors = self.anchor_generator(
            cls_scores, device=cls_scores[0].device)

        mlvl_scores = []
        mlvl_proposals = []
        nms_pre = self.test_cfg.get('nms_pre', -1)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            scores, proposals = self.bbox_coder(
                rpn_cls_score,
                rpn_bbox_pred,
                anchors,
                min_num_bboxes=nms_pre,
                num_classes=rpn_cls_score.shape[1] * 4 //
                rpn_bbox_pred.shape[1],
                use_sigmoid_cls=self.use_sigmoid_cls,
                input_x=x)

            if nms_pre > 0:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=2)
                else:
                    max_scores, _ = scores[:, :, :-1].max(dim=2)

                _, topk_inds = max_scores.topk(nms_pre, dim=1)

                proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_proposals = torch.cat(mlvl_proposals, dim=1)

        if self.use_sigmoid_cls:
            max_scores, _ = mlvl_scores.max(dim=2)
        else:
            max_scores, _ = mlvl_scores[:, :, :mlvl_scores.shape[2] -
                                        1].max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)

        if self.use_sigmoid_cls:
            padding = mlvl_scores[:, :, :1] * 0
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=2)

        num_bboxes = mlvl_proposals.shape[1]

        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
