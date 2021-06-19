import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper('mmdet.models.RPNHead')
class RPNHeadWraper(nn.Module):

    def __init__(self, module):
        super(RPNHeadWraper, self).__init__()
        self.module = module
        self.anchor_generator = build_wraper(self.module.anchor_generator)
        self.bbox_coder = build_wraper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        if 'nms' in self.test_cfg:
            self.test_cfg.nms_thr = self.test_cfg.nms['iou_threshold']
        if 'max_per_img' in self.test_cfg:
            self.test_cfg.nms_post = self.test_cfg.max_per_img
        self.rpn_nms = BatchedNMS(0.0, self.test_cfg.nms_thr, -1)

    def forward(self, feat, x):
        module = self.module
        nms_pre = self.test_cfg.nms_pre if self.test_cfg.nms_pre > 0 else 1000
        nms_post = self.test_cfg.nms_post
        use_sigmoid_cls = module.use_sigmoid_cls

        cls_scores, bbox_preds = module(feat)

        mlvl_anchors = self.anchor_generator(
            cls_scores, device=cls_scores[0].device)

        mlvl_scores = []
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]

            scores, proposals = self.bbox_coder(
                rpn_cls_score,
                rpn_bbox_pred,
                anchors,
                min_num_bboxes=nms_pre,
                num_classes=1,
                use_sigmoid_cls=use_sigmoid_cls,
                input_x=x)

            if nms_pre > 0:
                _, topk_inds = scores.squeeze(2).topk(nms_pre, dim=1)
                proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        scores = torch.cat(mlvl_scores, dim=1)
        proposals = torch.cat(mlvl_proposals, dim=1)

        _, proposals, scores, _ = self.rpn_nms(scores, proposals,
                                               scores.size(1), nms_post)

        return proposals
