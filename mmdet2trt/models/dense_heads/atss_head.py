import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import register_wraper

from .anchor_head import AnchorHeadWraper


@register_wraper('mmdet.models.dense_heads.ATSSHead')
class ATSSHeadWraper(AnchorHeadWraper):

    def __init__(self, module):
        super(ATSSHeadWraper, self).__init__(module)

    def forward(self, feat, x):
        module = self.module

        cls_scores, bbox_preds, centernesses = module(feat)

        mlvl_anchors = self.anchor_generator(
            cls_scores, device=cls_scores[0].device)

        mlvl_scores = []
        mlvl_proposals = []
        mlvl_centerness = []
        nms_pre = self.test_cfg.get('nms_pre', -1)
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(centerness.shape[0],
                                                       -1).sigmoid()
            scores, proposals = self.bbox_coder(
                cls_score,
                bbox_pred,
                anchors,
                min_num_bboxes=-1,
                num_classes=cls_score.shape[1] * 4 // bbox_pred.shape[1],
                use_sigmoid_cls=True,
                input_x=x)

            if nms_pre > 0:
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                centerness = mm2trt_util.pad_with_value(centerness, 1, nms_pre)
                proposals = mm2trt_util.pad_with_value(proposals, 1, nms_pre)

                max_scores, _ = (scores * centerness[:, :, None]).max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                centerness = mm2trt_util.gather_topk(centerness, 1, topk_inds)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)
            mlvl_centerness.append(centerness)

        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_proposals = torch.cat(mlvl_proposals, dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # mlvl_scores = mlvl_scores*mlvl_centerness[:, :, None]
        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)

        mlvl_scores = mm2trt_util.pad_with_value(mlvl_scores, 2, 1, 0.)

        num_bboxes = mlvl_proposals.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
