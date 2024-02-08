import mmdet2trt.core.post_processing.batched_nms as batched_nms
import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.bbox import batched_distance2bbox
from mmdet2trt.models.builder import register_wrapper
from mmdet2trt.models.dense_heads.anchor_free_head import AnchorFreeHeadWraper


@register_wrapper('mmdet.models.VFNetHead')
class VFNetHeadWraper(AnchorFreeHeadWraper):

    def __init__(self, module):
        super(VFNetHeadWraper, self).__init__(module)
        self.rcnn_nms = batched_nms.BatchedNMS(
            module.test_cfg.score_thr,
            module.test_cfg.nms.iou_threshold,
            backgroundLabelId=-1)

    def _get_points_single(self, feat, stride, flatten=False):
        y, x = super()._get_points_single(feat, stride, flatten)
        if self.module.use_atss:
            points = torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride),
                dim=-1) + stride * self.module.anchor_center_offset
        else:
            points = torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride),
                dim=-1) + stride // 2
        return points

    def forward(self, feat, x):
        module = self.module
        cfg = self.test_cfg
        dense_outputs = module(feat)

        if len(dense_outputs) == 3:
            # old
            cls_scores, _, bbox_preds_refine = dense_outputs
        else:
            # new
            cls_scores, bbox_preds_refine = dense_outputs

        mlvl_points = self.get_points(cls_scores)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds_refine,
                                                mlvl_points):
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.shape[0], -1, module.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(bbox_pred.shape[0], -1, 4)
            points = points.unsqueeze(0)
            points = points.expand_as(bbox_pred[:, :, :2])
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                points = mm2trt_util.pad_with_value(points, 1, nms_pre)

                # do topk
                max_scores, _ = scores.max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                points = mm2trt_util.gather_topk(points, 1, topk_inds)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)

            bboxes = batched_distance2bbox(points, bbox_pred, x.shape[2:])
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        mlvl_scores = torch.cat(mlvl_scores, dim=1)

        mlvl_proposals = mlvl_bboxes.unsqueeze(2)

        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)

        num_bboxes = mlvl_proposals.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
