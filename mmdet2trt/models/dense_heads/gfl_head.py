import mmdet2trt.ops.util_ops as mm2trt_util
import torch
import torch.nn.functional as F
from mmdet2trt.core.bbox.transforms import batched_distance2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import register_wraper

from .anchor_head import AnchorHeadWraper


@register_wraper('mmdet.models.GFLHead')
class GFLHeadWraper(AnchorHeadWraper):

    def __init__(self, module):
        super(GFLHeadWraper, self).__init__(module)

        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr,
            module.test_cfg.nms.iou_threshold,
            backgroundLabelId=-1)

    def batched_integral(self, intergral, x):
        batch_size = x.size(0)
        x = F.softmax(x.reshape(batch_size, -1, intergral.reg_max + 1), dim=2)
        x = F.linear(x,
                     intergral.project.type_as(x).unsqueeze(0)).reshape(
                         batch_size, -1, 4)
        return x

    def batched_anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, :, 2] + anchors[:, :, 0]) / 2
        anchors_cy = (anchors[:, :, 3] + anchors[:, :, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def forward(self, feat, x):
        img_shape = x.shape[2:]
        module = self.module
        cfg = self.test_cfg

        cls_scores, bbox_preds = module(feat)

        num_levels = len(cls_scores)
        mlvl_anchors = self.anchor_generator(
            cls_scores, device=cls_scores[0].device)

        mlvl_scores = []
        mlvl_proposals = []
        nms_pre = self.test_cfg.get('nms_pre', -1)
        for idx in range(num_levels):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            stride = module.anchor_generator.strides[idx]
            scores = rpn_cls_score.permute(0, 2, 3, 1).reshape(
                rpn_cls_score.shape[0], -1, module.cls_out_channels).sigmoid()
            bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = self.batched_integral(module.integral,
                                              bbox_pred) * stride[0]
            anchors = anchors.unsqueeze(0)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                anchors = mm2trt_util.pad_with_value(anchors, 1, nms_pre)

                # do topk
                max_scores, _ = scores.max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                anchors = mm2trt_util.gather_topk(anchors, 1, topk_inds)

            proposals = batched_distance2bbox(
                self.batched_anchor_center(anchors),
                bbox_pred,
                max_shape=img_shape)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_proposals = torch.cat(mlvl_proposals, dim=1)
        mlvl_proposals = mlvl_proposals.unsqueeze(2)

        topk_pre = max(1000, nms_pre)
        max_scores, _ = mlvl_scores.max(dim=2)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.size(1)), dim=1)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)

        num_bboxes = mlvl_proposals.shape[1]

        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
