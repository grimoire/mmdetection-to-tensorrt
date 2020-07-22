import torch
from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn

import mmdet2trt.ops.util_ops as mm2trt_util
from mmdet2trt.models.dense_heads.anchor_free_head import AnchorFreeHeadWarper
from mmdet2trt.core.bbox import batched_distance2bbox

import mmdet2trt.core.post_processing.batched_nms as batched_nms
import mmdet2trt


@register_warper("mmdet.models.FCOSHead")
class FCOSHeadWarper(AnchorFreeHeadWarper):

    def __init__(self, module):
        super(FCOSHeadWarper, self).__init__(module)
        self.rcnn_nms = batched_nms.BatchedNMS(module.test_cfg.score_thr, module.test_cfg.nms.iou_threshold, backgroundLabelId = -1)


    def _get_points_single(self, feat, stride, flatten=False):
        y, x = super()._get_points_single(feat, stride, flatten)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def forward(self, feat, x):
        module = self.module
        cfg = self.test_cfg
        cls_scores, bbox_preds, centernesses = module(feat)
        mlvl_points = self.get_points(cls_scores)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            scores = cls_score.permute(0, 2, 3, 1).reshape(cls_score.shape[0],
                -1, module.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3, 1).reshape(centerness.shape[0],-1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.shape[0], -1, 4)
            points = points.unsqueeze(0)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # concate zero to enable topk, dirty way, will find a better way in future
                score_pad = (scores[:,0:1,...]*0.).repeat(1,nms_pre,1)
                scores = torch.cat([scores, score_pad], dim=1)
                centerness_pad = centerness[:,0:1].repeat(1,nms_pre)
                centerness = torch.cat([centerness, centerness_pad], dim=1)
                bbox_pred_pad = bbox_pred[:,0:1, ...].repeat(1,nms_pre,1)
                bbox_pred = torch.cat([bbox_pred, bbox_pred_pad], dim=1)
                points_pad = points[:,0:1, ...].repeat(1,nms_pre,1)
                points = torch.cat([points, points_pad], dim=1)

                # do topk
                max_scores, _ = (scores * centerness[:, :, None]).max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre)
                points_gather = topk_inds.unsqueeze(-1).repeat(1,1,points.shape[2])
                points = points.gather(1, points_gather)
                bbox_pred_gather = topk_inds.unsqueeze(-1).repeat(1,1,bbox_pred.shape[2])
                bbox_pred = bbox_pred.gather(1, bbox_pred_gather)
                scores_gather = topk_inds.unsqueeze(-1).repeat(1,1,scores.shape[2])
                scores = scores.gather(1, scores_gather)
                centerness = centerness.gather(1, topk_inds)
                # points = points[:, topk_inds, :]
                # bbox_pred = bbox_pred[:, topk_inds, :]
                # scores = scores[:, topk_inds, :]
                # centerness = centerness[:, topk_inds]
            bboxes=batched_distance2bbox(points, bbox_pred, x.shape[2:])
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)


        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        mlvl_scores = torch.cat(mlvl_scores, dim=1)

        mlvl_proposals = mlvl_bboxes.unsqueeze(2)

        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(min(topk_pre, mlvl_scores.shape[1]), dim=1)
        proposal_topk_inds = topk_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mlvl_proposals.shape[2], mlvl_proposals.shape[3])
        mlvl_proposals = mlvl_proposals.gather(1, proposal_topk_inds)
        score_topk_inds = topk_inds.unsqueeze(-1).repeat(1, 1, mlvl_scores.shape[2])
        mlvl_scores = mlvl_scores.gather(1, score_topk_inds)

        mlvl_proposals = mlvl_proposals.repeat(1,1,mlvl_scores.shape[2],1)
        num_bboxes = mlvl_proposals.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id



