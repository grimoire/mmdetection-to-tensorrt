import torch
from torch import nn
from mmdet2trt.models.builder import register_warper, build_warper
from .anchor_head import AnchorHeadWarper

import mmdet2trt.ops.util_ops as mm2trt_util
from mmdet2trt.core.bbox.iou_calculators import bbox_overlaps_batched

@register_warper("mmdet.models.dense_heads.paa_head.PAAHead")
class PPAHeadWarper(AnchorHeadWarper):
    def __init__(self, module):
        super(PPAHeadWarper, self).__init__(module)


    def forward(self, feat, x):
        module = self.module

        cls_scores, bbox_preds, iou_preds = module(feat)
        
        num_levels = len(cls_scores)
        mlvl_anchors = self.anchor_generator(cls_scores, device = cls_scores[0].device)
        
        mlvl_scores = []
        mlvl_proposals = []
        mlvl_iou_preds = []
        nms_pre = self.test_cfg.get('nms_pre', -1)
        for cls_score, bbox_pred, iou_pred, anchors in zip(
                cls_scores, bbox_preds, iou_preds, mlvl_anchors):
            iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(iou_pred.shape[0],-1).sigmoid()
            scores, proposals = self.bbox_coder(cls_score, 
                                                bbox_pred, 
                                                anchors, 
                                                min_num_bboxes = -1, 
                                                num_classes = cls_score.shape[1]*4//bbox_pred.shape[1],
                                                use_sigmoid_cls = True, 
                                                input_x = x
                                                )
                             
            if nms_pre>0:
                scores=mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                iou_pred=mm2trt_util.pad_with_value(iou_pred, 1, nms_pre)
                proposals=mm2trt_util.pad_with_value(proposals, 1, nms_pre)

                max_scores, _ = (scores * iou_pred[:, :, None]).sqrt().max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                iou_pred = mm2trt_util.gather_topk(iou_pred, 1, topk_inds)
            
            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)
            mlvl_iou_preds.append(iou_pred)

        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_proposals = torch.cat(mlvl_proposals, dim=1)
        mlvl_iou_preds = torch.cat(mlvl_iou_preds, dim=1)

        mlvl_scores = (mlvl_scores*mlvl_iou_preds[:, :, None]).sqrt()
        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_proposals = mm2trt_util.gather_topk(mlvl_proposals, 1, topk_inds)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)
        
        mlvl_scores=mm2trt_util.pad_with_value(mlvl_scores, 2, 1, 0.)

        num_bboxes = mlvl_proposals.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(mlvl_scores, mlvl_proposals, num_bboxes, self.test_cfg.max_per_img)

        if module.with_score_voting:
            return self.score_voting_batched(num_detected, proposals, scores, cls_id, 
                                            mlvl_proposals, mlvl_scores, self.test_cfg.score_thr)

        return num_detected, proposals, scores, cls_id


    def score_voting_batched(self, num_detected, proposals, scores, cls_id, 
                                mlvl_bboxes, mlvl_nms_scores, score_thr):
        module = self.module
        batch_size = num_detected.size(0)
        mlvl_bboxes = mlvl_bboxes.view(batch_size,-1,4)

        proposals_voted = []
        scores_voted = []
        cls_id_voted = []
        for cls in range(module.cls_out_channels):
            candidate_cls_scores = mlvl_nms_scores[:, :, cls]
            candidate_cls_bboxes = mlvl_bboxes
            det_cls_mask = (cls_id == cls).float()
            det_cls_bboxes = proposals * det_cls_mask[..., None]
            det_cls_scores = scores * det_cls_mask

            det_candidate_ious = bbox_overlaps_batched(det_cls_bboxes,
                                                        candidate_cls_bboxes)
            pos_ious = det_candidate_ious
            pis = (torch.exp(-(1 - pos_ious)**2 / 0.025) *
                       candidate_cls_scores[:,None,:])[:, :, :, None]
            voted_bbox = torch.sum(
                pis * candidate_cls_bboxes[:, None, :, :], dim=2) / torch.sum(
                    pis, dim=2)
            proposals_voted.append(voted_bbox)
            scores_voted.append(det_cls_scores)
            cls_id_voted.append((det_cls_scores*0).int()+cls + 1)
        
        proposals_voted = torch.cat(proposals_voted, dim=1)
        scores_voted = torch.cat(scores_voted, dim=1)
        cls_id_voted = torch.cat(cls_id_voted, dim=1)

        k = proposals.size(1)
        _, topk_inds = scores_voted.topk(k, dim=1)
        scores_voted = mm2trt_util.gather_topk(scores_voted, 1, topk_inds)
        proposals_voted = mm2trt_util.gather_topk(proposals_voted, 1, topk_inds)
        cls_id_voted = mm2trt_util.gather_topk(cls_id_voted, 1, topk_inds)

        scores_mask = (scores_voted>score_thr).int()
        cls_id_voted = cls_id_voted*scores_mask + (cls_id_voted*(1-scores_mask)-1)

        return  num_detected, proposals_voted, scores_voted, cls_id_voted