import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper('mmdet.models.dense_heads.SABLRetinaHead')
class SABLRetinaHeadWraper(nn.Module):

    def __init__(self, module):
        super(SABLRetinaHeadWraper, self).__init__()
        self.module = module
        self.square_anchor_generator = build_wraper(
            self.module.square_anchor_generator)
        self.bbox_coder = build_wraper(self.module.bbox_coder)

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.use_sigmoid_cls = self.module.use_sigmoid_cls
        self.side_num = module.side_num
        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr,
            module.test_cfg.nms.iou_threshold,
            backgroundLabelId=self.num_classes)

    def forward(self, feat, x):
        batch_size = feat[0].size(0)
        module = self.module
        img_shape = x.shape[2:]
        cfg = self.test_cfg

        cls_scores, bbox_preds = module(feat)

        mlvl_anchors = self.square_anchor_generator(
            cls_scores, device=cls_scores[0].device)

        mlvl_scores = []
        mlvl_bboxes = []
        mlvl_confids = []
        nms_pre = self.test_cfg.get('nms_pre', -1)

        bbox_cls_preds = [bb[0] for bb in bbox_preds]
        bbox_reg_preds = [bb[1] for bb in bbox_preds]
        for cls_score, bbox_cls_pred, bbox_reg_pred, anchors in zip(
                cls_scores, bbox_cls_preds, bbox_reg_preds, mlvl_anchors):
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     module.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.side_num * 4)
            bbox_reg_pred = bbox_reg_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.side_num * 4)
            anchors = anchors.unsqueeze(0).expand_as(bbox_cls_pred[:, :, :4])

            # do topk
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # pad to make sure shape>nms_pred
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_cls_pred = mm2trt_util.pad_with_value(
                    bbox_cls_pred, 1, nms_pre)
                bbox_reg_pred = mm2trt_util.pad_with_value(
                    bbox_reg_pred, 1, nms_pre)
                anchors = mm2trt_util.pad_with_value(anchors, 1, nms_pre)
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=2)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=2)

                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                bbox_cls_pred = mm2trt_util.gather_topk(
                    bbox_cls_pred, 1, topk_inds)
                bbox_reg_pred = mm2trt_util.gather_topk(
                    bbox_reg_pred, 1, topk_inds)
                anchors = mm2trt_util.gather_topk(anchors, 1, topk_inds)

            bbox_preds = [
                bbox_cls_pred.contiguous(),
                bbox_reg_pred.contiguous()
            ]

            bboxes, confids = self.bbox_coder.decode(
                anchors.contiguous(), bbox_preds, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_confids.append(confids)
        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_confids = torch.cat(mlvl_confids, dim=1)

        mlvl_bboxes = mlvl_bboxes.unsqueeze(2)
        mlvl_scores = mlvl_scores * mlvl_confids.unsqueeze(-1)

        max_scores, _ = mlvl_scores.max(dim=2)
        topk_pre = max(1000, nms_pre)
        _, topk_inds = max_scores.topk(
            min(topk_pre, mlvl_scores.shape[1]), dim=1)
        mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)
        mlvl_bboxes = mm2trt_util.gather_topk(mlvl_bboxes, 1, topk_inds)

        if self.use_sigmoid_cls:
            padding = mlvl_scores[:, :, :1] * 0
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=2)
        # if not self.use_sigmoid_cls:
        #     mlvl_scores = mlvl_scores[:,:,:-1]
        mlvl_bboxes = mlvl_bboxes.repeat(1, 1, self.num_classes + 1, 1)

        num_bboxes = mlvl_bboxes.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_bboxes, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
