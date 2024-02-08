import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wrapper, register_wrapper
from torch import nn


@register_wrapper('mmdet.models.dense_heads.YOLOXHead')
class YOLOXHeadWraper(nn.Module):

    def __init__(self, module):
        super(YOLOXHeadWraper, self).__init__()
        self.module = module
        self.prior_generator = build_wrapper(self.module.prior_generator)
        self.cls_out_channels = self.module.cls_out_channels
        iou_thr = 0.7
        if 'iou_thr' in module.test_cfg.nms:
            iou_thr = module.test_cfg.nms.iou_thr
        elif 'iou_threshold' in module.test_cfg.nms:
            iou_thr = module.test_cfg.nms.iou_threshold

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr, iou_thr, backgroundLabelId=-1)

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def forward(self, feats, x):
        num_imgs = feats[0].size(0)
        module = self.module
        cls_scores, bbox_preds, objectnesses = module(feats)

        mlvl_priors = self.prior_generator(
            [cls_score.shape[-2:] for cls_score in cls_scores],
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        flatten_cls_scores = flatten_cls_scores * flatten_objectness.unsqueeze(
            2)
        flatten_bboxes = flatten_bboxes.unsqueeze(2)

        num_bboxes = flatten_bboxes.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            flatten_cls_scores, flatten_bboxes, min(num_bboxes, 5000),
            self.test_cfg.get('max_per_img', 100))
        return num_detected, proposals, scores, cls_id
