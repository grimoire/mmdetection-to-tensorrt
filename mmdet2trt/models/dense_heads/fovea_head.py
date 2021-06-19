import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import register_wraper
from mmdet2trt.models.dense_heads.anchor_free_head import AnchorFreeHeadWraper


@register_wraper('mmdet.models.FoveaHead')
class FoveaHeadWraper(AnchorFreeHeadWraper):

    def __init__(self, module):
        super(FoveaHeadWraper, self).__init__(module)

    def forward(self, feat, x):
        img_shape = x.shape[2:]
        module = self.module
        cfg = self.test_cfg
        cls_scores, bbox_preds = module(feat)
        mlvl_points = self.get_points(cls_scores, flatten=True)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, base_len, (y, x) in zip(
                cls_scores, bbox_preds, module.strides, module.base_edge_list,
                mlvl_points):
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.shape[0], -1, module.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(bbox_pred.shape[0], -1,
                                                     4).exp()
            x = x.unsqueeze(0) + 0.5
            y = y.unsqueeze(0) + 0.5
            x = x.expand_as(bbox_pred[:, :, 0])
            y = y.expand_as(bbox_pred[:, :, 0])
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                y = mm2trt_util.pad_with_value(y, 1, nms_pre)
                x = mm2trt_util.pad_with_value(x, 1, nms_pre)

                # do topk
                max_scores, _ = (scores).max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                y = mm2trt_util.gather_topk(y, 1, topk_inds)
                x = mm2trt_util.gather_topk(x, 1, topk_inds)

            x1 = (stride * x - base_len * bbox_pred[:, :, 0]).\
                clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, :, 1]).\
                clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, :, 2]).\
                clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, :, 3]).\
                clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
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
