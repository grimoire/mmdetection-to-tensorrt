import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import build_wrapper, register_wrapper
from mmdet2trt.models.dense_heads.anchor_free_head import AnchorFreeHeadWraper


@register_wrapper('mmdet.models.RepPointsHead')
class RepPointsHeadWraper(AnchorFreeHeadWraper):

    def __init__(self, module):
        super(RepPointsHeadWraper, self).__init__(module)

        if hasattr(self.module, 'prior_generator'):
            # mmdet 2.18
            self.prior_generator = build_wrapper(self.module.prior_generator)
        elif hasattr(self.module, 'point_generators'):
            # mmdet 2.10
            self.point_generators = [
                build_wrapper(generator)
                for generator in self.module.point_generators
            ]
        else:
            self.point_generator = build_wrapper(self.module.point_generator)

    def forward(self, feat, x):
        img_shape = x.shape[2:]
        module = self.module
        cfg = self.test_cfg

        dense_outputs = module(feat)

        if len(dense_outputs) == 3:
            # old
            cls_scores, _, pts_preds_refine = dense_outputs
            bbox_preds_refine = [
                module.points2bbox(pts_pred_refine)
                for pts_pred_refine in pts_preds_refine
            ]
        else:
            # mmdet 2.18+
            cls_scores, pts_preds_refine = dense_outputs

        bbox_preds_refine = pts_preds_refine

        if hasattr(self.module, 'prior_generator'):
            # mmdet 2.18
            featmap_sizes = [
                cls_scores[i].size()[-2:] for i in range(len(cls_scores))
            ]
            mlvl_points = self.prior_generator.forward(featmap_sizes,
                                                       cls_scores[0].device)
        elif hasattr(self.module, 'point_generators'):
            # mmdet 2.10
            num_levels = len(cls_scores)
            mlvl_points = [
                self.point_generators[i](cls_scores[i],
                                         module.point_strides[i])
                for i in range(num_levels)
            ]
        else:
            featmap_sizes = [
                cls_scores[i].size()[-2:] for i in range(len(cls_scores))
            ]
            mlvl_points = self.point_generator.forward(featmap_sizes,
                                                       cls_scores[0].device)

        mlvl_bboxes = []
        mlvl_scores = []
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds_refine, mlvl_points)):
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                cls_score.shape[0], -1, module.cls_out_channels).sigmoid()
            if module.use_sigmoid_cls:
                scores = scores.sigmoid()
            else:
                scores = scores.softmax(-1)[:, :, :-1]
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(bbox_pred.shape[0], -1, 4)
            points = points[:, :2].unsqueeze(0).expand_as(bbox_pred[:, :, :2])

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0:
                # concate zero to enable topk,
                # dirty way, will find a better way in future
                scores = mm2trt_util.pad_with_value(scores, 1, nms_pre, 0.)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                points = mm2trt_util.pad_with_value(points, 1, nms_pre)
                max_scores, _ = (scores).max(dim=2)
                _, topk_inds = max_scores.topk(nms_pre, dim=1)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)
                points = mm2trt_util.gather_topk(points, 1, topk_inds)

            bbox_pos_center = torch.cat([points[:, :, :2], points[:, :, :2]],
                                        dim=2)
            bboxes = bbox_pred * module.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, :, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, :, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, :, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, :, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        mlvl_scores = torch.cat(mlvl_scores, dim=1)
        mlvl_bboxes = mlvl_bboxes.unsqueeze(2)

        # topk again
        if nms_pre > 0:
            max_scores, _ = (mlvl_scores).max(dim=2)
            _, topk_inds = max_scores.topk(nms_pre, dim=1)
            mlvl_bboxes = mm2trt_util.gather_topk(mlvl_bboxes, 1, topk_inds)
            mlvl_scores = mm2trt_util.gather_topk(mlvl_scores, 1, topk_inds)

        num_bboxes = mlvl_bboxes.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            mlvl_scores, mlvl_bboxes, num_bboxes, self.test_cfg.max_per_img)
        return num_detected, proposals, scores, cls_id
