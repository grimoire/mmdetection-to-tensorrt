import torch
from mmdet2trt.models.builder import register_wraper

from .corner_head import CornerHeadWraper


@register_wraper('mmdet.models.dense_heads.CentripetalHead')
class CentripetalHeadWraper(CornerHeadWraper):

    def __init__(self, module):
        super(CentripetalHeadWraper, self).__init__(module)

    def forward(self, feat, x):
        module = self.module
        img_meta = {
            'pad_shape': (x.shape[2], x.shape[3], 3),
            'border': (0, 0, 0, 0)
        }

        forward_results = module(feat)

        tl_heats = forward_results[0]
        br_heats = forward_results[1]
        tl_offs = forward_results[2]
        br_offs = forward_results[3]
        tl_centripetal_shifts = forward_results[6]
        br_centripetal_shifts = forward_results[7]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            tl_heat=tl_heats[-1].sigmoid(),
            br_heat=br_heats[-1].sigmoid(),
            tl_off=tl_offs[-1],
            br_off=br_offs[-1],
            tl_emb=None,
            br_emb=None,
            tl_centripetal_shift=tl_centripetal_shifts[-1],
            br_centripetal_shift=br_centripetal_shifts[-1],
            img_meta=img_meta,
            k=module.test_cfg.corner_topk,
            kernel=module.test_cfg.local_maximum_kernel,
            distance_threshold=module.test_cfg.distance_threshold)

        cls_mask = []
        for i in range(self.num_classes):
            cls_mask.append((batch_clses == i).int().float())
        cls_mask = torch.cat(cls_mask, dim=2)
        batch_scores = batch_scores * cls_mask
        batch_bboxes = batch_bboxes.unsqueeze(2)

        num_bboxes = batch_bboxes.shape[1]

        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            batch_scores, batch_bboxes, num_bboxes, self.test_cfg.max_per_img)

        return num_detected, proposals, scores, cls_id
