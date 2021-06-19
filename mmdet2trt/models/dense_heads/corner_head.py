import torch
import torch.nn.functional as F
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import register_wraper
from torch import nn


@register_wraper('mmdet.models.CornerHead')
class CornerHeadWraper(nn.Module):

    def __init__(self, module):
        super(CornerHeadWraper, self).__init__()
        self.module = module
        self.num_classes = module.num_classes
        self.test_cfg = module.test_cfg

        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr,
            module.test_cfg.nms_cfg.iou_threshold,
            backgroundLabelId=-1)

    def forward(self, feat, x):
        module = self.module
        img_meta = {
            'pad_shape': (x.shape[2], x.shape[3], 3),
            'border': (0, 0, 0, 0)
        }

        tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs = module(feat)
        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
            tl_heat=tl_heats[-1].sigmoid(),
            br_heat=br_heats[-1].sigmoid(),
            tl_off=tl_offs[-1],
            br_off=br_offs[-1],
            tl_emb=tl_embs[-1],
            br_emb=br_embs[-1],
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

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).repeat(1, 1, dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _local_maximum(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _topk(self, scores, k=20):
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def decode_heatmap(self,
                       tl_heat,
                       br_heat,
                       tl_off,
                       br_off,
                       tl_emb=None,
                       br_emb=None,
                       tl_centripetal_shift=None,
                       br_centripetal_shift=None,
                       img_meta=None,
                       k=100,
                       kernel=3,
                       distance_threshold=0.5,
                       num_dets=1000):
        with_embedding = tl_emb is not None and br_emb is not None
        with_centripetal_shift = (
            tl_centripetal_shift is not None
            and br_centripetal_shift is not None)
        assert with_embedding + with_centripetal_shift == 1
        batch, _, height, width = tl_heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        tl_heat = self._local_maximum(tl_heat, kernel=kernel)
        br_heat = self._local_maximum(br_heat, kernel=kernel)

        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = self._topk(tl_heat, k=k)
        br_scores, br_inds, br_clses, br_ys, br_xs = self._topk(br_heat, k=k)

        # We use repeat instead of expand here because expand is a
        # shallow-copy function. Thus it could cause unexpected testing result
        # sometimes. Using expand will decrease about 10% mAP during testing
        # compared to repeat.
        tl_ys = tl_ys.view(batch, k, 1).repeat(1, 1, k)
        tl_xs = tl_xs.view(batch, k, 1).repeat(1, 1, k)
        br_ys = br_ys.view(batch, 1, k).repeat(1, k, 1)
        br_xs = br_xs.view(batch, 1, k).repeat(1, k, 1)

        tl_off = self._transpose_and_gather_feat(tl_off, tl_inds)
        tl_off = tl_off.view(batch, k, 1, 2)
        br_off = self._transpose_and_gather_feat(br_off, br_inds)
        br_off = br_off.view(batch, 1, k, 2)

        tl_xs = tl_xs + tl_off[..., 0]
        tl_ys = tl_ys + tl_off[..., 1]
        br_xs = br_xs + br_off[..., 0]
        br_ys = br_ys + br_off[..., 1]

        if with_centripetal_shift:
            tl_centripetal_shift = self._transpose_and_gather_feat(
                tl_centripetal_shift, tl_inds).view(batch, k, 1, 2).exp()
            br_centripetal_shift = self._transpose_and_gather_feat(
                br_centripetal_shift, br_inds).view(batch, 1, k, 2).exp()

            tl_ctxs = tl_xs + tl_centripetal_shift[..., 0]
            tl_ctys = tl_ys + tl_centripetal_shift[..., 1]
            br_ctxs = br_xs - br_centripetal_shift[..., 0]
            br_ctys = br_ys - br_centripetal_shift[..., 1]

        # all possible boxes based on top k corners (ignoring class)
        zero_tensor = torch.zeros([1], dtype=torch.int32)
        w_ratio = ((inp_w + zero_tensor).to(tl_heat.device)).float() / (
            (width + zero_tensor).to(tl_heat.device)).float()
        h_ratio = ((inp_h + zero_tensor).to(tl_heat.device)).float() / (
            (height + zero_tensor).to(tl_heat.device)).float()
        tl_xs *= w_ratio
        tl_ys *= h_ratio
        br_xs *= w_ratio
        br_ys *= h_ratio

        if with_centripetal_shift:
            tl_ctxs *= w_ratio
            tl_ctys *= h_ratio
            br_ctxs *= w_ratio
            br_ctys *= h_ratio

        x_off = img_meta['border'][2]
        y_off = img_meta['border'][0]

        tl_xs -= x_off
        tl_ys -= y_off
        br_xs -= x_off
        br_ys -= y_off

        tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
        tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
        br_xs *= br_xs.gt(0.0).type_as(br_xs)
        br_ys *= br_ys.gt(0.0).type_as(br_ys)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        area_bboxes = ((br_xs - tl_xs) * (br_ys - tl_ys)).abs()

        if with_centripetal_shift:
            tl_ctxs -= x_off
            tl_ctys -= y_off
            br_ctxs -= x_off
            br_ctys -= y_off

            tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
            tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
            br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
            br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)

            ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys),
                                    dim=3)
            area_ct_bboxes = ((br_ctxs - tl_ctxs) * (br_ctys - tl_ctys)).abs()

            rcentral = torch.zeros_like(ct_bboxes)
            # magic nums from paper section 4.1
            mu = torch.ones_like(area_bboxes) / 2.4
            mu_mask = (area_bboxes > 3500).int().type_as(mu)
            # mu[area_bboxes > 3500] = 1 / 2.1  # large bbox have smaller mu
            mu = (1 / 2.1) * mu_mask + mu * (1. - mu_mask)

            bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
            rcentral0 = bboxes_center_x - mu * (bboxes[..., 2] -
                                                bboxes[..., 0]) / 2
            rcentral1 = bboxes_center_y - mu * (bboxes[..., 3] -
                                                bboxes[..., 1]) / 2
            rcentral2 = bboxes_center_x + mu * (bboxes[..., 2] -
                                                bboxes[..., 0]) / 2
            rcentral3 = bboxes_center_y + mu * (bboxes[..., 3] -
                                                bboxes[..., 1]) / 2
            rcentral = torch.stack(
                [rcentral0, rcentral1, rcentral2, rcentral3], dim=-1)
            area_rcentral = ((rcentral2 - rcentral0) *
                             (rcentral3 - rcentral1)).abs()
            dists = area_ct_bboxes / area_rcentral

            tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (
                ct_bboxes[..., 0] >= rcentral[..., 2])
            tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (
                ct_bboxes[..., 1] >= rcentral[..., 3])
            br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (
                ct_bboxes[..., 2] >= rcentral[..., 2])
            br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (
                ct_bboxes[..., 3] >= rcentral[..., 3])

        if with_embedding:
            tl_emb = self._transpose_and_gather_feat(tl_emb, tl_inds)
            tl_emb = tl_emb.view(batch, k, 1)
            br_emb = self._transpose_and_gather_feat(br_emb, br_inds)
            br_emb = br_emb.view(batch, 1, k)
            dists = torch.abs(tl_emb - br_emb)

        tl_scores = tl_scores.view(batch, k, 1).repeat(1, 1, k)
        br_scores = br_scores.view(batch, 1, k).repeat(1, k, 1)

        scores = (tl_scores + br_scores) / 2  # scores for all possible boxes

        # tl and br should have same class
        tl_clses = tl_clses.view(batch, k, 1).repeat(1, 1, k)
        br_clses = br_clses.view(batch, 1, k).repeat(1, k, 1)
        cls_inds = (tl_clses != br_clses)

        # reject boxes based on distances
        dist_inds = dists > distance_threshold

        # reject boxes based on widths and heights
        width_inds = (br_xs <= tl_xs)
        height_inds = (br_ys <= tl_ys)

        score_neg_mask = cls_inds | width_inds | height_inds | dist_inds
        if with_centripetal_shift:
            score_neg_mask = score_neg_mask | tl_ctx_inds | tl_cty_inds \
                | br_ctx_inds | br_cty_inds

        score_neg_mask = score_neg_mask.int().type_as(scores)
        scores = (1 - score_neg_mask) * scores + score_neg_mask * (-1)

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = self._gather_feat(bboxes, inds)

        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = self._gather_feat(clses, inds).float()

        return bboxes, scores, clses
