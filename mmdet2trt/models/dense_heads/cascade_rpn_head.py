import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn

from .rpn_head import RPNHeadWraper


@register_wraper('mmdet.models.dense_heads.StageCascadeRPNHead')
class StageCascadeRPNHeadWraper(RPNHeadWraper):

    def __init__(self, module):
        super(StageCascadeRPNHeadWraper, self).__init__(module)

        ks = 3
        pad = (ks - 1) // 2
        idx = torch.linspace(-pad, pad, 2 * pad + 1)
        yy, xx = torch.meshgrid(idx, idx)  # return order matters
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        self.shape_offset_xx = xx
        self.shape_offset_yy = yy

    def forward(self, x, offset_list):
        return self.module(x, offset_list)

    def get_anchors(self, featmaps, device='cuda'):
        return self.anchor_generator(featmaps, device=device)

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):

        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            xx = self.shape_offset_xx.type_as(anchor_list[0]).to(device)
            yy = self.shape_offset_yy.type_as(anchor_list[0]).to(device)
            w = (anchors[:, :, 2] - anchors[:, :, 0]) / stride
            h = (anchors[:, :, 3] - anchors[:, :, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, :, None] * xx  # (NA, ks**2)
            offset_y = h[:, :, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert anchors.size(1) == feat_h * feat_w

            x = (anchors[:, :, 0] + anchors[:, :, 2]) * 0.5
            y = (anchors[:, :, 1] + anchors[:, :, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.linspace(0, feat_w - 1, feat_w, device=anchors.device)
            yy = torch.linspace(0, feat_h - 1, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_lvls = len(anchor_list)
        device = anchor_list[0].device

        mlvl_offset = []
        for lvl in range(num_lvls):
            c_offset_x, c_offset_y = _ctr_offset(anchor_list[lvl],
                                                 anchor_strides[lvl],
                                                 featmap_sizes[lvl])
            s_offset_x, s_offset_y = _shape_offset(anchor_list[lvl],
                                                   anchor_strides[lvl])
            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, :, None]
            offset_y = s_offset_y + c_offset_y[:, :, None]

            # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
            offset = torch.cat([offset_y, offset_x], dim=-1)
            # offset = offset.reshape(offset.size(0), offset.size(1),
            #                         -1)  # [NA, 2*ks**2]
            mlvl_offset.append(offset)
        return mlvl_offset

    def refine_bboxes(self, anchor_list, bbox_preds, img_meta):
        img_shape = img_meta['img_shape']
        batch_size = bbox_preds[0].size(0)
        num_levels = len(bbox_preds)
        mlvl_anchors = []
        for i in range(num_levels):
            bbox_pred = bbox_preds[i]
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            bboxes = self.bbox_coder.decode(
                anchor_list[i].unsqueeze(0), bbox_pred, max_shape=img_shape)
            mlvl_anchors.append(bboxes)
        return mlvl_anchors

    def get_bboxes(self, mlvl_anchors, cls_scores, bbox_preds, img_metas,
                   test_cfg):
        x = img_metas['x']
        nms_pre = test_cfg.nms_pre if self.test_cfg.nms_pre > 0 else 1000
        nms_post = test_cfg.nms_post
        use_sigmoid_cls = self.module.use_sigmoid_cls

        mlvl_scores = []
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]

            scores, proposals = self.bbox_coder(
                rpn_cls_score,
                rpn_bbox_pred,
                anchors.squeeze(0),
                min_num_bboxes=nms_pre,
                num_classes=1,
                use_sigmoid_cls=use_sigmoid_cls,
                input_x=x)
            if nms_pre > 0:
                _, topk_inds = scores.squeeze(2).topk(nms_pre, dim=1)
                proposals = mm2trt_util.gather_topk(proposals, 1, topk_inds)
                scores = mm2trt_util.gather_topk(scores, 1, topk_inds)

            mlvl_scores.append(scores)
            mlvl_proposals.append(proposals)

        scores = torch.cat(mlvl_scores, dim=1)
        proposals = torch.cat(mlvl_proposals, dim=1)

        _, proposals, scores, _ = self.rpn_nms(scores, proposals,
                                               scores.size(1), nms_post)

        return proposals


@register_wraper('mmdet.models.dense_heads.CascadeRPNHead')
class CascadeRPNHeadWraper(nn.Module):

    def __init__(self, module):
        super(CascadeRPNHeadWraper, self).__init__()
        self.module = module
        self.stages = [build_wraper(stage) for stage in self.module.stages]
        self.test_cfg = module.test_cfg

    def forward(self, feat, x):
        featmap_sizes = [featmap.size()[-2:] for featmap in feat]
        anchor_list = self.stages[0].get_anchors(feat, feat[0].device)

        for i in range(self.module.num_stages):
            stage = self.stages[i]
            if stage.module.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.module.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            feat, cls_score, bbox_pred = stage(feat, offset_list)

            if i < self.module.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  {'img_shape': x.shape[2:]})

        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                   bbox_pred, {'x': x},
                                                   self.test_cfg)
        return proposal_list
