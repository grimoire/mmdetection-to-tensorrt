import mmdet2trt.ops.util_ops as mm2trt_util
import torch
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper('mmdet.models.dense_heads.YOLOV3Head')
class YOLOV3HeadWraper(nn.Module):

    def __init__(self, module):
        super(YOLOV3HeadWraper, self).__init__()
        self.module = module
        self.anchor_generator = build_wraper(self.module.anchor_generator)
        self.bbox_coder = build_wraper(self.module.bbox_coder)
        self.featmap_strides = module.featmap_strides
        self.num_attrib = module.num_attrib
        self.num_levels = module.num_levels
        iou_thr = 0.7
        if 'iou_thr' in module.test_cfg.nms:
            iou_thr = module.test_cfg.nms.iou_thr
        elif 'iou_threshold' in module.test_cfg.nms:
            iou_thr = module.test_cfg.nms.iou_threshold

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.rcnn_nms = BatchedNMS(
            module.test_cfg.score_thr, iou_thr, backgroundLabelId=-1)

    def forward(self, feats, x):

        module = self.module
        cfg = self.test_cfg

        pred_maps_list = module(feats)[0]

        multi_lvl_anchors = self.anchor_generator(
            pred_maps_list, device=pred_maps_list[0].device)

        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]
            batch_size = pred_map.shape[0]
            pred_map = pred_map.permute(0, 2, 3,
                                        1).reshape(batch_size, -1,
                                                   self.num_attrib)
            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            pred_map_pre_proposal = torch.sigmoid(pred_map[..., :2])
            pred_map_post_proposal = pred_map[..., 2:4]
            pred_map_proposal = torch.cat(
                [pred_map_pre_proposal, pred_map_post_proposal], dim=-1)
            anchors = multi_lvl_anchors[i].unsqueeze(0).expand_as(
                pred_map_proposal)
            bbox_pred = self.bbox_coder.decode(anchors, pred_map_proposal,
                                               stride)

            conf_pred = torch.sigmoid(pred_map[..., 4]).view(batch_size, -1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                batch_size, -1, self.num_classes)  # Cls pred one-hot.

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre:
                conf_pred = mm2trt_util.pad_with_value(conf_pred, 1, nms_pre,
                                                       0.)
                cls_pred = mm2trt_util.pad_with_value(cls_pred, 1, nms_pre)
                bbox_pred = mm2trt_util.pad_with_value(bbox_pred, 1, nms_pre)
                _, topk_inds = conf_pred.topk(nms_pre, dim=1)
                conf_pred = mm2trt_util.gather_topk(conf_pred, 1, topk_inds)
                cls_pred = mm2trt_util.gather_topk(cls_pred, 1, topk_inds)
                bbox_pred = mm2trt_util.gather_topk(bbox_pred, 1, topk_inds)

            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).float()
            conf_pred = conf_pred * conf_inds

            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes, dim=1)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores, dim=1)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores, dim=1)

        multi_lvl_cls_scores = multi_lvl_cls_scores \
            * multi_lvl_conf_scores.unsqueeze(2)
        multi_lvl_bboxes = multi_lvl_bboxes.unsqueeze(2)
        num_bboxes = multi_lvl_bboxes.shape[1]
        num_detected, proposals, scores, cls_id = self.rcnn_nms(
            multi_lvl_cls_scores, multi_lvl_bboxes, num_bboxes,
            self.test_cfg.max_per_img)
        return num_detected, proposals, scores, cls_id
