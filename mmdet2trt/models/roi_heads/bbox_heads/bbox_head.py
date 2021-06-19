import torch
import torch.nn.functional as F
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn


@register_wraper(
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.ConvFCBBoxHead')
@register_wraper(
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared2FCBBoxHead')
@register_wraper(
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.Shared4Conv1FCBBoxHead'
)
class BBoxHeadWraper(nn.Module):

    def __init__(self, module, test_cfg):
        super(BBoxHeadWraper, self).__init__()

        self.module = module
        self.bbox_coder = build_wraper(self.module.bbox_coder)
        self.test_cfg = test_cfg
        self.num_classes = module.num_classes
        self.rcnn_nms = BatchedNMS(
            test_cfg.score_thr,
            test_cfg.nms.iou_threshold,
            backgroundLabelId=module.num_classes)

    def forward(self, x):
        return self.module(x)

    def regress_by_class(self, rois, label, bbox_pred, img_shape):
        module = self.module
        reg_class_agnostic = module.reg_class_agnostic

        if not reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois.unsqueeze(0), bbox_pred.unsqueeze(0), max_shape=img_shape)
        else:
            new_rois = self.bbox_coder.decode(
                rois[:, 1:].unsqueeze(0),
                bbox_pred.unsqueeze(0),
                max_shape=img_shape)

        new_rois = new_rois.squeeze(0)
        return new_rois

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, batch_size,
                   num_proposals, cfg):
        module = self.module
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=-1)

        if bbox_pred is not None:
            if rois.size(1) == 4:
                bboxes = self.bbox_coder.decode(
                    rois.unsqueeze(0),
                    bbox_pred.unsqueeze(0),
                    max_shape=img_shape)
            else:
                bboxes = self.bbox_coder.decode(
                    rois[:, 1:].unsqueeze(0),
                    bbox_pred.unsqueeze(0),
                    max_shape=img_shape)
            bboxes = bboxes.squeeze(0)
        else:
            if rois.size(1) == 4:
                bboxes = rois
            else:
                bboxes = rois[:, 1:]

        scores = scores.view(batch_size, num_proposals, -1)
        bboxes = bboxes.view(batch_size, num_proposals, -1, 4)
        num_bboxes = bboxes.shape[1]
        if bboxes.size(2) == module.num_classes:
            bboxes_ext = bboxes[:, :, 0:1, :] * 0
            bboxes = torch.cat([bboxes, bboxes_ext], 2)
        else:
            bboxes = bboxes.repeat(1, 1, module.num_classes + 1, 1)
        num_detections, det_boxes, det_scores, det_classes = self.rcnn_nms(
            scores, bboxes, num_bboxes, cfg.max_per_img)

        return num_detections, det_boxes, det_scores, det_classes
