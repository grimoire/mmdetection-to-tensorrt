import torch
import torch.nn.functional as F
from mmdet2trt.models.builder import register_wraper

from .bbox_head import BBoxHeadWraper


@register_wraper('mmdet.models.roi_heads.bbox_heads.sabl_head.SABLHead')
class SABLHeadWraper(BBoxHeadWraper):

    def __init__(self, module, test_cfg):
        super(SABLHeadWraper, self).__init__(module, test_cfg)

    def regress_by_class(self, rois, label, bbox_pred, img_shape):
        if rois.size(1) == 4:
            new_rois, _ = self.bbox_coder.decode(
                rois.unsqueeze(0), [bb.unsqueeze(0) for bb in bbox_pred],
                max_shape=img_shape)
        else:
            bboxes, _ = self.bbox_coder.decode(
                rois[:, 1:].unsqueeze(0),
                [bb.unsqueeze(0) for bb in bbox_pred],
                max_shape=img_shape)
            new_rois = torch.cat((rois[:, 0:1], bboxes), dim=2)

        new_rois = new_rois.squeeze(0)
        return new_rois

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, batch_size,
                   num_proposals, cfg):
        module = self.module
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=-1)

        if rois.size(1) == 4:
            bboxes, confids = self.bbox_coder.decode(
                rois.unsqueeze(0), [bb.unsqueeze(0) for bb in bbox_pred],
                max_shape=img_shape)
        else:
            bboxes, confids = self.bbox_coder.decode(
                rois[:, 1:].unsqueeze(0),
                [bb.unsqueeze(0) for bb in bbox_pred],
                max_shape=img_shape)
        bboxes = bboxes.squeeze(0)
        confids = confids.squeeze(0)
        scores = scores * confids.unsqueeze(1)

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
