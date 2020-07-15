from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS

@register_warper("mmdet.models.roi_heads.standard_roi_head.StandardRoIHead")
class StandardRoIHeadWarper(nn.Module):
    def __init__(self, module):
        super(StandardRoIHeadWarper, self).__init__()
        self.module = module

        # self.bbox_roi_extractor = module.bbox_roi_extractor
        self.bbox_roi_extractor = build_warper(module.bbox_roi_extractor)
        
        self.bbox_head = module.bbox_head
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        self.test_cfg = module.test_cfg
        self.rcnn_nms = BatchedNMS(module.test_cfg.score_thr, module.test_cfg.nms.iou_threshold, backgroundLabelId = self.bbox_head.num_classes)

    def forward(self, feat ,proposals):
        zeros = proposals.new_zeros([proposals.shape[0], 1])
        rois = torch.cat([zeros, proposals], dim=1)

        roi_feats = self.bbox_roi_extractor(
            feat[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.shared_head is not None:
            roi_feats = self.shared_head(roi_feats)
        # rcnn
        cls_score, bbox_pred = self.bbox_head(roi_feats)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1)
        bboxes = delta2bbox(proposals, bbox_pred, self.bbox_head.bbox_coder.means,
                    self.bbox_head.bbox_coder.stds)

        num_bboxes = bboxes.shape[0]
        scores = scores.unsqueeze(0)
        bboxes = bboxes.view(1, num_bboxes, -1, 4)
        bboxes_ext = bboxes.new_zeros((1,num_bboxes, 1, 4))
        bboxes = torch.cat([bboxes, bboxes_ext], 2)
        num_detections, det_boxes, det_scores, det_classes = self.rcnn_nms(scores, bboxes, num_bboxes, self.test_cfg.max_per_img)

        return num_detections, det_boxes, det_scores, det_classes
