from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS

@register_warper("mmdet.models.roi_heads.CascadeRoIHead")
class CascadeRoIHeadWarper(nn.Module):
    def __init__(self, module):
        super(CascadeRoIHeadWarper, self).__init__()
        self.module = module

        self.bbox_roi_extractor = module.bbox_roi_extractor
        self.bbox_head = module.bbox_head
        if module.with_shared_head:
            self.shared_head = module.shared_head
        else:
            self.shared_head = None

        self.test_cfg = module.test_cfg

        self.num_stages = module.num_stages
        self.rcnn_nms = BatchedNMS(module.test_cfg.score_thr, module.test_cfg.nms.iou_threshold, backgroundLabelId = self.bbox_head[-1].num_classes)


    def _bbox_forward(self, stage, x, rois):

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        if rois.shape[1] == 4:
            zeros = rois.new_zeros([rois.shape[0], 1])
            rois = torch.cat([zeros, rois], dim=1)
        
        roi_feats = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = bbox_head(roi_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=roi_feats)
        return bbox_results
        

    def regress_by_class(self, stage, rois, label, bbox_pred):
        bbox_head = self.bbox_head[stage]
        reg_class_agnostic = bbox_head.reg_class_agnostic

        if not reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        
        means = bbox_head.bbox_coder.means
        stds = bbox_head.bbox_coder.stds

        new_rois = delta2bbox(rois, bbox_pred, means, stds)
        return new_rois



    def forward(self, feat ,proposals): 
        ms_scores = []
        rois = proposals
        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, feat, rois)
            ms_scores.append(bbox_results['cls_score'])
            bbox_pred = bbox_results['bbox_pred']

            if i < self.num_stages - 1:
                bbox_label = bbox_results['cls_score'].argmax(dim=1)
                rois = self.regress_by_class(i, rois, bbox_label, bbox_pred)

        ### bbox_head.get_boxes
        cls_score = sum(ms_scores) / self.num_stages

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1)
        bbox_head = self.bbox_head[-1]
        bboxes = delta2bbox(rois, bbox_pred, bbox_head.bbox_coder.means,
                    bbox_head.bbox_coder.stds)
        
        num_bboxes = bboxes.shape[0]
        scores = scores.unsqueeze(0)
        bboxes = bboxes.view(1, num_bboxes, -1, 4)
        bboxes = bboxes.repeat(1, 1, bbox_head.num_classes, 1)
        bboxes_ext = bboxes.new_zeros((1,num_bboxes, 1, 4))
        bboxes = torch.cat([bboxes, bboxes_ext], 2)
        num_detections, det_boxes, det_scores, det_classes = self.rcnn_nms(scores, bboxes, num_bboxes, self.test_cfg.max_per_img)

        return num_detections, det_boxes, det_scores, det_classes
