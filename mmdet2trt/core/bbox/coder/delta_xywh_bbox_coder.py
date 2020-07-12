from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox


def delta2bbox_custom_func(cls_scores, bbox_preds, anchors,
                      min_num_bboxes, num_classes, use_sigmoid_cls,
                      target_mean, target_std, input_x=None):
                      
    rpn_cls_score = cls_scores.squeeze(0)
    rpn_bbox_pred = bbox_preds.squeeze(0)
    anchors = anchors.squeeze(0)
    # rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
    rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
    if use_sigmoid_cls:
        rpn_cls_score = rpn_cls_score.reshape(rpn_cls_score.shape[0], -1)
        # rpn_cls_score = rpn_cls_score.reshape(-1)
        scores = rpn_cls_score.sigmoid()
    else:
        rpn_cls_score = rpn_cls_score.reshape(-1, 2)
        scores = rpn_cls_score.softmax(dim=1)

    rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
    
    # anchors = mlvl_anchors[idx]
    img_meta = None
    if input_x is not None:
        height, width = input_x.shape[2:]
        img_meta = (height, width)
    proposals = delta2bbox(anchors, rpn_bbox_pred, target_mean,
                            target_std, img_meta)

    scores = scores.view(1,-1, 1)
    proposals = proposals.view(1, -1, 4)

    if scores.shape[1]<min_num_bboxes:
        pad_size = min_num_bboxes-scores.shape[1]
        scores = torch.nn.functional.pad(scores, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)
        proposals = torch.nn.functional.pad(proposals, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)
        
    proposals = proposals.view(1, -1, 1, 4)

    return scores, proposals


@register_warper("mmdet.core.bbox.coder.DeltaXYWHBBoxCoder")
class DeltaXYWHBBoxCoderWarper(nn.Module):
    def __init__(self, module):
        super(DeltaXYWHBBoxCoderWarper, self).__init__()
        self.means =  module.means
        self.stds = module.stds


    def forward(self, cls_scores, bbox_preds, anchors,
                min_num_bboxes, num_classes, use_sigmoid_cls, input_x=None):
        target_mean = self.means
        target_std = self.stds
        
        return delta2bbox_custom_func(cls_scores, bbox_preds, anchors,
                      min_num_bboxes, num_classes, use_sigmoid_cls,
                      target_mean, target_std, input_x)