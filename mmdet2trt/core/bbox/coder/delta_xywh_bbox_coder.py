from mmdet2trt.models.builder import register_warper, build_warper
import torch
from torch import nn
import numpy as np
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
        # rpn_cls_score = rpn_cls_score.reshape(-1)
        scores = rpn_cls_score.sigmoid()
    else:
        rpn_cls_score = rpn_cls_score.reshape(-1, num_classes)
        scores = rpn_cls_score.softmax(dim=1)

    rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
    
    # anchors = mlvl_anchors[idx]
    img_meta = None
    if input_x is not None:
        height, width = input_x.shape[2:]
        img_meta = (height, width)
    proposals = delta2bbox(anchors, rpn_bbox_pred, target_mean,
                            target_std, img_meta)

    scores = scores.contiguous().view(1,-1, num_classes)
    proposals = proposals.view(1, -1, 4)

    if scores.shape[1]<min_num_bboxes:
        pad_size = min_num_bboxes-scores.shape[1]
        scores = torch.nn.functional.pad(scores, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)
        proposals = torch.nn.functional.pad(proposals, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)
        
    proposals = proposals.view(1, -1, 1, 4)

    return scores, proposals


def delta2bbox_batched(rois,
                        deltas,
                        means=(0., 0., 0., 0.),
                        stds=(1., 1., 1., 1.),
                        max_shape=None,
                        wh_ratio_clip=16 / 1000):
                        
    means = deltas.new_tensor(means).repeat(1, deltas.size(2) // 4).unsqueeze(0)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(2) // 4).unsqueeze(0)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, :, 0::4]
    dy = denorm_deltas[:, :, 1::4]
    dw = denorm_deltas[:, :, 2::4]
    dh = denorm_deltas[:, :, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = ((rois[:, :, 0:1] + rois[:, :, 2:3]) * 0.5).expand_as(dx)
    py = ((rois[:, :, 1:2] + rois[:, :, 3:4]) * 0.5).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, :, 2:3] - rois[:, :, 0:1]).expand_as(dw)
    ph = (rois[:, :, 3:4] - rois[:, :, 1:2]).expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()

    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1]-1)
        y1 = y1.clamp(min=0, max=max_shape[0]-1)
        x2 = x2.clamp(min=0, max=max_shape[1]-1)
        y2 = y2.clamp(min=0, max=max_shape[0]-1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes



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

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        return delta2bbox_batched(bboxes, pred_bboxes, self.means, self.stds, max_shape=max_shape, wh_ratio_clip=wh_ratio_clip)