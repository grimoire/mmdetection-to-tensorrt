import numpy as np
import torch
from mmdet2trt.models.builder import register_wrapper
from torch import nn


def delta2bbox_custom_func(cls_scores,
                           bbox_preds,
                           anchors,
                           min_num_bboxes,
                           target_mean,
                           target_std,
                           max_shape=None):

    proposals = delta2bbox_batched(anchors, bbox_preds, target_mean,
                                   target_std, max_shape)

    scores = cls_scores
    if scores.shape[1] < min_num_bboxes:
        pad_size = min_num_bboxes - scores.shape[1]
        scores = torch.nn.functional.pad(
            scores, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)
        proposals = torch.nn.functional.pad(
            proposals, [0, 0, 0, pad_size, 0, 0], mode='constant', value=0)

    return scores, proposals


def delta2bbox_batched(rois,
                       deltas,
                       means=(0., 0., 0., 0.),
                       stds=(1., 1., 1., 1.),
                       max_shape=None,
                       wh_ratio_clip=16 / 1000,
                       clip_border=True,
                       add_ctr_clamp=False,
                       ctr_clamp=32):

    means = deltas.new_tensor(means).view(1, -1)
    stds = deltas.new_tensor(stds).view(1, -1)
    delta_shape = deltas.shape
    reshaped_deltas = deltas.view(delta_shape[:-1] + (-1, 4))
    denorm_deltas = reshaped_deltas * stds + means

    dxy = denorm_deltas[..., :2]
    dwh = denorm_deltas[..., 2:]

    xy1 = rois[..., None, :2]
    xy2 = rois[..., None, 2:]

    pxy = (xy1 + xy2) * 0.5
    pwh = xy2 - xy1
    dxy_wh = pwh * dxy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    # Use exp(network energy) to enlarge/shrink each roi
    half_gwh = pwh * dwh.exp() * 0.5
    # Use network energy to shift the center of each roi
    gxy = pxy + dxy_wh

    # Convert center-xy/width/height to top-left, bottom-right
    xy1 = gxy - half_gwh
    xy2 = gxy + half_gwh

    x1 = xy1[..., 0]
    y1 = xy1[..., 1]
    x2 = xy2[..., 0]
    y2 = xy2[..., 1]

    if clip_border and max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


@register_wrapper('mmdet.core.bbox.coder.DeltaXYWHBBoxCoder')
class DeltaXYWHBBoxCoderWraper(nn.Module):

    def __init__(self, module):
        super(DeltaXYWHBBoxCoderWraper, self).__init__()
        self.means = module.means
        self.stds = module.stds

    def forward(self,
                cls_scores,
                bbox_preds,
                anchors,
                min_num_bboxes,
                num_classes,
                use_sigmoid_cls,
                max_shape=None):
        target_mean = self.means
        target_std = self.stds

        batch_size = cls_scores.shape[0]
        if len(anchors.shape) == 2:
            anchors = anchors.unsqueeze(0)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(batch_size, -1, num_classes)
        if use_sigmoid_cls:
            cls_scores = cls_scores.sigmoid()
        else:
            cls_scores = cls_scores.softmax(dim=2)

        bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        scores, proposals = delta2bbox_custom_func(cls_scores, bbox_preds,
                                                   anchors, min_num_bboxes,
                                                   target_mean, target_std,
                                                   max_shape)
        proposals = proposals.view(batch_size, -1, 1, 4)
        return scores, proposals

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        return delta2bbox_batched(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape=max_shape,
            wh_ratio_clip=wh_ratio_clip)
