from mmdet2trt.models.builder import register_warper, build_warper
from mmdet2trt.ops import util_ops
import torch
from torch import nn

def batched_blr2bboxes(priors,
                        tblr,
                        normalizer=4.0,
                        normalize_by_wh=True,
                        max_shape=None):
    if not isinstance(normalizer, float):
        normalizer = tblr.new_tensor(normalizer)
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, :, 0:2] + priors[:, :, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, :, 2:4] - priors[:, :, 0:2]
        w = wh[:, :, 0:1]
        h = wh[:, :, 1:2]
        loc_decode_h = loc_decode[:, :, :2]*h
        loc_decode_w = loc_decode[:, :, 2:]*w
        loc_decode = torch.cat([loc_decode_h, loc_decode_w], dim=2)
    top, bottom, left, right = loc_decode.chunk(4, dim=2)
    xmin = prior_centers[:, :, 0:1] - left
    xmax = prior_centers[:, :, 0:1] + right
    ymin = prior_centers[:, :, 1:2] - top
    ymax = prior_centers[:, :, 1:2] + bottom
    if max_shape is not None:
        xmin = xmin.clamp(min=0, max=max_shape[1])
        ymin = ymin.clamp(min=0, max=max_shape[0])
        xmax = xmax.clamp(min=0, max=max_shape[1])
        ymax = ymax.clamp(min=0, max=max_shape[0])
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=2)
    return boxes


@register_warper("mmdet.core.bbox.coder.TBLRBBoxCoder")
class TBLRBBoxCoderWarper(nn.Module):
    def __init__(self, module):
        super(TBLRBBoxCoderWarper, self).__init__()
        self.normalizer = module.normalizer

    def forward(self, cls_scores, bbox_preds, anchors,
                min_num_bboxes, num_classes, use_sigmoid_cls, input_x=None):
        cls_scores = cls_scores.permute(0, 2, 3, 1).reshape(cls_scores.shape[0], -1, num_classes)
        if use_sigmoid_cls:
            scores = cls_scores.sigmoid()
        else:
            cls_scores = cls_scores
            scores = cls_scores.softmax(dim=2)

        bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(bbox_preds.shape[0], -1, 4)
        anchors = anchors.unsqueeze(0)

        proposals = batched_blr2bboxes(anchors, bbox_preds, normalizer=self.normalizer, max_shape=input_x.shape[2:])
        if min_num_bboxes>0:
            scores = util_ops.pad_with_value(scores, 1, min_num_bboxes, 0)
            proposals = util_ops.pad_with_value(proposals, 1, min_num_bboxes)

        proposals = proposals.unsqueeze(2)
        return scores, proposals
        

        