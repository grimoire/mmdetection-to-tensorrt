import numpy as np
import torch
from mmdet2trt.models.builder import register_wraper
from torch import nn
from torch.nn import functional as F

from .transforms import bbox_rescale_batched


def bucket2bbox_batched(proposals,
                        cls_preds,
                        offset_preds,
                        num_buckets,
                        scale_factor=1.0,
                        max_shape=None):
    batch_size = cls_preds.size(0)
    side_num = int(np.ceil(num_buckets / 2.0))
    cls_preds = cls_preds.view(batch_size, -1, side_num)
    offset_preds = offset_preds.view(batch_size, -1, side_num)

    scores = F.softmax(cls_preds, dim=2)
    score_topk, score_label = scores.topk(2, dim=2, largest=True, sorted=True)

    rescaled_proposals = bbox_rescale_batched(proposals, scale_factor)

    pw = rescaled_proposals[..., 2] - rescaled_proposals[..., 0]
    ph = rescaled_proposals[..., 3] - rescaled_proposals[..., 1]
    px1 = rescaled_proposals[..., 0]
    py1 = rescaled_proposals[..., 1]
    px2 = rescaled_proposals[..., 2]
    py2 = rescaled_proposals[..., 3]

    bucket_w = pw / num_buckets
    bucket_h = ph / num_buckets

    score_inds_l = score_label[:, 0::4, 0]
    score_inds_r = score_label[:, 1::4, 0]
    score_inds_t = score_label[:, 2::4, 0]
    score_inds_d = score_label[:, 3::4, 0]

    l_buckets = px1 + (0.5 + score_inds_l.float()) * bucket_w
    r_buckets = px2 - (0.5 + score_inds_r.float()) * bucket_w
    t_buckets = py1 + (0.5 + score_inds_t.float()) * bucket_h
    d_buckets = py2 - (0.5 + score_inds_d.float()) * bucket_h

    offsets = offset_preds.view(batch_size, -1, 4, side_num)

    l_offsets = offsets[:, :,
                        0, :].gather(2, score_inds_l.unsqueeze(2)).squeeze(2)
    r_offsets = offsets[:, :,
                        1, :].gather(2, score_inds_r.unsqueeze(2)).squeeze(2)
    t_offsets = offsets[:, :,
                        2, :].gather(2, score_inds_t.unsqueeze(2)).squeeze(2)
    d_offsets = offsets[:, :,
                        3, :].gather(2, score_inds_d.unsqueeze(2)).squeeze(2)

    x1 = l_buckets - l_offsets * bucket_w
    x2 = r_buckets - r_offsets * bucket_w
    y1 = t_buckets - t_offsets * bucket_h
    y2 = d_buckets - d_offsets * bucket_h

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.cat(
        [x1[:, :, None], y1[:, :, None], x2[:, :, None], y2[:, :, None]],
        dim=-1)

    loc_confidence = score_topk[:, :, 0]
    top2_neighbor_inds = (score_label[:, :, 0] -
                          score_label[:, :, 1]).float().abs() - 1
    loc_confidence += score_topk[:, :, 1] * top2_neighbor_inds.float()
    loc_confidence = loc_confidence.view(batch_size, -1, 4).mean(dim=2)

    return bboxes, loc_confidence


@register_wraper('mmdet.core.bbox.coder.BucketingBBoxCoder')
class BucketingBBoxCoderWraper(nn.Module):

    def __init__(self, module):
        super(BucketingBBoxCoderWraper, self).__init__()
        self.module = module

    def forward(self,
                cls_scores,
                bbox_preds,
                anchors,
                min_num_bboxes,
                num_classes,
                use_sigmoid_cls,
                input_x=None):
        pass

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        module = self.module
        cls_preds, offset_preds = pred_bboxes
        decoded_bboxes = bucket2bbox_batched(bboxes, cls_preds, offset_preds,
                                             module.num_buckets,
                                             module.scale_factor, max_shape)
        return decoded_bboxes
