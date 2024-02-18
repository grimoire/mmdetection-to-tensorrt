import numpy as np
import torch
from mmdet2trt.models.builder import register_wrapper
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

    pxy1 = rescaled_proposals[..., :2]
    pxy2 = rescaled_proposals[..., 2:]

    pwh = pxy2 - pxy1
    bucket_wh = pwh / num_buckets

    score_label_reshaped = score_label[..., 0].reshape((batch_size, -1, 4))
    score_inds_lt = score_label_reshaped[:, :, ::2]
    score_inds_rd = score_label_reshaped[:, :, 1::2]

    lt_buckets = pxy1 + (0.5 + score_inds_lt.float()) * bucket_wh
    rd_buckets = pxy2 - (0.5 + score_inds_rd.float()) * bucket_wh

    offsets = offset_preds.view(batch_size, -1, 4, side_num)

    lt_offsets = offsets[:, :, ::2, :].gather(
        3, score_inds_lt.unsqueeze(-1)).squeeze(-1)
    rd_offsets = offsets[:, :, 1::2, :].gather(
        3, score_inds_rd.unsqueeze(-1)).squeeze(-1)

    xy1 = lt_buckets - lt_offsets * bucket_wh
    xy2 = rd_buckets - rd_offsets * bucket_wh

    x1 = xy1[..., 0]
    y1 = xy1[..., 1]
    x2 = xy2[..., 0]
    y2 = xy2[..., 1]

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


@register_wrapper('mmdet.models.task_modules.coders.BucketingBBoxCoder')
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
