import mmdet2trt.ops.util_ops as mm2trt_util
import torch


def batched_distance2bbox(points, distance, max_shape=None):
    x1 = points[:, :, 0] - distance[:, :, 0]
    y1 = points[:, :, 1] - distance[:, :, 1]
    x2 = points[:, :, 0] + distance[:, :, 2]
    y2 = points[:, :, 1] + distance[:, :, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2roi(proposals):
    num_proposals = proposals.shape[1]
    rois_pad = mm2trt_util.arange_by_input(proposals, 0).unsqueeze(1)
    rois_pad = rois_pad.repeat(1, num_proposals).view(-1, 1)
    proposals = proposals.view(-1, 4)
    rois = torch.cat([rois_pad, proposals], dim=1)
    return rois


def batched_bbox_cxcywh_to_xyxy(bbox):
    cx = bbox[:, :, 0]
    cy = bbox[:, :, 1]
    w = bbox[:, :, 2]
    h = bbox[:, :, 3]
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]

    return torch.stack(bbox_new, dim=-1)
