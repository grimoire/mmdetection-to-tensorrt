import torch


def bbox_rescale_batched(bboxes, scale_factor=1.0):

    if bboxes.size(2) == 5:
        bboxes_ = bboxes[:, :, 1:]
        inds_ = bboxes[:, :, :1]
    else:
        bboxes_ = bboxes

    p1 = bboxes_[:, :, :2]
    p2 = bboxes_[:, :, 2:]
    cxy = (p1 + p2) * 0.5
    half_wh = (p2 - p1) * scale_factor * 0.5

    new_p1 = cxy - half_wh
    new_p2 = cxy + half_wh

    if bboxes.size(2) == 5:
        rescaled_bboxes = torch.cat([inds_, new_p1, new_p2], dim=-1)
    else:
        rescaled_bboxes = torch.cat([new_p1, new_p2], dim=-1)
    return rescaled_bboxes
