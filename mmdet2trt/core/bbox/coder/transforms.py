import torch



def bbox_rescale_batched(bboxes, scale_factor=1.0):
    
    if bboxes.size(2) == 5:
        bboxes_ = bboxes[:, :, 1:]
        inds_ = bboxes[:, :, 0]
    else:
        bboxes_ = bboxes
    cx = (bboxes_[:, :, 0] + bboxes_[:, :, 2]) * 0.5
    cy = (bboxes_[:, :, 1] + bboxes_[:,:, 3]) * 0.5
    w = bboxes_[:, :, 2] - bboxes_[:, :, 0]
    h = bboxes_[:, :, 3] - bboxes_[:, :, 1]
    w = w * scale_factor
    h = h * scale_factor
    x1 = cx - 0.5 * w
    x2 = cx + 0.5 * w
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    if bboxes.size(2) == 5:
        rescaled_bboxes = torch.stack([inds_, x1, y1, x2, y2], dim=-1)
    else:
        rescaled_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return rescaled_bboxes
