import torch


def bbox_overlaps_batched(bboxes1,
                          bboxes2,
                          mode='iou',
                          is_aligned=False,
                          eps=1e-6):

    if is_aligned:
        lt = torch.max(bboxes1[:, :, :2], bboxes2[:, :, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, :, 2:], bboxes2[:, :, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (
            bboxes1[:, :, 3] - bboxes1[:, :, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (
                bboxes2[:, :, 3] - bboxes2[:, :, 1])
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        bboxes1_permute = bboxes1.permute(1, 0, 2)
        bboxes2_permute = bboxes2.permute(1, 0, 2)
        lt = torch.max(bboxes1_permute[:, None, :, :2],
                       bboxes2_permute[:, :, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1_permute[:, None, :, 2:],
                       bboxes2_permute[:, :, 2:])  # [rows, cols, 2]
        lt = lt.permute(2, 0, 1, 3)
        rb = rb.permute(2, 0, 1, 3)

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, :, 0] * wh[:, :, :, 1]
        area1 = (bboxes1[:, :, 2] - bboxes1[:, :, 0]) * (
            bboxes1[:, :, 3] - bboxes1[:, :, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, :, 2] - bboxes2[:, :, 0]) * (
                bboxes2[:, :, 3] - bboxes2[:, :, 1])
            union = area1.unsqueeze(-1) + area2.unsqueeze(1) - overlap
        else:
            union = area1.unsqueeze(-1)

    if not isinstance(eps, torch.Tensor):
        eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious
