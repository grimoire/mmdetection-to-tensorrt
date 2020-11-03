import torch


def merge_aug_masks(aug_masks, rcnn_test_cfg, weights=None):
    if weights is None:
        merged_masks = torch.mean(torch.stack(aug_masks, 0), 0)
    else:
        merged_masks = torch.mean(torch.stack(aug_masks, 0), 0)

    return merged_masks
