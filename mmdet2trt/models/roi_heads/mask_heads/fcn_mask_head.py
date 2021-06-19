import torch
import torch.nn.functional as F
from mmdet2trt.models.builder import register_wraper
from torch import nn


@register_wraper('mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead')
class FCNMaskHeadWraper(nn.Module):

    def __init__(self, module, test_cfg):
        super(FCNMaskHeadWraper, self).__init__()

        self.module = module
        self.test_cfg = test_cfg

    def forward(self, x):
        return self.module(x)


class MaskProcessor(nn.Module):

    def __init__(self, max_width, max_height):
        super(MaskProcessor, self).__init__()
        self.max_height = max_height
        self.max_width = max_width

        img_y = torch.arange(0, max_height, dtype=torch.float32) + 0.5
        img_x = torch.arange(0, max_width, dtype=torch.float32) + 0.5

        self.img_y = img_y
        self.img_x = img_x

    def forward(self, masks, boxes):

        N, C, H, W = masks.shape
        masks = masks.view(-1, 1, H, W)
        boxes = boxes.view(-1, 4)

        x0 = boxes[:, 0:1]
        y0 = boxes[:, 1:2]
        x1 = boxes[:, 2:3] + 1e-5
        y1 = boxes[:, 3:4] + 1e-5

        img_y = self.img_y.to(masks.device)
        img_x = self.img_x.to(masks.device)

        img_y = (img_y - y0) / (y1 - y0) * 2 - 1
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1

        expand_shape = img_y.new_zeros([N * C, img_y.size(1), img_x.size(1)])

        gx = img_x[:, None, :].expand_as(expand_shape)
        gy = img_y[:, :, None].expand_as(expand_shape)
        grid = torch.stack([gx, gy], dim=3)

        img_masks = F.grid_sample(
            masks.to(dtype=torch.float32), grid, align_corners=False)

        mask_h = img_masks.shape[2]
        mask_w = img_masks.shape[3]
        img_masks = img_masks.view(N, C, mask_h, mask_w)

        return img_masks
