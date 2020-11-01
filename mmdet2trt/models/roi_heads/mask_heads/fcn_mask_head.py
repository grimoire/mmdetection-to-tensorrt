from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn
import torch.nn.functional as F
from mmdet2trt.core.post_processing.batched_nms import BatchedNMS


@register_wraper("mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead")
class FCNMaskHeadWraper(nn.Module):
    def __init__(self, module, test_cfg):
        super(FCNMaskHeadWraper, self).__init__()

        self.module = module
        self.test_cfg = test_cfg

    def forward(self, x):
        return self.module(x)
