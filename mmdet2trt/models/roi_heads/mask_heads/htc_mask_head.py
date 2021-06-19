from mmdet2trt.models.builder import register_wraper
from torch import nn


@register_wraper('mmdet.models.roi_heads.mask_heads.htc_mask_head.HTCMaskHead')
class HTCMaskHeadWraper(nn.Module):

    def __init__(self, module, test_cfg):
        super(HTCMaskHeadWraper, self).__init__()

        self.module = module
        self.test_cfg = test_cfg

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        return self.module(
            x,
            res_feat=res_feat,
            return_logits=return_logits,
            return_feat=return_feat)
