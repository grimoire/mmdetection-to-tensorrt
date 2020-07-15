import torch
from torch import nn

class RoiAlignExtractor(nn.Module):
    def __init__(self, module):
        super(RoiAlignExtractor, self).__init__()
        self.module = module

        self.roi_layers = self.module.roi_layers
        self.featmap_strides = self.module.featmap_strides
        self.finest_scale = self.module.finest_scale

    def forward(self, feats, rois, roi_scale_factor=None):
        return self.module(feats, rois, roi_scale_factor)