from mmdet2trt.models.builder import register_wraper
from torch import nn


@register_wraper('mmdet.models.roi_heads.roi_extractors'
                 '.generic_roi_extractor.GenericRoIExtractor')
class GenericRoIExtractorWraper(nn.Module):

    def __init__(self, module):
        super(GenericRoIExtractorWraper, self).__init__()
        self.module = module
        self.featmap_strides = self.module.featmap_strides
        self.num_inputs = self.module.num_inputs

    def forward(self, feats, rois, roi_scale_factor=None):
        return self.module(feats, rois, roi_scale_factor)
