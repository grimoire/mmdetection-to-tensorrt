from mmdet2trt.models.builder import register_wraper
from mmdet2trt.models.roi_heads.roi_extractors.pooling_layers import \
    build_roi_extractor
from torch import nn


@register_wraper('mmdet.models.roi_heads.roi_extractors'
                 '.single_level_roi_extractor.SingleRoIExtractor')
class SingleRoIExtractorWraper(nn.Module):

    def __init__(self, module):
        super(SingleRoIExtractorWraper, self).__init__()
        self.module = module

        pooling_name = type(self.module.roi_layers[0]).__name__
        self.roi_extractor = build_roi_extractor(pooling_name, module)
        self.featmap_strides = self.module.featmap_strides
        self.num_inputs = self.module.num_inputs

    def forward(self, feats, rois, roi_scale_factor=None):
        return self.roi_extractor(feats, rois, roi_scale_factor)
