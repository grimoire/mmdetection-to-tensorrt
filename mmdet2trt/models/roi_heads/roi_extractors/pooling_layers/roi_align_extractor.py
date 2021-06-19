from torch import nn


class RoiAlignExtractor(nn.Module):

    def __init__(self, module):
        super(RoiAlignExtractor, self).__init__()
        self.module = module

        self.roi_layers = self.module.roi_layers
        self.featmap_strides = self.module.featmap_strides
        self.finest_scale = self.module.finest_scale

    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        module = self.module
        out_size = module.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(
            rois.size(0), module.out_channels, *out_size)

        target_lvls = module.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = module.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask
            if inds.any():
                rois_ = rois[inds, :]
                roi_feats_t = module.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in module.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats

        # return self.module(feats, rois, roi_scale_factor)
