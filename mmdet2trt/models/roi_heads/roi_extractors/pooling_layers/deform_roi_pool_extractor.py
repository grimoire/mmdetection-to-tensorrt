import torch
from mmdet2trt.models.builder import build_wraper, register_wraper
from torch import nn

import mmcv.ops

deformable_roi_pool_wrap = mmcv.ops.deform_roi_pool


@register_wraper('mmcv.ops.DeformRoIPoolPack')
class DeformRoIPoolPackWraper(nn.Module):

    def __init__(self, module):
        super(DeformRoIPoolPackWraper, self).__init__()
        self.module = module

    def forward(self, input, rois):
        # assert input.size(1) == self.module.output_channels
        x = deformable_roi_pool_wrap(input, rois, None,
                                     self.module.output_size,
                                     self.module.spatial_scale,
                                     self.module.sampling_ratio,
                                     self.module.gamma)
        rois_num = rois.size(0)
        offset = self.module.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.module.output_size[0],
                             self.module.output_size[1])
        return deformable_roi_pool_wrap(input, rois, offset,
                                        self.module.output_size,
                                        self.module.spatial_scale,
                                        self.module.sampling_ratio,
                                        self.module.gamma)


@register_wraper('mmcv.ops.ModulatedDeformRoIPoolPack')
class ModulatedDeformRoIPoolPackWraper(nn.Module):

    def __init__(self, module):
        super(ModulatedDeformRoIPoolPackWraper, self).__init__()
        self.module = module

    def forward(self, input, rois):
        x = deformable_roi_pool_wrap(input, rois, None,
                                     self.module.output_size,
                                     self.module.spatial_scale,
                                     self.module.sampling_ratio,
                                     self.module.gamma)
        rois_num = rois.size(0)
        offset = self.module.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.module.output_size[0],
                             self.module.output_size[1])
        mask = self.module.mask_fc(x.view(rois_num, -1))
        mask = mask.view(rois_num, 1, self.module.output_size[0],
                         self.module.output_size[1])
        d = deformable_roi_pool_wrap(input, rois, offset,
                                     self.module.output_size,
                                     self.module.spatial_scale,
                                     self.module.sampling_ratio,
                                     self.module.gamma)
        return d * mask


class DeformRoiPoolExtractor(nn.Module):

    def __init__(self, module):
        super(DeformRoiPoolExtractor, self).__init__()
        self.module = module

        self.roi_layers = [
            build_wraper(layer) for layer in self.module.roi_layers
        ]
        self.featmap_strides = self.module.featmap_strides
        self.finest_scale = self.module.finest_scale

    def _get_layer_mask(self, x, value):
        value += 2e-15
        t1 = (x - value).clamp(0, value)
        t2 = value - x.clamp(0, value)
        return t1 / (t1 + t2 + 1e-15)

    def _get_between_mask(self, x, min_value, max_value=None):
        return ((x >= min_value) & (x < max_value)).float()

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1)
        return target_lvls

    def forward(self, feats, rois, roi_scale_factor=None):
        out_feats = []

        num_levels = len(self.roi_layers)

        target_lvls = self.map_roi_levels(
            rois, num_levels).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-5

        target_lvls = target_lvls
        if roi_scale_factor is not None:
            rois = self.module.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            feat = feats[i]
            mask = self._get_between_mask(target_lvls, i, i + 1)

            roi_feats_t = self.roi_layers[i](feat, rois)
            roi_feats_t *= mask

            out_feats.append(roi_feats_t)

        return sum(out_feats)
