import torch
from torch import nn

import mmdet2trt
from mmdet2trt.models.builder import register_wraper


@register_wraper('mmdet.core.anchor.point_generator.PointGenerator')
class PointGeneratorWraper(nn.Module):

    def __init__(self, module):
        super(PointGeneratorWraper, self).__init__()

    def forward(self, featmap, stride):
        shift_yy, shift_xx = mmdet2trt.ops.arange_gridmesh(
            featmap, strides=[stride, stride])

        shift_yy = shift_yy.flatten()
        shift_xx = shift_xx.flatten()
        stride = shift_yy * 0. + stride
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        return shifts


@register_wraper('mmdet.core.anchor.point_generator.MlvlPointGenerator')
class MlvlPointGeneratorWraper(nn.Module):

    def __init__(self, module):
        super(MlvlPointGeneratorWraper, self).__init__()
        self.module = module
        self.strides = module.strides
        self.offset = module.offset
        self.num_levels = module.num_levels

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(y.size(0))
        yy = y.view(-1, 1).repeat(1, x.size(0)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def forward(self, featmap_sizes, device, with_stride=False):
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                device='cuda',
                with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 device='cuda',
                                 with_stride=False):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0., feat_w, device=device) +
                   self.offset) * stride_w
        shift_y = (torch.arange(0., feat_h, device=device) +
                   self.offset) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_zeros((shift_xx.size(0), )) + stride_w
            stride_h = shift_xx.new_zeros((shift_yy.size(0), )) + stride_h
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                 dim=-1)
        all_points = shifts.to(device)
        return all_points
