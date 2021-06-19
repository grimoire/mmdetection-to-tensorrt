from mmdet2trt.models.builder import register_wraper
from torch import nn


class AnchorGeneratorSingle(nn.Module):

    def __init__(self, module, index):
        super(AnchorGeneratorSingle, self).__init__()
        self.generator = module

        self.index = index
        self.base_size = module.base_sizes[index]
        if hasattr(module, 'base_anchors'):
            self.base_anchor = module.base_anchors[index].contiguous()
        else:
            self.scales = module.scales
            self.ratios = module.ratios
            self.scale_major = module.scale_major
            self.ctr = None
            if module.centers is not None:
                self.ctr = module.centers[index]

    def forward(self, x, stride=None, device='cuda'):
        if stride is None:
            stride = self.generator.strides[self.index]
        height, width = x.shape[2:]
        return self.generator.single_level_grid_anchors(
            self.generator.base_anchors[self.index].to(device),
            (height, width),
            stride=stride,
            device=device)


@register_wraper('mmdet.core.anchor.anchor_generator.YOLOAnchorGenerator')
@register_wraper('mmdet.core.AnchorGenerator')
class AnchorGeneratorWraper(nn.Module):

    def __init__(self, module):
        super(AnchorGeneratorWraper, self).__init__()
        self.generator = module

        self.base_sizes = module.base_sizes
        if hasattr(module, 'base_anchors'):
            self.base_anchors = [
                anchor.contiguous() for anchor in module.base_anchors
            ]
        else:
            self.scales = module.scales
            self.ratios = module.ratios
            self.scale_major = module.scale_major
            self.centers = module.centers

        self.ag_single_list = [
            AnchorGeneratorSingle(module, index)
            for index in range(len(self.base_sizes))
        ]

    def forward(self, feat_list, device='cuda'):
        multi_level_anchors = []
        for index, x in enumerate(feat_list):
            anchors = self.ag_single_list[index](
                x, stride=self.generator.strides[index], device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors


@register_wraper('mmdet.core.anchor.anchor_generator.SSDAnchorGenerator')
class SSDAnchorGeneratorWraper(nn.Module):

    def __init__(self, module):
        super(SSDAnchorGeneratorWraper, self).__init__()
        self.generator = module
        self.mlvl_anchors = None

    def forward(self, feat_list, device='cuda'):
        if self.mlvl_anchors is None:
            num_levels = len(feat_list)
            featmap_sizes = [
                feat_list[i].shape[-2:] for i in range(num_levels)
            ]

            mlvl_anchors = self.generator.grid_anchors(
                featmap_sizes, device=device)
            self.mlvl_anchors = mlvl_anchors

        return self.mlvl_anchors
