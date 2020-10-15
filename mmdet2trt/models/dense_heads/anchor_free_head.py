import torch
from mmdet2trt.models.builder import register_wraper, build_wraper
import torch
from torch import nn

import mmdet2trt.ops as mm2trt_ops

from mmdet2trt.core.post_processing.batched_nms import BatchedNMS


class AnchorFreeHeadWraper(nn.Module):
    def __init__(self, module):
        super(AnchorFreeHeadWraper, self).__init__()
        self.module = module

        self.test_cfg = module.test_cfg
        self.num_classes = self.module.num_classes
        self.rcnn_nms = BatchedNMS(module.test_cfg.score_thr,
                                   module.test_cfg.nms.iou_threshold,
                                   backgroundLabelId=-1)

    def forward(self, feat, x):
        raise NotImplementedError

    def _get_points_single(self, feat, stride, flatten=False):
        y, x = mm2trt_ops.arange_gridmesh(feat)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_points(self, features, flatten=False):

        mlvl_points = []
        for i, feat in enumerate(features):
            mlvl_points.append(
                self._get_points_single(feat,
                                        self.module.strides[i],
                                        flatten=flatten))
        return mlvl_points