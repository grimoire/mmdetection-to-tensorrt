import numpy as np
from collections.abc import Iterable

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_gridanchordynamic_plugin(layer_name,
                                    base_size,
                                    stride,
                                    scales=np.array([1.]),
                                    ratios=np.array([1.]),
                                    scale_major=True,
                                    center_x=-1,
                                    center_y=-1,
                                    base_anchors=None):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'GridAnchorDynamicPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_base_size = trt.PluginField("base_size",
                                   np.array([base_size], dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    pfc.append(pf_base_size)

    pf_stride = trt.PluginField("stride", np.array([stride], dtype=np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    pf_scales = trt.PluginField("scales",
                                np.array(scales).astype(np.float32),
                                trt.PluginFieldType.FLOAT32)
    pfc.append(pf_scales)

    pf_ratios = trt.PluginField("ratios",
                                np.array(ratios).astype(np.float32),
                                trt.PluginFieldType.FLOAT32)
    pfc.append(pf_ratios)

    pf_scale_major = trt.PluginField(
        "scale_major", np.array([int(scale_major)], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_scale_major)

    pf_center_x = trt.PluginField("center_x",
                                  np.array([center_x], dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_center_x)

    pf_center_y = trt.PluginField("center_y",
                                  np.array([center_y], dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_center_y)

    if base_anchors is not None:
        pf_base_anchors = trt.PluginField(
            "base_anchors",
            np.array(base_anchors).astype(np.float32),
            trt.PluginFieldType.FLOAT32)
        pfc.append(pf_base_anchors)

    return creator.create_plugin(layer_name, pfc)
