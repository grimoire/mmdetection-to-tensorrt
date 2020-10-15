import numpy as np
from collections.abc import Iterable

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_deformable_pool_plugin(layer_name, out_size, spatial_scale,
                                  sampling_ratio, gamma):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'DeformablePoolPluginDynamic', '1', '')

    if not isinstance(out_size, Iterable):
        out_size = [out_size, out_size]

    pfc = trt.PluginFieldCollection()

    pf_out_size = trt.PluginField("out_size", np.array(out_size,
                                                       dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_out_size)

    pf_spatial_scale = trt.PluginField(
        "spatial_scale", np.array([spatial_scale], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_spatial_scale)

    pf_sampling_ratio = trt.PluginField(
        "sampling_ratio", np.array([sampling_ratio], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_sampling_ratio)

    pf_gamma = trt.PluginField("gamma", np.array([gamma], dtype=np.float32),
                               trt.PluginFieldType.FLOAT32)
    pfc.append(pf_gamma)

    return creator.create_plugin(layer_name, pfc)
