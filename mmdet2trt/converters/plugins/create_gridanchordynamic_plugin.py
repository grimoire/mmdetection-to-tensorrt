import ctypes
import os.path as osp

import numpy as np
import tensorrt as trt

from .globals import dir_path

ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))


def create_gridanchordynamic_plugin(layer_name, stride):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'GridAnchorDynamicPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_stride = trt.PluginField("stride", np.array([stride], dtype=np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    return creator.create_plugin(layer_name, pfc)
