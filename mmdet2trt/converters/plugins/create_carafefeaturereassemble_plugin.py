import numpy as np

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_carafefeaturereassemble_plugin(layer_name,
                                        scale_factor,
                                        up_kernel,
                                        up_group,
                                        type_id=trt.DataType.FLOAT):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'CarafeFeatureReassemblePluginDynamic', '1', '')
    
    pfc = trt.PluginFieldCollection()

    pf_scale_factor = trt.PluginField("scale_factor", np.array(
        [scale_factor], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_scale_factor)

    pf_up_kernel = trt.PluginField("up_kernel", np.array([up_kernel], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_up_kernel)

    pf_up_group = trt.PluginField("up_group", np.array([up_group], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_up_group)
    
    pf_type_id = trt.PluginField("type_id", np.array(
        [type_id], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_type_id)
    
    return creator.create_plugin(layer_name, pfc)
