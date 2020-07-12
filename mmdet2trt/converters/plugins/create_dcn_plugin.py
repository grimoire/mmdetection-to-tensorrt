import numpy as np
from collections.abc import Iterable

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt



def create_dcn_plugin(layer_name,
                     out_channels,
                     kernel_size,
                     W,
                     type_id=trt.DataType.FLOAT,
                     stride=[1, 1],
                     padding=[0, 0],
                     dilation=[1, 1],
                     deformable_group=1,
                     group=1):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'DeformableConvPluginDynamic', '1', '')
    if not isinstance(kernel_size, Iterable):
        kernel_size = [kernel_size, kernel_size]

    if not isinstance(stride, Iterable):
        stride = [stride, stride]

    if not isinstance(padding, Iterable):
        padding = [padding, padding]

    if not isinstance(dilation, Iterable):
        dilation = [dilation, dilation]

    pfc = trt.PluginFieldCollection()

    pf_out_channels = trt.PluginField("out_dims", np.array(
        [out_channels], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_out_channels)

    pf_kernel_size = trt.PluginField("kernel_size", np.array(
        kernel_size, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_kernel_size)

    pf_W = trt.PluginField("W", W, trt.PluginFieldType.FLOAT32)
    pfc.append(pf_W)

    pf_type_id = trt.PluginField("type_id", np.array(
        [type_id], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_type_id)

    pf_stride = trt.PluginField("stride", np.array(
        stride, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    pf_padding = trt.PluginField("padding", np.array(
        padding, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_padding)

    pf_dilation = trt.PluginField("dilation", np.array(
        dilation, dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_dilation)

    pf_deformable_group = trt.PluginField("deformable_group", np.array(
        [deformable_group], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_deformable_group)

    pf_group = trt.PluginField("group", np.array(
        [group], dtype=np.int32), trt.PluginFieldType.INT32)
    pfc.append(pf_group)

    return creator.create_plugin(layer_name, pfc)
