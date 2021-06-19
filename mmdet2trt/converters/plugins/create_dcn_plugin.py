import ctypes
import os.path as osp
from collections.abc import Iterable

import numpy as np
import tensorrt as trt

from .globals import dir_path

ctypes.CDLL(osp.join(dir_path, 'libamirstan_plugin.so'))


def create_dcn_plugin(layer_name,
                      stride=[1, 1],
                      padding=[0, 0],
                      dilation=[1, 1],
                      deformable_group=1,
                      group=1):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'DeformableConvPluginDynamic', '1', '')
    if not isinstance(stride, Iterable):
        stride = [stride, stride]

    if not isinstance(padding, Iterable):
        padding = [padding, padding]

    if not isinstance(dilation, Iterable):
        dilation = [dilation, dilation]

    pfc = trt.PluginFieldCollection()

    pf_stride = trt.PluginField('stride', np.array(stride, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    pf_padding = trt.PluginField('padding', np.array(padding, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    pfc.append(pf_padding)

    pf_dilation = trt.PluginField('dilation',
                                  np.array(dilation, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_dilation)

    pf_deformable_group = trt.PluginField(
        'deformable_group', np.array([deformable_group], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_deformable_group)

    pf_group = trt.PluginField('group', np.array([group], dtype=np.int32),
                               trt.PluginFieldType.INT32)
    pfc.append(pf_group)

    return creator.create_plugin(layer_name, pfc)


def create_dcnv2_plugin(layer_name,
                        stride=[1, 1],
                        padding=[0, 0],
                        dilation=[1, 1],
                        deformable_group=1,
                        group=1):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'ModulatedDeformableConvPluginDynamic', '1', '')
    if not isinstance(stride, Iterable):
        stride = [stride, stride]

    if not isinstance(padding, Iterable):
        padding = [padding, padding]

    if not isinstance(dilation, Iterable):
        dilation = [dilation, dilation]

    pfc = trt.PluginFieldCollection()

    pf_stride = trt.PluginField('stride', np.array(stride, dtype=np.int32),
                                trt.PluginFieldType.INT32)
    pfc.append(pf_stride)

    pf_padding = trt.PluginField('padding', np.array(padding, dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    pfc.append(pf_padding)

    pf_dilation = trt.PluginField('dilation',
                                  np.array(dilation, dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_dilation)

    pf_deformable_group = trt.PluginField(
        'deformable_group', np.array([deformable_group], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_deformable_group)

    pf_group = trt.PluginField('group', np.array([group], dtype=np.int32),
                               trt.PluginFieldType.INT32)
    pfc.append(pf_group)

    return creator.create_plugin(layer_name, pfc)
