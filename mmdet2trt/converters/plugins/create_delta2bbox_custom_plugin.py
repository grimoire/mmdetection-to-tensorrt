import numpy as np
from collections.abc import Iterable

import os
import os.path as osp
from .globals import dir_path
import ctypes
ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))

import tensorrt as trt


def create_delta2bbox_custom_plugin(layer_name,
                                    use_sigmoid_cls,
                                    min_num_bbox,
                                    num_classes=1,
                                    target_means=[0., 0., 0., 0.],
                                    target_stds=[1., 1., 1., 1.]):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'Delta2BBoxPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_use_sigmoid_cls = trt.PluginField(
        "use_sigmoid_cls", np.array([use_sigmoid_cls], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_use_sigmoid_cls)

    pf_min_num_bbox = trt.PluginField("min_num_bbox",
                                      np.array([min_num_bbox], dtype=np.int32),
                                      trt.PluginFieldType.INT32)
    pfc.append(pf_min_num_bbox)

    pf_num_classes = trt.PluginField("num_classes",
                                     np.array([num_classes], dtype=np.int32),
                                     trt.PluginFieldType.INT32)
    pfc.append(pf_num_classes)

    pf_target_means = trt.PluginField("target_means",
                                      np.array(target_means, dtype=np.float32),
                                      trt.PluginFieldType.FLOAT32)
    pfc.append(pf_target_means)

    pf_target_stds = trt.PluginField("target_stds",
                                     np.array(target_stds, dtype=np.float32),
                                     trt.PluginFieldType.FLOAT32)
    pfc.append(pf_target_stds)

    return creator.create_plugin(layer_name, pfc)
