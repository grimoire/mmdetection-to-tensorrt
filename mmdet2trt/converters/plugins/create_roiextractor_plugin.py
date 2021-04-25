import ctypes
import os.path as osp

import numpy as np
import tensorrt as trt

from .globals import dir_path

ctypes.CDLL(osp.join(dir_path, "libamirstan_plugin.so"))


def create_roiextractor_plugin(layer_name, out_size, sample_num,
                               featmap_strides, roi_scale_factor, finest_scale,
                               aligned):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'RoiExtractorPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_out_size = trt.PluginField("out_size",
                                  np.array([out_size], dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_out_size)

    pf_sample_num = trt.PluginField("sample_num",
                                    np.array([sample_num], dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    pfc.append(pf_sample_num)

    pf_featmap_strides = trt.PluginField(
        "featmap_strides", np.array(featmap_strides, dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_featmap_strides)

    pf_roi_scale_factor = trt.PluginField(
        "roi_scale_factor", np.array([roi_scale_factor], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_roi_scale_factor)

    pf_finest_scale = trt.PluginField("finest_scale",
                                      np.array([finest_scale], dtype=np.int32),
                                      trt.PluginFieldType.INT32)
    pfc.append(pf_finest_scale)

    pf_aligned = trt.PluginField("aligned", np.array([aligned],
                                                     dtype=np.int32),
                                 trt.PluginFieldType.INT32)
    pfc.append(pf_aligned)

    return creator.create_plugin(layer_name, pfc)
