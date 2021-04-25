import ctypes
import os.path as osp

import numpy as np
import tensorrt as trt

from .globals import dir_path

ctypes.CDLL(osp.join(dir_path, 'libamirstan_plugin.so'))


def create_batchednms_plugin(layer_name,
                             scoreThreshold,
                             iouThreshold,
                             topK,
                             keepTopK,
                             numClasses=1,
                             backgroundLabelId=-1,
                             shareLocation=False,
                             isNormalized=False,
                             clipBoxes=True):

    plugin_name = 'BatchedNMS_TRT_CUSTOM'

    creator = trt.get_plugin_registry().get_plugin_creator(
        plugin_name, '1', '')

    pfc = trt.PluginFieldCollection()

    # plugins

    pf_shareLocation = trt.PluginField(
        'shareLocation', np.array([shareLocation], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_shareLocation)

    pf_isNormalized = trt.PluginField('isNormalized',
                                      np.array([isNormalized], dtype=np.int32),
                                      trt.PluginFieldType.INT32)
    pfc.append(pf_isNormalized)

    pf_clipBoxes = trt.PluginField('clipBoxes',
                                   np.array([clipBoxes], dtype=np.int32),
                                   trt.PluginFieldType.INT32)
    pfc.append(pf_clipBoxes)

    pf_backgroundLabelId = trt.PluginField(
        'backgroundLabelId', np.array([backgroundLabelId], dtype=np.int32),
        trt.PluginFieldType.INT32)
    pfc.append(pf_backgroundLabelId)

    pf_numClasses = trt.PluginField('numClasses',
                                    np.array([numClasses], dtype=np.int32),
                                    trt.PluginFieldType.INT32)
    pfc.append(pf_numClasses)

    pf_topK = trt.PluginField('topK', np.array([topK], dtype=np.int32),
                              trt.PluginFieldType.INT32)
    pfc.append(pf_topK)

    pf_keepTopK = trt.PluginField('keepTopK',
                                  np.array([keepTopK], dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_keepTopK)

    pf_scoreThreshold = trt.PluginField(
        'scoreThreshold', np.array([scoreThreshold], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_scoreThreshold)

    pf_iouThreshold = trt.PluginField(
        'iouThreshold', np.array([iouThreshold], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_iouThreshold)

    # plugins

    return creator.create_plugin(layer_name, pfc)
