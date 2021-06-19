import numpy as np
import torch
from mmdet.datasets.pipelines import Compose
from torch2trt_dynamic import TRTModule

import mmcv


def init_detector(trt_model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path))

    return model_trt


def inference_detector(model, img, cfg, device):
    if isinstance(cfg, str):
        cfg = mmcv.Config.fromfile(cfg)

    device = torch.device(device)

    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)

    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = test_pipeline(data)

    tensor = data['img'][0]
    if isinstance(tensor, mmcv.parallel.DataContainer):
        tensor = tensor.data
    tensor = tensor.unsqueeze(0).to(device)
    img_metas = data['img_metas']
    scale_factor = img_metas[0].data['scale_factor']
    scale_factor = torch.tensor(
        scale_factor, dtype=torch.float32, device=device)

    with torch.no_grad():
        result = model(tensor)
        result = list(result)
        result[1] = result[1] / scale_factor

    return result
