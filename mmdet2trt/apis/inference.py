from torch2trt import TRTModule
from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose
import torch
import mmcv
import numpy as np
import time
import logging

def init_detector(trt_model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path))

    return model_trt


def inference_detector(model, img, cfg, device):
    if isinstance(cfg, str):
        cfg = mmcv.Config.fromfile(cfg)
    
    device = torch.device(device)
    
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)

    tensor = data['img'][0].unsqueeze(0).to(device)
    img_metas = data['img_metas']
    scale_factor = img_metas[0].data['scale_factor']
    scale_factor = torch.tensor(scale_factor, dtype=torch.float32, device=device)

    with torch.no_grad():
        result = model(tensor)
        result = list(result)
        result[1] = result[1]/scale_factor

    return result


