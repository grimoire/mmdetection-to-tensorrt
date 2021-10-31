import logging

import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.datasets.pipelines import Compose
from mmdet.models import BaseDetector
from torch2trt_dynamic import TRTModule

import mmcv


def init_trt_model(trt_model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path))

    return model_trt


def inference_trt_model(model, img, cfg, device):
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


def get_classes_from_config(model_cfg):
    model_cfg_str = model_cfg
    if isinstance(model_cfg, str):
        model_cfg = mmcv.Config.fromfile(model_cfg)

    from mmdet.datasets import DATASETS, build_dataset

    try:
        dataset = build_dataset(model_cfg)
        return dataset.CLASSES
    except Exception:
        logging.warn(
            'Can not load dataset from config. Use default CLASSES instead.')

    module_dict = DATASETS.module_dict
    data_cfg = model_cfg.data

    def get_module_from_train_val(train_val_cfg):
        while train_val_cfg.type == 'RepeatDataset' or \
         train_val_cfg.type == 'MultiImageMixDataset':
            train_val_cfg = train_val_cfg.dataset
        return module_dict[train_val_cfg.type]

    data_cfg_type_list = ['train', 'val', 'test']

    MODULE = None
    for data_cfg_type in data_cfg_type_list:
        if data_cfg_type in data_cfg:
            tmp_data_cfg = data_cfg.get(data_cfg_type)
            MODULE = get_module_from_train_val(tmp_data_cfg)
            if 'classes' in tmp_data_cfg:
                return MODULE.get_classes(tmp_data_cfg.classes)
            break

    assert MODULE is not None, f'No dataset config found in: {model_cfg_str}'

    return MODULE.CLASSES


class TRTDetector(BaseDetector):
    """TRTDetector."""

    def __init__(self, trt_module, model_cfg, device_id=0):
        super().__init__()

        self._dummy_param = torch.nn.Parameter(torch.tensor(0.0))
        if isinstance(model_cfg, str):
            model_cfg = mmcv.Config.fromfile(model_cfg)
        self.CLASSES = get_classes_from_config(model_cfg)
        self.cfg = model_cfg
        self.device_id = device_id

        model = trt_module
        if isinstance(trt_module, str):
            model = TRTModule()
            model.load_state_dict(torch.load(trt_module))
        self.model = model

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def val_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def train_step(self, data, optimizer):
        raise NotImplementedError('This method is not implemented.')

    def aforward_test(self, *, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward(self, img, img_metas, *args, **kwargs):
        outputs = self.forward_test(img, img_metas, *args, **kwargs)
        batch_num_dets, batch_boxes, batch_scores, batch_labels = outputs[:4]
        batch_dets = torch.cat(
            [batch_boxes, batch_scores.unsqueeze(-1)], dim=-1)
        batch_masks = None if len(outputs) < 5 else outputs[4]
        batch_size = img[0].shape[0]
        img_metas = img_metas[0]
        results = []
        rescale = kwargs.get('rescale', True)
        for i in range(batch_size):
            num_dets = batch_num_dets[i]
            dets, labels = batch_dets[i][:num_dets], batch_labels[i][:num_dets]
            if rescale:
                scale_factor = img_metas[i]['scale_factor']

                if isinstance(scale_factor, (list, tuple, np.ndarray)):
                    assert len(scale_factor) == 4
                    scale_factor = dets.new_tensor(scale_factor)[
                        None, :]  # [1,4]
                dets[:, :4] /= scale_factor

            if 'border' in img_metas[i]:
                # offset pixel of the top-left corners between original image
                # and padded/enlarged image, 'border' is used when exporting
                # CornerNet and CentripetalNet to onnx
                x_off = img_metas[i]['border'][2]
                y_off = img_metas[i]['border'][0]
                dets[:, [0, 2]] -= x_off
                dets[:, [1, 3]] -= y_off
                dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)

            dets_results = bbox2result(dets, labels, len(self.CLASSES))

            if batch_masks is not None:
                masks = batch_masks[i]
                img_h, img_w = img_metas[i]['img_shape'][:2]
                ori_h, ori_w = img_metas[i]['ori_shape'][:2]
                masks = masks[:, :img_h, :img_w]
                if rescale:
                    masks = masks.astype(np.float32)
                    masks = torch.from_numpy(masks)
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0), size=(ori_h, ori_w))
                    masks = masks.squeeze(0).detach().numpy()
                if masks.dtype != np.bool:
                    masks = masks >= 0.5
                segms_results = [[] for _ in range(len(self.CLASSES))]
                for j in range(len(dets)):
                    segms_results[labels[j]].append(masks[j])
                results.append((dets_results, segms_results))
            else:
                results.append(dets_results)
        return results

    def forward_test(self, imgs, *args, **kwargs):
        input_data = imgs[0].contiguous()
        with torch.cuda.device(self.device_id), torch.no_grad():
            outputs = self.model(input_data)
        return outputs


def create_wrap_detector(trt_model, model_cfg, device_id=0):
    if isinstance(trt_model, str):
        trt_model = init_trt_model(trt_model)
    return TRTDetector(trt_model, model_cfg, device_id).cuda(device_id)
