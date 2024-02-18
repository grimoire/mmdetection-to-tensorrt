import logging
from typing import List, Tuple, Union

import mmengine
import numpy as np
import torch
from addict import Addict
from mmdet.models import BaseDetector
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.registry import DATASETS, MODELS, init_default_scope
from mmengine.structures import InstanceData
from torch import Tensor
from torch2trt_dynamic import TRTModule

import mmcv
from mmcv.transforms import Compose

logger = logging.getLogger('mmdet2trt')


def init_trt_model(trt_model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_model_path))

    return model_trt


def inference_trt_model(model, img, cfg, device):
    if isinstance(cfg, str):
        cfg = mmengine.Config.fromfile(cfg)

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

    with torch.inference_mode():
        result = model(tensor)
        result = list(result)
        result[1] = result[1] / scale_factor

    return result


def get_dataset_meta(model_cfg):
    init_default_scope(model_cfg.get('default_scope', 'mmdet'))
    dataset = model_cfg.val_dataloader.dataset
    dataset.lazy_init = True
    dataset = DATASETS.build(model_cfg.val_dataloader.dataset)
    return dataset.metainfo


def get_classes_from_config(model_cfg):
    try:
        return get_dataset_meta(model_cfg)['classes']
    except Exception as e:
        logger.warning('Load class names from dataset failed. with error:')
        raise e


class TRTDetector(BaseDetector):
    """TRTDetector."""

    def __init__(self, trt_module, model_cfg, device_id=0):
        super().__init__()
        if isinstance(model_cfg, str):
            model_cfg = mmengine.Config.fromfile(model_cfg)
        init_default_scope(model_cfg.get('default_scope', 'mmdet'))
        # self.CLASSES = get_classes_from_config(model_cfg)
        self.dataset_meta = get_dataset_meta(model_cfg)
        self.cfg = model_cfg
        self.device_id = device_id

        model = trt_module
        if isinstance(trt_module, str):
            model = TRTModule()
            model.load_state_dict(torch.load(trt_module))
        self.model = model

        self.data_preprocessor = self._build_data_preprocessor(model_cfg)

    def _build_data_preprocessor(self, cfg):
        return MODELS.build(cfg.model.data_preprocessor)

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        raise NotImplementedError('This method is not implemented.')

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        raise NotImplementedError('This method is not implemented.')

    @torch.inference_mode()
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        def __rescale_bboxes(bboxes, scale_factor):
            scale_factor = bboxes.new_tensor(scale_factor)[None, :]
            if scale_factor.size(-1) == 2:
                scale_factor = scale_factor.repeat(1, 2)
            return bboxes / scale_factor

        outputs = self.forward_test(batch_inputs)
        batch_num_dets, batch_boxes, batch_scores, batch_labels = outputs[:4]
        batch_num_dets = batch_num_dets.cpu()
        batch_masks = None if len(outputs) < 5 else outputs[4]
        batch_size = batch_inputs.shape[0]

        results = []
        for i in range(batch_size):
            in_data_sample = batch_data_samples[i]
            metainfo = in_data_sample.metainfo

            num_dets = batch_num_dets[i]
            bboxes = batch_boxes[i, :num_dets]
            labels = batch_labels[i, :num_dets].int()
            scores = batch_scores[i, :num_dets]

            old_bboxes = bboxes

            if rescale:
                assert 'scale_factor' in metainfo
                bboxes = __rescale_bboxes(bboxes, metainfo['scale_factor'])

            pred_instances = InstanceData(metainfo=metainfo)
            pred_instances.scores = scores
            pred_instances.bboxes = bboxes
            pred_instances.labels = labels

            # if 'border' in img_metas[i]:
            #     # offset pixel of the top-left corners between original image
            #     # and padded/enlarged image, 'border' is used when exporting
            #     # CornerNet and CentripetalNet to onnx
            #     x_off = img_metas[i]['border'][2]
            #     y_off = img_metas[i]['border'][0]
            #     dets[:, [0, 2]] -= x_off
            #     dets[:, [1, 3]] -= y_off
            #     dets[:, :4] *= (dets[:, :4] > 0).astype(dets.dtype)

            if batch_masks is not None:
                masks = batch_masks[i, :num_dets].unsqueeze(1)
                class_agnostic = True
                if num_dets > 0:
                    masks = FCNMaskHead._predict_by_feat_single(
                        Addict(class_agnostic=class_agnostic),
                        masks,
                        old_bboxes,
                        labels,
                        img_meta=metainfo,
                        rcnn_test_cfg=self.cfg.model.test_cfg.rcnn,
                        rescale=rescale,
                        activate_map=True)
                pred_instances.masks = masks

            out_data_sample = DetDataSample(
                metainfo=metainfo, pred_instances=pred_instances)
            results.append(out_data_sample)
        return results

    def forward_test(self, imgs):
        input_data = imgs.contiguous()
        with torch.cuda.device(self.device_id), torch.inference_mode():
            outputs = self.model(input_data)
        return outputs


def create_wrap_detector(trt_model, model_cfg, device_id=0):
    if isinstance(trt_model, str):
        trt_model = init_trt_model(trt_model)
    return TRTDetector(trt_model, model_cfg, device_id).cuda(device_id)
