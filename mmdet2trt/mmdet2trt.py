import logging
import time
from typing import Any, Dict

import mmengine
import tensorrt as trt
import torch
from mmdet2trt.models.builder import build_wrapper
from mmdet2trt.models.detectors import TwoStageDetectorWraper
from mmdet.apis import init_detector
from torch2trt_dynamic import BuildEngineConfig, module2trt

logger = logging.getLogger('mmdet2trt')


class Int8CalibDataset():
    r"""
    datas used to calibrate int8 model
    feed to int8_calib_dataset
    """

    def __init__(self, image_paths, config, shape_ranges):
        r"""
        datas used to calibrate int8 model
        feed to int8_calib_dataset
        Args:
            image_paths (list[str]): image paths to calib
            config (str|dict): config of mmdetection model
            shape_ranges: same as mmdet2trt
        """
        from mmcv.transforms import Compose
        from mmengine.registry import init_default_scope
        if isinstance(config, str):
            config = mmengine.Config.fromfile(config)

        init_default_scope(config.get('default_scope', 'mmdet'))
        self.cfg = config
        self.image_paths = image_paths
        self.opt_shape = shape_ranges['x']['opt']

        test_pipeline = config.val_dataloader.dataset.pipeline
        self.test_pipeline = Compose(test_pipeline)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        data = dict(img=image_path, img_path=image_path)
        data = self.test_pipeline(data)

        tensor = data['inputs'].unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, self.opt_shape[-2:]).squeeze(0)

        return dict(x=tensor.cuda())


def _get_shape_ranges(config):
    img_scale = config.test_pipeline[1]['scale']
    min_scale = min(img_scale)
    max_scale = max(img_scale) + 32
    opt_shape_param = dict(
        x=dict(
            min=[1, 3, min_scale, min_scale],
            opt=[1, 3, img_scale[1], img_scale[0]],
            max=[1, 3, max_scale, max_scale],
        ))
    return opt_shape_param


def _make_dummy_input(shape_ranges, device):
    dummy_shape = shape_ranges['x']['opt']
    dummy_input = torch.rand(dummy_shape).to(device)
    dummy_input = (dummy_input - 0.45) / 0.27
    dummy_input = dummy_input.contiguous()
    return dummy_input


def _get_trt_calib_algorithm(int8_calib_alg):
    int8_calib_algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    if int8_calib_alg == 'minmax':
        int8_calib_algorithm = trt.CalibrationAlgoType.MINMAX_CALIBRATION
    elif int8_calib_alg == 'entropy':
        int8_calib_algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    else:
        raise ValueError('int8_calib_alg should be "minmax" or "entropy"')
    return int8_calib_algorithm


def mmdet2trt(config,
              checkpoint: str,
              device: str = 'cuda:0',
              fp16_mode: bool = False,
              int8_mode: bool = False,
              int8_calib_dataset: Any = None,
              int8_calib_alg: str = 'entropy',
              max_workspace_size: int = None,
              shape_ranges: Dict[str, Dict] = None,
              trt_log_level: str = 'INFO',
              return_wrap_model: bool = False,
              enable_mask=False):
    r"""
    create TensorRT model from MMDetection.
    Args:
        config (str): config file path of MMDetection model
        checkpoint (str): checkpoint file path of MMDetection model
        device (str): convert gpu device
        fp16_mode (bool): create fp16 mode engine.
        int8_mode (bool): create int8 mode engine.
        int8_calib_dataset (object): dataset object used to do data calibrate
        int8_calib_alg (str): how to calibrate int8, ["minmax", "entropy"]
        max_workspace_size (int): TensorRT workspace size.
            some tactic might need large workspace.
        shape_ranges (Dict[str, Dict]): the min/optimize/max shape of
            input tensor
        trt_log_level (str): TensorRT log level,
            options: ["VERBOSE", "INFO", "WARNING", "ERROR"]
        return_wrap_model (bool): return pytorch wrap model, used for debug
        enable_mask (bool): if output the instance segmentation result
            (w/o post-process)
    """

    def _make_build_engine_config(shape_ranges, max_workspace_size,
                                  int8_calib_dataset):
        int8_calib_algorithm = _get_trt_calib_algorithm(int8_calib_alg)
        build_engine_config = BuildEngineConfig(
            shape_ranges=shape_ranges,
            pool_size=max_workspace_size,
            fp16=fp16_mode,
            int8=int8_mode,
            int8_calib_dataset=int8_calib_dataset,
            int8_calib_algorithm=int8_calib_algorithm,
            int8_batch_size=1)
        return build_engine_config

    device = torch.device(device)

    logger.info('Loading model from config: {}'.format(config))
    torch_model = init_detector(config, checkpoint=checkpoint, device=device)
    cfg = torch_model.cfg

    logger.info('Wrapping model')
    wrap_config = {'enable_mask': enable_mask}
    wrapped_model = build_wrapper(
        torch_model, TwoStageDetectorWraper, wrap_config=wrap_config)

    if shape_ranges is None:
        shape_ranges = _get_shape_ranges(cfg)

    dummy_input = _make_dummy_input(shape_ranges, device)

    logger.info('Model warmup.')
    with torch.cuda.device(device), torch.inference_mode():
        wrapped_model(dummy_input)

    logger.info('Converting model')
    start = time.time()
    with torch.cuda.device(device), torch.inference_mode():
        trt_log_level = getattr(trt.Logger, trt_log_level)
        build_engine_config = _make_build_engine_config(
            shape_ranges=shape_ranges,
            max_workspace_size=max_workspace_size,
            int8_calib_dataset=int8_calib_dataset)
        trt_model = module2trt(
            wrapped_model, [dummy_input],
            config=build_engine_config,
            log_level=trt_log_level)

    duration = time.time() - start
    logger.info('Conversion took {} s'.format(duration))

    if return_wrap_model:
        return trt_model, wrapped_model

    return trt_model


def mask_processor2trt(max_width,
                       max_height,
                       max_batch_size=1,
                       max_box_per_batch=10,
                       mask_size=[28, 28],
                       device='cuda:0',
                       fp16_mode=False,
                       max_workspace_size=None,
                       trt_log_level='INFO',
                       return_wrap_model=False):

    from mmdet2trt.models.roi_heads.mask_heads.fcn_mask_head import \
        MaskProcessor

    logger.info('Wrapping MaskProcessor')
    wrapped_model = MaskProcessor(max_width=max_width, max_height=max_height)

    batch_size = max_batch_size
    num_boxes = max_box_per_batch
    opt_shape_param = [[
        [1, 1] + mask_size,
        [batch_size, num_boxes] + mask_size,
        [batch_size, num_boxes] + mask_size,
    ], [
        [1, 1, 4],
        [batch_size, num_boxes, 4],
        [batch_size, num_boxes, 4],
    ]]
    device = torch.device(device)

    dummy_mask_shape = opt_shape_param[0][1]
    dummy_mask = torch.rand(dummy_mask_shape).to(device)
    dummy_mask = dummy_mask.contiguous()

    dummy_box_shape = opt_shape_param[1][1]
    dummy_box_pre = torch.rand(dummy_box_shape[0], dummy_box_shape[1],
                               2) * torch.tensor([max_width, max_height]) / 2
    dummy_box_post = torch.rand(dummy_box_shape[0], dummy_box_shape[1],
                                2) * torch.tensor([max_width, max_height]) / 2
    dummy_box = torch.cat([dummy_box_pre, dummy_box_pre + dummy_box_post],
                          dim=-1).to(device)

    logger.info('Converting MaskProcessor')
    start = time.time()
    with torch.cuda.device(device), torch.no_grad():
        trt_log_level = getattr(trt.Logger, trt_log_level)
        build_engine_config = BuildEngineConfig(
            shape_ranges=opt_shape_param,
            pool_size=max_workspace_size,
            fp16=fp16_mode)
        trt_model = module2trt(
            wrapped_model, [dummy_mask, dummy_box],
            config=build_engine_config,
            log_level=trt_log_level)

    duration = time.time() - start
    logger.info('Conversion took {} s'.format(duration))

    if return_wrap_model:
        return trt_model, wrapped_model

    return trt_model
