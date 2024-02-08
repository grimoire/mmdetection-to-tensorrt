import argparse
import logging
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import tensorrt as trt
import torch
from mmdet2trt.models.builder import build_wrapper
from mmdet2trt.models.detectors import TwoStageDetectorWraper
from mmdet.apis import init_detector
from torch2trt_dynamic import BuildEngineConfig, module2trt

import mmcv

logger = logging.getLogger('mmdet2trt')


class Int8CalibDataset():
    r"""
    datas used to calibrate int8 model
    feed to int8_calib_dataset
    """

    def __init__(self, image_paths, config, opt_shape_param):
        r"""
        datas used to calibrate int8 model
        feed to int8_calib_dataset
        Args:
            image_paths (list[str]): image paths to calib
            config (str|dict): config of mmdetection model
            opt_shape_param: same as mmdet2trt
        """
        from mmdet.apis.inference import LoadImage
        from mmdet.datasets.pipelines import Compose
        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)

        self.cfg = config
        self.image_paths = image_paths
        self.opt_shape = opt_shape_param[0][1]

        test_pipeline = [LoadImage()] + config.data.test.pipeline[1:]
        self.test_pipeline = Compose(test_pipeline)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        data = dict(img=image_path)
        data = self.test_pipeline(data)

        tensor = data['img'][0].unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, self.opt_shape[-2:]).squeeze(0)

        return [tensor]


def _get_shape_ranges(config):
    img_scale = config.test_pipeline[1]['img_scale']
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
    create tensorrt model from mmdetection.
    Args:
        config (str): config file path of mmdetection model
        checkpoint (str): checkpoint file path of mmdetection model
        device (str): convert gpu device
        fp16_mode (bool): create fp16 mode engine.
        int8_mode (bool): create int8 mode engine.
        int8_calib_dataset (object): dataset object used to do data calibrate
        int8_calib_alg (str): how to calibrate int8, ["minmax", "entropy"]
        max_workspace_size (int): tensorrt workspace size.
            some tactic might need large workspace.
        shape_ranges (Dict[str, Dict]): the min/optimize/max shape of
            input tensor
        trt_log_level (str): tensorrt log level,
            options: ["VERBOSE", "INFO", "WARNING", "ERROR"]
        return_wrap_model (bool): return pytorch wrap model, used for debug
        enable_mask (bool): weither output the instance segmentation result
            (w/o postprocess)
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
    wrap_model = build_wrapper(
        torch_model, TwoStageDetectorWraper, wrap_config=wrap_config)

    if shape_ranges is None:
        shape_ranges = _get_shape_ranges(cfg)

    dummy_input = _make_dummy_input(shape_ranges, device)

    logger.info('Model warmup.')
    with torch.no_grad():
        wrap_model(dummy_input)

    logger.info('Converting model')
    start = time.time()
    with torch.cuda.device(device), torch.no_grad():
        trt_log_level = getattr(trt.Logger, trt_log_level)
        build_engine_config = _make_build_engine_config(
            shape_ranges=shape_ranges,
            max_workspace_size=max_workspace_size,
            int8_calib_dataset=int8_calib_dataset)
        trt_model = module2trt(
            wrap_model, [dummy_input],
            config=build_engine_config,
            log_level=trt_log_level)

    duration = time.time() - start
    logger.info('Conversion took {} s'.format(duration))

    if return_wrap_model:
        return trt_model, wrap_model

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
    wrap_model = MaskProcessor(max_width=max_width, max_height=max_height)

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
            wrap_model, [dummy_mask, dummy_box],
            config=build_engine_config,
            log_level=trt_log_level)

    duration = time.time() - start
    logger.info('Conversion took {} s'.format(duration))

    if return_wrap_model:
        return trt_model, wrap_model

    return trt_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to a mmdet Config file')
    parser.add_argument('checkpoint', help='Path to a mmdet Checkpoint file')
    parser.add_argument(
        'output', help='Path where tensorrt model will be saved')
    parser.add_argument(
        '--fp16', action='store_true', help='Enable fp16 inference')
    parser.add_argument(
        '--enable-mask', action='store_true', help='Enable mask output')
    parser.add_argument(
        '--save-engine',
        action='store_true',
        help='Enable saving TensorRT engine. '
        '(will be saved at Path(output).with_suffix(\'.engine\')).',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device used for conversion.')
    parser.add_argument(
        '--max-workspace-gb',
        type=float,
        default=None,
        help='The maximum `device` (GPU) temporary memory in GB (gigabytes)'
        ' which TensorRT can use at execution time.',
    )
    parser.add_argument(
        '--min-scale',
        type=int,
        nargs=4,
        default=None,
        help='Minimum input scale in '
        '[batch_size, channels, height, width] order.'
        ' Only used if all min-scale, opt-scale and max-scale are set.',
    )
    parser.add_argument(
        '--opt-scale',
        type=int,
        nargs=4,
        default=None,
        help='Optimal input scale in '
        '[batch_size, channels, height, width] order.'
        ' Only used if all min-scale, opt-scale and max-scale are set.',
    )
    parser.add_argument(
        '--max-scale',
        type=int,
        nargs=4,
        default=None,
        help='Maximum input scale in '
        '[batch_size, channels, height, width] order.'
        ' Only used if all min-scale, opt-scale and max-scale are set.',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Python logging level.',
    )
    parser.add_argument(
        '--trt-log-level',
        default='INFO',
        choices=['VERBOSE', 'INFO', 'WARNING', 'ERROR'],
        help='TensorRT logging level.',
    )
    parser.add_argument(
        '--output-names',
        nargs=4,
        type=str,
        default=['num_detections', 'boxes', 'scores', 'classes'],
        help='Names for the output nodes of the created TRTModule',
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    if all(
            getattr(args, x) is not None
            for x in ['min_scale', 'opt_scale', 'max_scale']):
        shape_range = dict(
            x=dict(min=args.min_scale, opt=args.opt_scale, max=args.max_scale))
    else:
        shape_range = None

    work_space_size = None
    if args.max_workspace_gb is not None:
        work_space_size = int(args.max_workspace_gb * 1e9)
    trt_model = mmdet2trt(
        args.config,
        args.checkpoint,
        device=args.device,
        fp16_mode=args.fp16,
        max_workspace_size=work_space_size,
        shape_ranges=shape_range,
        trt_log_level=args.trt_log_level,
        output_names=args.output_names,
        enable_mask=args.enable_mask)

    logger.info('Saving TRT model to: {}'.format(args.output))
    torch.save(trt_model.state_dict(), args.output)

    if args.save_engine:
        logger.info('Saving TRT model engine to: {}'.format(
            Path(args.output).with_suffix('.engine')))
        with open(Path(args.output).with_suffix('.engine'), 'wb') as f:
            f.write(trt_model.state_dict()['engine'])


if __name__ == '__main__':
    main()
