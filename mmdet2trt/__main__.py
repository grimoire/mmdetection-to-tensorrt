import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from .mmdet2trt import mmdet2trt

logger = logging.getLogger('mmdet2trt')


def _get_default_path(config_path):
    config_path = Path(config_path)
    return config_path.with_suffix('.pth').name


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to a mmdet Config file')
    parser.add_argument('checkpoint', help='Path to a mmdet Checkpoint file')
    parser.add_argument(
        '--output',
        default=None,
        help='Path where tensorrt model will be saved')
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
    args = parser.parse_args()
    return args


def _save_model(trt_model, output_path, save_engine):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Saving TRT model to: {}'.format(output_path))
    torch.save(trt_model.state_dict(), output_path)

    if save_engine:
        logger.info('Saving TRT model engine to: {}'.format(
            output_path.with_suffix('.engine')))
        with open(output_path.with_suffix('.engine'), 'wb') as f:
            f.write(trt_model.state_dict()['engine'])


def main():
    args = _parse_args()

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
        enable_mask=args.enable_mask)

    output_path = args.output
    if output_path is None:
        output_path = _get_default_path(args.config)

    _save_model(trt_model, output_path, args.save_engine)


if __name__ == '__main__':
    main()
