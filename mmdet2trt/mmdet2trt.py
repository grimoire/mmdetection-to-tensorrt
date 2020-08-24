import tensorrt as trt
from torch2trt import torch2trt
from mmdet.apis import init_detector

from mmdet2trt.models.builder import build_warper
from mmdet2trt.models.detectors import TwoStageDetectorWarper

import logging
import torch
import time

from argparse import ArgumentParser
from pathlib import Path


logger = logging.getLogger("mmdet2trt")


def mmdet2trt(
    config,
    checkpoint,
    device="cuda:0",
    fp16_mode=False,
    max_workspace_size=0.5e9,
    opt_shape_param=None,
    trt_log_level="INFO",
    return_warp_model=False,
    output_names=["num_detections", "boxes", "scores", "classes"],
):

    device = torch.device(device)

    logger.info("Loading model from config: {}".format(config))
    torch_model = init_detector(config, checkpoint=checkpoint, device=device)
    cfg = torch_model.cfg

    logger.info("Wrapping model")
    warp_model = build_warper(torch_model, TwoStageDetectorWarper)

    if opt_shape_param is None:
        img_scale = cfg.test_pipeline[1]["img_scale"]
        min_scale = min(img_scale)
        max_scale = max(img_scale)+32
        opt_shape_param = [
            [
                [1, 3, min_scale, min_scale],
                [1, 3, img_scale[1], img_scale[0]],
                [1, 3, max_scale, max_scale],
            ]
        ]

    dummy_shape = opt_shape_param[0][1]
    dummy_input = torch.rand(dummy_shape).to(device)

    logger.info("Model warmup")
    with torch.no_grad():
        result = warp_model(dummy_input)

    logger.info("Converting model")
    start = time.time()
    with torch.cuda.device(device), torch.no_grad():
        trt_model = torch2trt(
            warp_model,
            [dummy_input],
            log_level=getattr(trt.Logger, trt_log_level),
            fp16_mode=fp16_mode,
            opt_shape_param=opt_shape_param,
            max_workspace_size=int(max_workspace_size),
            keep_network=False,
            strict_type_constraints=True,
            output_names=output_names,
        )

    duration = time.time() - start
    logger.info("Conversion took {} s".format(duration))

    if return_warp_model:
        return trt_model, warp_model

    return trt_model


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Path to a mmdet Config file")
    parser.add_argument("checkpoint", help="Path to a mmdet Checkpoint file")
    parser.add_argument("output", help="Path where tensorrt model will be saved")
    parser.add_argument("--fp16", type=bool, default=True, help="Enable fp16 inference")
    parser.add_argument(
        "--save-engine",
        type=bool,
        default=True,
        help="Enable saving TensorRT engine. (will be saved at Path(output).with_suffix('.engine')).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used for conversion."
    )
    parser.add_argument(
        "--max-workspace-gb",
        type=float,
        default=0.5,
        help="The maximum `device` (GPU) temporary memory in GB (gigabytes) which TensorRT can use at execution time.",
    )
    parser.add_argument(
        "--min-scale",
        type=int,
        nargs=2,
        default=None,
        help="Minimum input scale in [height, width] order. Only used if all min-scale, opt-scale and max-scale are set.",
    )
    parser.add_argument(
        "--opt-scale",
        type=int,
        nargs=2,
        default=None,
        help="Optimal input scale in [height, width] order. Only used if all min-scale, opt-scale and max-scale are set.",
    )
    parser.add_argument(
        "--max-scale",
        type=int,
        nargs=2,
        default=None,
        help="Maximum input scale in [height, width] order. Only used if all min-scale, opt-scale and max-scale are set.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    parser.add_argument(
        "--trt-log-level",
        default="INFO",
        choices=["VERBOSE", "INFO", "WARNING", "ERROR"],
        help="TensorRT logging level.",
    )
    parser.add_argument(
        "--output-names",
        nargs=4,
        type=str,
        default=["num_detections", "boxes", "scores", "classes"],
        help="Names for the output nodes of the created TRTModule",
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    if all(
        getattr(args, x) is not None for x in ["min_scale", "opt_scale", "max_scale"]
    ):
        opt_shape_param = [args.min_scale, args.opt_scale, args.max_scale]
    else:
        opt_shape_param = None

    trt_model = mmdet2trt(
        args.config,
        args.checkpoint,
        device=args.device,
        fp16_mode=args.fp16,
        max_workspace_size=int(args.max_workspace_gb * 1e9),
        opt_shape_param=opt_shape_param,
        trt_log_level=args.trt_log_level,
        output_names=args.output_names,
    )

    logger.info("Saving TRT model to: {}".format(args.output))
    torch.save(trt_model.state_dict(), args.output)

    if args.save_engine:
        logger.info("Saving TRT model engine to: {}".format(Path(args.output).with_suffix(".engine")))
        with open(Path(args.output).with_suffix(".engine"), "wb") as f:
            f.write(trt_model.state_dict()["engine"])


if __name__ == "__main__":
    main()
