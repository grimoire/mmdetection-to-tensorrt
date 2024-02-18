from argparse import ArgumentParser

import torch
from mmdet2trt import mmdet2trt
from mmdet2trt.apis import create_wrap_detector
from mmdet.apis import inference_detector
from mmdet.registry import VISUALIZERS

import mmcv


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='mmdet Config file')
    parser.add_argument('checkpoint', help='mmdet Checkpoint file')
    parser.add_argument('save_path', help='tensorrt model save path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--fp16', action='store_true', help='enable fp16 inference')
    args = parser.parse_args()

    cfg_path = args.config

    trt_model = mmdet2trt(
        cfg_path, args.checkpoint, fp16_mode=args.fp16, device=args.device)
    torch.save(trt_model.state_dict(), args.save_path)

    trt_detector = create_wrap_detector(args.save_path, cfg_path, args.device)

    image_path = args.img

    result = inference_detector(trt_detector, image_path)

    # visualize
    visualizer_cfg = dict(type='DetLocalVisualizer', name='visualizer')
    visualizer = VISUALIZERS.build(visualizer_cfg)
    visualizer.dataset_meta = trt_detector.dataset_meta

    image = mmcv.imread(image_path)
    visualizer.add_datasample(
        'result',
        mmcv.imconvert(image, 'bgr', 'rgb'),
        data_sample=result,
        draw_gt=False,
        show=True,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    main()
