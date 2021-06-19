import logging
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import torch
import tqdm
from mmdet2trt import mmdet2trt
from mmdet2trt.apis import inference_detector, init_detector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mmdet2trt')


def convert_test(cfg_path,
                 checkpoint,
                 trt_model_path,
                 opt_shape_param=None,
                 max_workspace_size=1 << 25,
                 device='cuda:0',
                 fp16=True,
                 enable_mask=False):
    logger.info('creating {} trt model.'.format(cfg_path))
    trt_model = mmdet2trt(
        cfg_path,
        checkpoint,
        opt_shape_param=opt_shape_param,
        max_workspace_size=int(max_workspace_size),
        fp16_mode=fp16,
        device=device,
        enable_mask=enable_mask)
    logger.info('finish, save trt_model in {}'.format(trt_model_path))
    torch.save(trt_model.state_dict(), trt_model_path)
    return trt_model


def inference_test(trt_model,
                   cfg_path,
                   device,
                   test_folder,
                   save_folder,
                   score_thr=0.3):
    file_list = os.listdir(test_folder)

    for file_name in tqdm.tqdm(file_list):
        if not file_name.lower().endswith('.jpg') or file_name.lower(
        ).endswith('.png'):
            continue

        image_path = osp.join(test_folder, file_name)

        result = inference_detector(trt_model, image_path, cfg_path, device)

        num_detections = result[0].item()
        trt_bbox = result[1][0]
        trt_score = result[2][0]
        trt_cls = result[3][0]

        image = cv2.imread(image_path)
        for i in range(num_detections):
            scores = trt_score[i].item()
            classes = int(trt_cls[i].item())
            if scores < score_thr:
                continue
            bbox = tuple(trt_bbox[i])
            bbox = tuple(int(v) for v in bbox)

            color = ((classes >> 2 & 1) * 128 + (classes >> 5 & 1) * 128,
                     (classes >> 1 & 1) * 128 + (classes >> 4 & 1) * 128,
                     (classes >> 0 & 1) * 128 + (classes >> 3 & 1) * 128)
            cv2.rectangle(image, bbox[:2], bbox[2:], color, thickness=5)
        cv2.imwrite(osp.join(save_folder, file_name), image)


TEST_MODE_DICT = {'convert': 1, 'inference': 1 << 1, 'all': 0b11}


def main():
    parser = ArgumentParser()
    parser.add_argument('test_folder', help='folder contain test images')
    parser.add_argument('config', help='mmdet Config file')
    parser.add_argument('checkpoint', help='mmdet Checkpoint file')
    parser.add_argument(
        'save_folder',
        help='tensorrt model and test images results save folder')
    parser.add_argument(
        '--trt_model_path',
        default='',
        help='save and inference model. '
        'default [save_folder]/trt_model.pth')
    parser.add_argument(
        '--opt_shape_param',
        default='[ [ [1,3,800,800], [1,3,800,1344], [1,3,1344,1344] ] ]',
        help='min/opt/max shape of input')
    parser.add_argument(
        '--max_workspace_size', default=1 << 30, help='max workspace size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--fp16', action='store_true', help='enable fp16 inference')
    parser.add_argument(
        '--enable_mask', action='store_true', help='enable mask output')
    parser.add_argument(
        '--test-mode',
        default='all',
        help='what to do in the test',
        choices=['convert', 'inference', 'all'])
    args = parser.parse_args()

    trt_model_path = args.trt_model_path
    if len(trt_model_path) == 0:
        trt_model_path = osp.join(args.save_folder, 'test_model.pth')

    if not osp.exists(args.save_folder):
        os.mkdir(args.save_folder)

    test_mode = TEST_MODE_DICT[args.test_mode]

    if test_mode & TEST_MODE_DICT['convert'] > 0:
        convert_test(
            args.config,
            args.checkpoint,
            trt_model_path,
            opt_shape_param=eval(args.opt_shape_param),
            max_workspace_size=args.max_workspace_size,
            device=args.device,
            fp16=args.fp16)
        trt_model = init_detector(trt_model_path)
    else:
        trt_model = init_detector(trt_model_path)

    if test_mode & TEST_MODE_DICT['inference'] > 0:
        inference_test(
            trt_model,
            args.config,
            args.device,
            args.test_folder,
            args.save_folder,
            score_thr=args.score_thr)


if __name__ == '__main__':
    main()
