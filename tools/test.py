import argparse

from mmdet2trt.apis import create_wrap_detector
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset

import mmcv
from mmcv.parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmdet2trt test (and eval) a model')
    parser.add_argument('config', help='mmdet config file path')
    parser.add_argument('trt_model_path', help='mmdet2trt model path')
    parser.add_argument('--out', help='output result file in pickle format')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # create dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = MMDataParallel(create_wrap_detector(args.trt_model_path, cfg))

    outputs = single_gpu_test(model, data_loader)
    if args.out:
        print('\nwriting results to {}'.format(args.out))
    mmcv.dump(outputs, args.out)
    # outputs = mmcv.load(args.out)
    print('eval bbox:')
    eval_result = dataset.evaluate(outputs, metric='bbox', classwise=False)
    print(eval_result)


if __name__ == '__main__':
    main()
