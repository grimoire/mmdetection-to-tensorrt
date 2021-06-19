import argparse

import torch
from mmdet2trt.apis import init_detector
from mmdet2trt.apis.test import convert_to_mmdet_result
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from torch import nn

import mmcv


class ModelWarper(nn.Module):

    def __init__(self, model, num_classes=80, device='cuda:0'):
        super(ModelWarper, self).__init__()
        self.model = model
        self.device = torch.device(device)
        self.num_classes = num_classes

    def forward(self, **kwargs):
        tensor = kwargs['img'][0].to(self.device)
        scale_factor = kwargs['img_metas'][0].data[0][0]['scale_factor']
        scale_factor = tensor.new_tensor(scale_factor)

        with torch.no_grad():
            result = self.model(tensor)
            result = list(result)
            result[1] = result[1] / scale_factor

        return convert_to_mmdet_result(result, self.num_classes)


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
    trt_model = init_detector(args.trt_model_path)

    # create dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = ModelWarper(trt_model, num_classes=len(dataset.CLASSES))

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
