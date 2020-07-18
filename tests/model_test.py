from mmdet2trt import mmdet2trt

import torch
import os
import os.path as osp
import cv2
from argparse import ArgumentParser
import tqdm

from mmdet2trt.apis import inference_detector, init_detector


def model_test(test_folder, cfg_path, checkpoint, save_folder, 
                opt_shape_param=None,
                max_workspace_size=1<<25,
                device="cuda:0",
                score_thr=0.3, 
                fp16=True):
    
    if not osp.exists(save_folder):
        os.mkdir(save_folder)
    trt_model_path = osp.join(save_folder, 'trt_model.pth')

    print("creating {} trt model.".format(cfg_path))
    trt_model = mmdet2trt(cfg_path, checkpoint,
                        opt_shape_param=opt_shape_param,
                        max_workspace_size=int(max_workspace_size), 
                        fp16_mode=fp16, 
                        device=device)
    print("finish, save trt_model in {}".format(trt_model_path))
    torch.save(trt_model.state_dict(), trt_model_path)

    trt_model = init_detector(trt_model_path)

    file_list = os.listdir(test_folder)

    for file_name in tqdm.tqdm(file_list):
        if not file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png'):
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
            
            color = ((classes>>2 &1) *128 + (classes>>5 &1) *128,
                    (classes>>1 &1) *128 + (classes>>4 &1) *128,
                    (classes>>0 &1) *128 + (classes>>3 &1) *128)
            cv2.rectangle(image, bbox[:2], bbox[2:], color, thickness=5)
        cv2.imwrite(osp.join(save_folder, file_name), image)
        
        


def main():
    parser = ArgumentParser()
    parser.add_argument('test_folder', help='folder contain test images')
    parser.add_argument('config', help='mmdet Config file')
    parser.add_argument('checkpoint', help='mmdet Checkpoint file')
    parser.add_argument('save_folder', help='tensorrt model and test images results save folder')
    parser.add_argument('--opt_shape_param', default='[ [ [1,3,800,800], [1,3,800,1344], [1,3,1344,1344] ] ]', help='min/opt/max shape of input')
    parser.add_argument('--max_workspace_size', default=1<<30, help='max workspace size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument("--fp16", type=bool, default=True, help="enable fp16 inference")
    args = parser.parse_args()


    model_test(args.test_folder, args.config, args.checkpoint, args.save_folder,
                opt_shape_param = eval(args.opt_shape_param),
                max_workspace_size = args.max_workspace_size,
                device = args.device,
                score_thr = args.score_thr,
                fp16=args.fp16)

if __name__ == '__main__':
    main()