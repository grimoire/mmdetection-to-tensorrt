# MMDet to tensorrt

This project aims to convert the mmdetection model to tensorrt model end2end. 
Focus on object detection for now, instance segmentation will be added in future.

## Requirement

- mmdet>=2.3.0
- https://github.com/grimoire/torch2trt_dynamic
- https://github.com/grimoire/amirstan_plugin


## Installation

```shell
git clone https://github.com/grimoire/mmdetection-to-tensorrt.git
cd mmdetection-to-tensorrt
python setup.py develop
```

## Usage

how to create a tensorrt model from mmdet model (converting might take few minutes）

```python

opt_shape_param=[
    [
        [1,3,800,800],      # min shape
        [1,3,800,1344],     # optimize shape
        [1,3,1344,1344],    # max shape
    ]
]
max_workspace_size=1<<30    # some module need large workspace, add workspace size when OOM.
trt_model = mmdet2trt(cfg_path, weight_path, opt_shape_param=opt_shape_param, fp16_mode=True, max_workspace_size=max_workspace_size)
torch.save(trt_model.state_dict(), save_path)
```

how to use the converted model

```python
trt_model = init_detector(save_path)
num_detections, trt_bbox, trt_score, trt_cls = inference_detector(trt_model, image_path, cfg_path, "cuda:0")
```

read demo/inference.py for more detail

## Support Model/Module

- [x] Faster R-CNN
- [x] Cascade R-CNN
- [x] Double-Head R-CNN
- [x] Group Normalization
- [x] Weight Standardization
- [x] DCN
- [x] SSD
- [x] RetinaNet
- [x] Libra R-CNN
