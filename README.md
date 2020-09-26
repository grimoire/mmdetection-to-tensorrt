# MMDet to tensorrt

This project aims to convert the mmdetection model to tensorrt model end2end.  
Focus on object detection for now, instance segmentation will be added in future.  

support:
- fp16
- int8(experiment)
- batched input
- dynamic input shape
- combination of different modules
- deepstream support

Any advices, bug reports and stars are welcome.

## Requirement

- mmdet>=2.3.0
- https://github.com/grimoire/torch2trt_dynamic
- https://github.com/grimoire/amirstan_plugin

### Important!
set the envoirment variable(in ~/.bashrc):

```shell
export AMIRSTAN_LIBRARY_PATH=<amirstan_plugin_root>/build/lib
```

## Installation

```shell
git clone https://github.com/grimoire/mmdetection-to-tensorrt.git
cd mmdetection-to-tensorrt
python setup.py develop
```

## Usage

how to create a tensorrt model from mmdet model (converting might take few minutes)(Might have some warning when converting.)  
detail can be found in [getting_started.md](./docs/getting_started.md)

### CLI

```bash
mmdet2trt ${CONFIG_PATH} ${CHECKPOINT_PATH} ${OUTPUT_PATH}
```

Run mmdet2trt -h for help on optional arguments.

### Python

```python
opt_shape_param=[
    [
        [1,3,320,320],      # min shape
        [1,3,800,1344],     # optimize shape
        [1,3,1344,1344],    # max shape
    ]
]
max_workspace_size=1<<30    # some module and tactic need large workspace.
trt_model = mmdet2trt(cfg_path, weight_path, opt_shape_param=opt_shape_param, fp16_mode=True, max_workspace_size=max_workspace_size)
torch.save(trt_model.state_dict(), save_path)
```

how to use the converted model

```python
trt_model = init_detector(save_path)
num_detections, trt_bbox, trt_score, trt_cls = inference_detector(trt_model, image_path, cfg_path, "cuda:0")
```

how to save the tensorrt engine

```python
with open(engine_path, mode='wb') as f:
    f.write(model_trt.state_dict()['engine'])
```

note that the bbox inference result did not divided by scale factor, divided by you self if needed.

play demo in demo/inference.py  

[getting_started.md](./docs/getting_started.md) for more detail

## How does it works?
Most other project use pytorch=>ONNX=>tensorRT route, This repo convert pytorch=>tensorRT directly, avoid unnecessary ONNX IR.
read https://github.com/NVIDIA-AI-IOT/torch2trt#how-does-it-work for detail.

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
- [x] FCOS
- [x] Fovea
- [x] CARAFE
- [x] FreeAnchor
- [x] RepPoints
- [x] NAS-FPN
- [x] ATSS
- [x] PAFPN
- [x] FSAF
- [x] GCNet
- [x] Guided Anchoring
- [x] Generalized Attention
- [x] Dynamic R-CNN
- [x] Hybrid Task Cascade (object detection only)
- [x] DetectoRS
- [x] Side-Aware Boundary Localization
- [x] YOLOv3
- [x] PAA
- [ ] CornerNet(WIP)


Tested on:
- torch=1.6.0
- tensorrt=7.1.3.4
- mmdetection=2.4.0
- cuda=10.2
- cudnn=8.0.2.39

If you find any error, please report in the issue.

## Contact

This repo is maintained by [@grimoire](https://github.com/grimoire)  

Discuss group: QQ:1107959378

