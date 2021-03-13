# MMDet to tensorrt

This project aims to convert the mmdetection model to tensorrt model end2end.  
Focus on object detection for now.  
Mask support is **experiment**.  

support:
- fp16
- int8(experiment)
- batched input
- dynamic input shape
- combination of different modules
- deepstream support

Any advices, bug reports and stars are welcome.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Requirement

- mmdet>=2.3.0
- https://github.com/grimoire/torch2trt_dynamic
- https://github.com/grimoire/amirstan_plugin

### Important!

Set the envoirment variable(in ~/.bashrc):

```shell
export AMIRSTAN_LIBRARY_PATH=${amirstan_plugin_root}/build/lib
```

## Installation

### Host
```shell
git clone https://github.com/grimoire/mmdetection-to-tensorrt.git
cd mmdetection-to-tensorrt
python setup.py develop
```

### Docker

Build docker image(Note that TensorRT7.0 might have memory leak, better to upgrade to 7.1+)
```shell
# cuda10.2 tensorrt7.0 pytorch1.6
sudo docker build -t mmdet2trt_docker:v1.0 docker/
```

Run (will show the help for the CLI entrypoint)

```shell
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} mmdet2trt_docker:v1.0
```

Or if you want to open a terminal inside de container:
```shell
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} --entrypoint bash mmdet2trt_docker:v1.0
```

Example conversion:
```shell
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} mmdet2trt_docker:v1.0 ${bind_path}/config.py ${bind_path}/checkpoint.pth ${bind_path}/output.trt
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
- [x] Hybrid Task Cascade
- [x] DetectoRS
- [x] Side-Aware Boundary Localization
- [x] YOLOv3
- [x] PAA
- [ ] CornerNet(WIP)
- [x] Generalized Focal Loss
- [x] Grid RCNN
- [x] VFNet
- [x] GROIE
- [x] Mask R-CNN(experiment)
- [x] Cascade Mask R-CNN(experiment)
- [x] Cascade RPN
- [x] DETR


Tested on:
- torch=1.6.0
- tensorrt=7.1.3.4
- mmdetection=2.10.0
- cuda=10.2
- cudnn=8.0.2.39

If you find any error, please report in the issue.

## FAQ

read [this page](./docs/FAQ.md) if you meet any problem.

## Contact

This repo is maintained by [@grimoire](https://github.com/grimoire)  

Discuss group: QQ:1107959378  
