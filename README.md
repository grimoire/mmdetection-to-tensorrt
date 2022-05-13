# MMDet to TensorRT

## News

OpenMMLab has release [MMDeploy](https://github.com/open-mmlab/mmdeploy) which support more inference engine and repos. PRs and advices are welcome !

## Introduction

This project aims to convert the mmdetection model to TensorRT model end2end.
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

- install mmdetection:

    ```bash
    # mim is so cool!
    pip install openmim
    mim install mmdet==2.14.0
    ```

- install [torch2trt_dynamic](https://github.com/grimoire/torch2trt_dynamic):

    ```bash
    git clone https://github.com/grimoire/torch2trt_dynamic.git torch2trt_dynamic
    cd torch2trt_dynamic
    python setup.py develop
    ```

- install [amirstan_plugin](https://github.com/grimoire/amirstan_plugin):
  - Install tensorrt: [TensorRT](https://developer.nvidia.com/tensorrt)
  - clone repo and build plugin

    ```bash
    git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git
    cd amirstan_plugin
    git submodule update --init --progress --depth=1
    mkdir build
    cd build
    cmake -DTENSORRT_DIR=${TENSORRT_DIR} ..
    make -j10
    ```

  - **DON'T FORGET** setting the envoirment variable(in `~/.bashrc`):

    ```bash
    export AMIRSTAN_LIBRARY_PATH=${amirstan_plugin_root}/build/lib
    ```

## Installation

### Host

```bash
git clone https://github.com/grimoire/mmdetection-to-tensorrt.git
cd mmdetection-to-tensorrt
python setup.py develop
```

### Docker

Build docker image

```bash
# cuda11.1 TensorRT7.2.2 pytorch1.8 cuda11.1
sudo docker build -t mmdet2trt_docker:v1.0 docker/
```

You can also specify CUDA, Pytorch and Torchvision versions with docker build args by:

```bash
# cuda11.1 tensorrt7.2.2 pytorch1.6 cuda10.2
sudo docker build -t mmdet2trt_docker:v1.0 --build-arg TORCH_VERSION=1.6.0 --build-arg TORCHVISION_VERSION=0.7.0 --build-arg CUDA=10.2 --docker/
```

Run (will show the help for the CLI entrypoint)

```bash
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} mmdet2trt_docker:v1.0
```

Or if you want to open a terminal inside de container:

```bash
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} --entrypoint bash mmdet2trt_docker:v1.0
```

Example conversion:

```bash
sudo docker run --gpus all -it --rm -v ${your_data_path}:${bind_path} mmdet2trt_docker:v1.0 ${bind_path}/config.py ${bind_path}/checkpoint.pth ${bind_path}/output.trt
```

## Usage

how to create a TensorRT model from mmdet model (converting might take few minutes)(Might have some warning when converting.)
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

# save converted model
torch.save(trt_model.state_dict(), save_model_path)

# save engine if you want to use it in c++ api
with open(save_engine_path, mode='wb') as f:
    f.write(trt_model.state_dict()['engine'])
```

**Note**:
- The input of the engine is the tensor **after preprocess**.
- The output of the engine is `num_dets, bboxes, scores, class_ids`. if you enable the `enable_mask` flag, there will be another output `mask`.
- The bboxes output of the engine did not divided by `scale factor`.

how to use the converted model

```python
from mmdet.apis import inference_detector
from mmdet2trt.apis import create_wrap_detector

# create wrap detector
trt_detector = create_wrap_detector(trt_model, cfg_path, device_id)

# result share same format as mmdetection
result = inference_detector(trt_detector, image_path)

# visualize
trt_detector.show_result(
    image_path,
    result,
    score_thr=score_thr,
    win_name='mmdet2trt',
    show=True)
```

Try demo in `demo/inference.py`, or `demo/cpp` if you want to do inference with c++ api.

Read [getting_started.md](./docs/getting_started.md) for more details.

## How does it works?

Most other project use pytorch=>ONNX=>tensorRT route, This repo convert pytorch=>tensorRT directly, avoid unnecessary ONNX IR.
Read [how-does-it-work](https://github.com/NVIDIA-AI-IOT/torch2trt#how-does-it-work) for detail.

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
- [x] YOLOX

Tested on:

- torch=1.8.1
- tensorrt=8.0.1.6
- mmdetection=2.18.0
- cuda=11.1

If you find any error, please report it in the issue.

## FAQ

read [this page](./docs/FAQ.md) if you meet any problem.

## Contact

This repo is maintained by [@grimoire](https://github.com/grimoire)

And send your resume to my e-mail if you want to join @OpenMMLab. Please read the JD for detail: [link](https://mp.weixin.qq.com/s/CzrOqITFZX-T_Kcor0hs2g)
