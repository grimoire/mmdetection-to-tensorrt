# C++ Sample to use pre-built TensorRT engine

Best to be used within built docker container (use provided in the project Dockerfile to build the image).

## Requirements
The sample needs additional installation of opencv:

```
apt-get install -y libopencv-dev
```

## Install

Within <mmdetection-to-trt-root/demo/cpp>

```
mkdir build & cd build
cmake -Damirstan_plugin_root=<path-to-amirstan_plugin-root> ..
make -j4
```

## Run the sample

```
build/trt_sample <serialized model filepath (.engine)> <test image(-s) paths>
```

## Where to get the pre-built .engine

The sample is implemented for the TensorRT model converted from mmdetection DCNv2 model (mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py). To use another model one must re-implement inout_transform.h (image transformation) and 'getDetections()' function within main.cpp (detections extraction) according to the model config file.

To obtain converted model and serialized built TensorRT engine run following command within provided docker container (~/space folder):

```
mmdet2trt --save-engine=true \
          --min-scale 1 3 320 320 \
          --opt-scale 1 3 544 960 \
          --max-scale 1 3 800 1344 \
          mmdetection/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py \
          <path-to-mmdetection-original-model> \
          <path-to-converted-model (result.pth)>
```

To run the command above one needs to download `cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_XXXXX-YYYYY.pth` model from official mmdetection git repository.
