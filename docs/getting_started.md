# Getting Started

This page provide details about mmdet2trt.

- [Getting Started](#getting-started)
  - [dynamic shape/batched input](#dynamic-shapebatched-input)
  - [fp16 support](#fp16-support)
  - [int8 support](#int8-support)
  - [max workspace size](#max-workspace-size)
  - [use in c++](#use-in-c)
  - [deepstream support](#deepstream-support)

## dynamic shape/batched input

`opt_shape_param` is used to set the min/optimize/max shape of the input tensor. For each dimension in it, min<=optimize<=max. For example:

```python
opt_shape_param=[
    [
        [1,3,320,320],
        [2,3,800,1312],
        [4,3,1344,1344],
    ]
]
trt_model = mmdet2trt(  ..., 
                        opt_shape_param=opt_shape_param, # set the opt shape
                        ...)
```

this config will give you input tensor size between (320, 320) to (1344, 1344), max batch_size=4  

**warning:**
Dynamic input shape and batch support might need more memory. Use fixed shape to avoid unnecessary memory usage(min=optimize=max).

## fp16 support

**fp16 mode** can easily accelerate the model, just set the `fp16_mode=True` to enable it.


```python
trt_model = mmdet2trt(  ..., 
                        fp16_mode=True, # enable fp16 mode 
                        ...)
```

## int8 support

**int8 mode** need more configs.  
- set `input8_mode=True`.
- provide calibrate dataset, the `__getitem__()` method of dataset should return a list of tensor with shape (C,H,W), the shape **must** be the same as `opt_shape_param[0][1][1:]` (optimize shape). The tensor should do the same preprocess as the model. There is a default dataset, you can also set your custom one.
- set the calibrate algrithm, support `entropy` and `minmax`.

```python
from mmdet2trt import mmdet2trt, Int8CalibDataset
cfg_path="..."  # mmdetection config path
model_path="..." # mmdetection checkpoint path
image_path_list = [...] # lists of image pathes
opt_shape_param=[
    [
        [...],
        [...],
        [...],
    ]
]
calib_dataset = Int8CalibDataset(image_path_list, cfg_path, opt_shape_param)
trt_model = mmdet2trt(cfg_path, model_path, 
                    opt_shape_param=opt_shape_param, 
                    int8_mode=True,
                    int8_calib_dataset=calib_dataset,
                    int8_calib_alg="entropy")
```

**warning:**
Not all model support int8 mode. If it doesn't works, try fix shape/batch size.

## max workspace size

Some layer need extra gpu memory. Any some optimize tactic also need more space. enlarge `max_workspace_size` may potentially accelerate your model with the cost of more memory.

## use in c++

The converted model is a python warp on engine.  
first, get the serialized engine from trt_model:

```python
with open(engine_path, mode='wb') as f:
    f.write(model_trt.state_dict()['engine'])
```

Link the `${AMIRSTAN_PLUGIN_DIR}/build/lib/libamirstan_plugin.so` in your project. compile and load the engine. enjoy.

**warning:**
might need to invode `initLibAmirstanInferPlugins()` in [amirInferPlugin.h](https://github.com/grimoire/amirstan_plugin/blob/master/include/plugin/amirInferPlugin.h) to load the plugins.  

The engine only contain inference forward. Preprocess(resize, normalize) and postprocess(divid scale factor) should be done in your project. 

## deepstream support

when converting model, set the output names:

```python
trt_model = mmdet2trt(  ..., 
                        output_names=["num_detections", "boxes", "scores", "classes"], # output names 
                        ...)
```

Create engine file:
```python
with open(engine_path, mode='wb') as f:
    f.write(model_trt.state_dict()['engine'])
```

in the deepstream model config file, set some config
```
[property]
...
net-scale-factor=0.0173         # compute from mean, std
offsets=123.675;116.28;103.53   # compute from mean, std
model-engine-file=trt.engine    # the engine file created by mmdet2trt
labelfile-path=labels.txt       # label file
...
```

in the same config file, set the plugin and parse function
```
[property]
...
parse-bbox-func-name=NvDsInferParseMmdet                # parse funtion name(amirstan plugin buildin)
output-blob-names=num_detections;boxes;scores;classes   # output blob names, same as convert output_names
custom-lib-path=libamirstan_plugin.so                   # amirstan plugin lib path
...
```

you might also need to set `group_threshold=0` (not sure why, please tell me if you know.)
```
[class-attrs-all]
...
group-threshold=0
...
```

enjoy the model in deepstream.

**warning:**
I am not so familiar with deepstream, if you find any thing wrong with above, please let me know.