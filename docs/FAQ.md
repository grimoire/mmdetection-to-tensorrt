# FAQ

- [FAQ](#faq)
  - [Model Convertion](#model-convertion)
    - [**Q: WARNING: Half2 support requested on hardware without native FP16 support, performance will be negatively affected.**](#q-warning-half2-support-requested-on-hardware-without-native-fp16-support-performance-will-be-negatively-affected)
  - [Model Inference](#model-inference)
    - [**Q: Inference take long time on single image.**](#q-inference-take-long-time-on-single-image)
    - [**Q: Memory leak when inference.**](#q-memory-leak-when-inference)
    - [**Q: error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]**](#q-error-parameter-check-failed-at-enginecppsetbindingdimensions1046-condition-profilemindimsdi--dimensionsdi)
    - [**Q: FP16 model if slower than FP32 model**](#q-fp16-model-if-slower-than-fp32-model)

this page provide some frequently ask question and solution.

## Model Convertion

### **Q: WARNING: Half2 support requested on hardware without native FP16 support, performance will be negatively affected.**

The device does not have full rate fp16 model.

## Model Inference

### **Q: Inference take long time on single image.**

The model will do some initial when first inference. Please warmup the model before inference on real data.

### **Q: Memory leak when inference.**

This is a bug of on old version of TensorRT, read [this](https://forums.developer.nvidia.com/t/context-setbindingdimensions-casing-gpu-memory-leak/83423/21) for detail. Please update TensorRT version.

### **Q: error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]**

The input tensor shape if out of the range, please enlarge the `opt_shape_param` when convert the model.
```python
    opt_shape_param=[
        [
            [1,3,224,224],    # min tensor shape
            [1,3,800,1312],  # shape used to do int8 calib
            [1,3,1344,1344], # max tensor shape
        ]
    ]
```

### **Q: FP16 model if slower than FP32 model**

Please check [this](#model-convertion) see if your device have full rate fp16 support.
