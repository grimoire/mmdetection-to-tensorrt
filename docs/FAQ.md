# FAQ

- [FAQ](#faq)
  - [Model Conversion](#model-conversion)
    - [**Q: WARNING: Half2 support requested on hardware without native FP16 support, performance will be negatively affected.**](#q-warning-half2-support-requested-on-hardware-without-native-fp16-support-performance-will-be-negatively-affected)
  - [Model Inference](#model-inference)
    - [**Q: Inference take a long time on a single image.**](#q-inference-take-a-long-time-on-a-single-image)
    - [**Q: Memory leak when inference.**](#q-memory-leak-when-inference)
    - [**Q: error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]**](#q-error-parameter-check-failed-at-enginecppsetbindingdimensions1046-condition-profilemindimsdi--dimensionsdi)
    - [**Q: FP16 model is slower than FP32 model**](#q-fp16-model-is-slower-than-fp32-model)
    - [**Q: error: [TensorRT] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS**](#q-error-tensorrt-internal-error-assertion-failed-cublasstatus--cublas_status_success)

This page provides some frequently asked questions and their solutions.

## Model Conversion

### **Q: WARNING: Half2 support requested on hardware without native FP16 support, performance will be negatively affected.**

The device does not have full-rate fp16 model.

## Model Inference

### **Q: Inference take a long time on a single image.**

The model will do some initial when first inference. Please warm up the model before inference on real data.

### **Q: Memory leak when inference.**

This is a bug of on old version TensorRT, read [this](https://forums.developer.nvidia.com/t/context-setbindingdimensions-casing-gpu-memory-leak/83423/21) for detail. Please update TensorRT version.

### **Q: error: parameter check failed at: engine.cpp::setBindingDimensions::1046, condition: profileMinDims.d[i] <= dimensions.d[i]**

The input tensor shape is out of the range. Please enlarge the `opt_shape_param` when converting the model.

```python
    opt_shape_param=[
        [
            [1,3,224,224],    # min tensor shape
            [1,3,800,1312],  # shape used to do int8 calib
            [1,3,1344,1344], # max tensor shape
        ]
    ]
```

### **Q: FP16 model is slower than FP32 model**

Please check [this](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) to see if your device has full-rate fp16 support.

### **Q: error: [TensorRT] INTERNAL ERROR: Assertion failed: cublasStatus == CUBLAS_STATUS_SUCCESS**

This is the answer from [Nvidia developer forums](https://forums.developer.nvidia.com/t/matrixmultiply-failed-on-tensorrt-7-2-1/158187/4):

TRT 7.2.1 switches to use cuBLASLt (previously it was cuBLAS). cuBLASLt is the defaulted choice for SM version >= 7.0. However,you may need [CUDA-10.2 Patch 1 (Released Aug 26, 2020)](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal) to resolve some cuBLASLt issues. Another option is to use the new [TacticSource](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#a9e1d81e5a8bfeb38b86e22a66d5f836a) API and disable cuBLASLt tactics if you dont want to upgrade.
