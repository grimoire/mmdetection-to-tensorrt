FROM nvcr.io/nvidia/tensorrt:24.01-py3

ARG CUDA=12.1
ARG TORCH_VERSION=2.2.0
ARG TORCHVISION_VERSION=0.17.0

ENV DEBIAN_FRONTEND=noninteractive

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim cmake libsm6 libxext6 libxrender-dev libgl1-mesa-glx git

### create folder
RUN mkdir ~/space &&\
    mkdir /root/.pip

### set pip source
# COPY ./pip.conf /root/.pip

### pytorch
RUN pip3 install torch==${TORCH_VERSION}+cu${CUDA//./} torchvision==${TORCHVISION_VERSION}+cu${CUDA//./} -f https://download.pytorch.org/whl/torch_stable.html


### install mmcv
RUN pip3 install openmim &&\
    mim install mmdet==3.3.0

### git amirstan plugin
RUN git clone --depth=1 https://github.com/grimoire/amirstan_plugin.git /root/space/amirstan_plugin &&\
    cd /root/space/amirstan_plugin &&\
    git submodule update --init --progress --depth=1

### install amirstan plugin
RUN cd /root/space/amirstan_plugin &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j10

ENV AMIRSTAN_LIBRARY_PATH=/root/space/amirstan_plugin/build/lib

### git torch2trt_dynamic
RUN git clone --depth=1 https://github.com/grimoire/torch2trt_dynamic.git /root/space/torch2trt_dynamic

### install torch2trt_dynamic
RUN cd /root/space/torch2trt_dynamic &&\
    python3 setup.py develop

### git mmdetection-to-tensorrt
RUN git clone --depth=1 https://github.com/grimoire/mmdetection-to-tensorrt.git /root/space/mmdetection-to-tensorrt

### install mmdetection-to-tensorrt
RUN cd /root/space/mmdetection-to-tensorrt &&\
    python3 setup.py develop

WORKDIR /root/space

CMD [ "--help" ]
ENTRYPOINT [ "mmdet2trt" ]
