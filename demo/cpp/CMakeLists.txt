cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(TensorRTSample)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

###  Set path to root of amirstan_plugin
set(amirstan_plugin_root "/root/space/amirstan_plugin")
message("Used amirstan_plugin root folder: ${amirstan_plugin_root}")

set(amirstan_includes "${amirstan_plugin_root}/include")
set(amirstan_libs "${amirstan_plugin_root}/build/lib/libamirstan_plugin.so")

# TARGETS --------------------------------------------------------------------------------------------------------------
add_executable(trt_sample main.cpp)
target_include_directories(trt_sample PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} ${amirstan_includes})
target_link_libraries(trt_sample PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} ${amirstan_libs})
