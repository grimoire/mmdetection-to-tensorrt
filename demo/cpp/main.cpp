#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <plugin/amirInferPlugin.h>
#include "data.h"
#include "inout_transform.h"

#include "NvInferVersion.h"

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept
#else
#define PLUGIN_NOEXCEPT
#endif

//  Some required utilities
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) PLUGIN_NOEXCEPT override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// Create definition TRTUniquePtr for unique pointer of TensorRT’s classes
// destroy TensorRT objects if something goes wrong
struct TRTDestroy {
    template< class T >
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

size_t get_size_by_dim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
    return size;
}

//  Functions to allocate/release input/output memory
bool allocate_cuda_inout_memory(TRTUniquePtr< nvinfer1::ICudaEngine >& engine,
                                TRTUniquePtr< nvinfer1::IExecutionContext >& context,
                                int batch_size,
                                std::vector< void* >& inout_buffer_pointers,
                                std::vector< void* >& input_buffer_pointers,
                                std::vector< void* >& output_buffer_pointers,
                                std::vector< nvinfer1::Dims >& input_dims,
                                std::vector< nvinfer1::Dims >& output_dims,
                                float*& cpu_transfer_buffer) {
    for (size_t i = 0; i < engine->getNbBindings(); ++i) {
        auto binding_size = get_size_by_dim(context->getBindingDimensions(i)) * batch_size * sizeof(float);
        void* buffer_pointer;
        cudaMalloc(&buffer_pointer, binding_size);
        inout_buffer_pointers.push_back(buffer_pointer);
        std::cout << "Allocated " << binding_size << " bytes in GPU, i = " << i << std::endl;
        if (engine->bindingIsInput(i)) {
            input_buffer_pointers.push_back(buffer_pointer);
            input_dims.emplace_back(context->getBindingDimensions(i));
            if (cudaSuccess != cudaMallocHost((void**)&cpu_transfer_buffer, binding_size)) {
                std::cerr << "Error: allocating pinned host memory!" << std::endl;
                return false;
            }
            std::cout << "Allocated " << binding_size << " bytes in GPU for input and in CPU for transfer" << std::endl;
        }
        else {
            output_buffer_pointers.push_back(buffer_pointer);
            output_dims.emplace_back(context->getBindingDimensions(i));
            std::cout << "Allocated " << binding_size << " bytes in GPU for output" << std::endl;
        }
    }
    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "Expect at least one input and one output for network" << std::endl;
        return false;
    }
    if (input_buffer_pointers.size() > 1 || input_buffer_pointers.size() < 1) {
        std::cerr << "Expect exactly one input for network" << std::endl;
        return false;
    }
    return true;
}

void deallocate_cuda_inout_memory(std::vector< void* >& inout_buffer_pointers,
                                  std::vector< void* >& input_buffer_pointers,
                                  std::vector< void* >& output_buffer_pointers,
                                  std::vector< nvinfer1::Dims >& input_dims,
                                  std::vector< nvinfer1::Dims >& output_dims,
                                  float*& cpu_transfer_buffer) {
    for (void* buf : inout_buffer_pointers) cudaFree(buf);
    inout_buffer_pointers.clear();
    input_buffer_pointers.clear();
    output_buffer_pointers.clear();
    input_dims.clear();
    output_dims.clear();
    cudaFreeHost(cpu_transfer_buffer);
}

// load the model engine
bool load_engine(std::string const& model_path, TRTUniquePtr< nvinfer1::ICudaEngine >& engine,
                 TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
    //  Load model data via stringstream to get model size
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    std::ifstream model_file( model_path );
    //  read the model data to the stringstream
    gieModelStream << model_file.rdbuf();
	model_file.close();
    //  find model size
    gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);
    //  load the model to the buffer
	void* modelData = malloc(modelSize);
	if( !modelData ) {
		std::cout << "failed to allocate " << modelSize << " bytes to deserialize model" << std::endl;
		return false;
	}
	gieModelStream.read((char*)modelData, modelSize);
    //  create TRT engine in cuda device from model data
    initLibAmirstanInferPlugins();
    TRTUniquePtr< nvinfer1::IRuntime >    runtime{nvinfer1::createInferRuntime(gLogger)};
    engine.reset(runtime->deserializeCudaEngine(modelData, modelSize, nullptr));
    free(modelData);
    if( !engine ) {
        std::cout << "failed to create CUDA engine" << std::endl;
        return false;
    }
    context.reset(engine->createExecutionContext());
    if( !context ) {
		std::cout << "failed to create execution context" << std::endl;
		return false;
	}
    return true;
}


// read network output
template < class T >
unsigned read_output(T* gpu_output, nvinfer1::Dims const& dims, int batch_size, std::vector< T >& cpu_output)
{
    // copy results from GPU to CPU
    size_t output_size = get_size_by_dim(dims) * batch_size;
    cpu_output.resize(output_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(T), cudaMemcpyDeviceToHost);
    return cpu_output.size();
}

// read network outputs
unsigned parse_detections(std::vector< void* >const& gpu_outputs, std::vector<nvinfer1::Dims>const& dims,
    int batch_size, cv::Size2d const& scale_factor, std::vector< Detection >& detections)
{
    assert(dims.size() == 4);
    std::vector<int> num_detections_vector;
    read_output((int *) gpu_outputs[0], dims[0], batch_size, num_detections_vector);
    assert(num_detections_vector.size() == 1);
    int num_detections = num_detections_vector[0];
    std::cout << "Got num detections: " << num_detections << std::endl;
    std::vector<float> boxes, scores, labels;
    for (unsigned int i=1; i<dims.size(); i++) {
        switch(i) {
            case 1: read_output((float *) gpu_outputs[i], dims[i], batch_size, boxes);  break;
            case 2: read_output((float *) gpu_outputs[i], dims[i], batch_size, scores); break;
            case 3: read_output((float *) gpu_outputs[i], dims[i], batch_size, labels); break;
        }
    }

    for (unsigned int i=0; i<num_detections; i++) {
        float left   = boxes[4*i];
        float top    = boxes[4*i+1];
        float right  = boxes[4*i+2];
        float bottom = boxes[4*i+3];
        float score  = scores[i];
        float label  = labels[i];
        Box scaled_back = ImageTransformer::transform_output_box(Box(left, top, right, bottom), scale_factor);
        detections.push_back(Detection(scaled_back, score, label));
    }

    return detections.size();
}

void printOutput(std::vector< Detection >const& detections)
{
    std::cout << "Detections amount: " << detections.size() << std::endl;

    for (unsigned i=0; i<detections.size(); i++) {
        Detection const& d = detections[i];
        std::cout << "Detection " << i << " " << d.score << " " << d.label << " (" <<
            d.box.left << ", " << d.box.top << ", " << d.box.right << ", " << d.box.bottom << ")"  << std::endl;
    }
    std::cout << "Detections amount: " << detections.size() << std::endl;
}

// pre-process the image and load it to tensor buffer
cv::Size2d processInput(std::string const& image_path, void* gpu_input, float*& cpu_transfer_buffer,
                        nvinfer1::Dims const& dims)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
        throw std::runtime_error(std::string("Error: cannot load image ") + image_path);

    auto channels     = dims.d[1];
    auto input_height = dims.d[2];
    auto input_width  = dims.d[3];
    auto input_size   = cv::Size(input_width, input_height);
    cv::Mat transformed;
    cv::Size2d scale_factor = ImageTransformer::transform_input_image(image, input_size, transformed);

    // split RGB to vector of 2D tensors and put to 1D buffer
    size_t input_bindin_size = channels * input_width * input_height * sizeof(float);
    std::vector< cv::Mat > rgb;
    for (size_t i = 0; i < channels; ++i) {
        rgb.emplace_back(cv::Mat(input_size, CV_32FC1, cpu_transfer_buffer + i * input_width * input_height));
    }
    cv::split(transformed, rgb);
    // copy input image into GPU device
    cudaMemcpy(gpu_input, cpu_transfer_buffer, input_bindin_size, cudaMemcpyHostToDevice);
    return scale_factor;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] <<
            " <serialized model engine> <results path> <image(-s)>" << std::endl;
        return -1;
    }
    std::string model_path(argv[1]);
    int batch_size = 1;
    std::string results_path = std::string(argv[2]);
    std::vector<std::string> filepaths;
    for (unsigned int i=3; i<argc; i++) filepaths.push_back(std::string(argv[i]));

    //  Parse the model and initialize the engine and the context
    TRTUniquePtr< nvinfer1::ICudaEngine > engine{nullptr};
    TRTUniquePtr< nvinfer1::IExecutionContext > context{nullptr};
    if (load_engine(model_path, engine, context)) {
        std::cout << "The model has been successfully parsed. Input dims set." << std::endl;
    }
    else return 1;

    //  Specify (fix) model input layer size
    cv::Size input_layer_size(960, 544);
    unsigned channels = 3;

    //  Allocate input/output memory
    std::vector< nvinfer1::Dims > input_dims; // we expect only one input
    std::vector< nvinfer1::Dims > output_dims; // and one output
    std::vector< void* > inout_buffer_pointers; // buffers for input and output data
    std::vector< void* > input_buffer_pointers, output_buffer_pointers;
    float* cpu_transfer_buffer;
    // set static dims
    context->setBindingDimensions(0, nvinfer1::Dims4(batch_size, channels,
                                  input_layer_size.height, input_layer_size.width));
    if (!allocate_cuda_inout_memory(engine, context, batch_size, inout_buffer_pointers,
                                        input_buffer_pointers, output_buffer_pointers,
                                        input_dims, output_dims, cpu_transfer_buffer))  return -1;

    //  Run the test
    float total1 = 0;
    float total2 = 0;
    unsigned miss = filepaths.size() > 5 ? 5 : 0;
    const auto t_start_total = std::chrono::high_resolution_clock::now();
    for (unsigned i=0; i<filepaths.size(); i++) {
        const auto t_start1 = std::chrono::high_resolution_clock::now();
        // preprocess input data
        cv::Size2d scale_factor = processInput(filepaths[i], input_buffer_pointers[0],
            cpu_transfer_buffer, input_dims[0]);
        const auto t_start2 = std::chrono::high_resolution_clock::now();
        // run inference
        context->enqueueV2(inout_buffer_pointers.data(), 0, nullptr);
        // extract results
        std::vector< Detection > detections;
        parse_detections(output_buffer_pointers, output_dims, batch_size, scale_factor, detections);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float ms1 = std::chrono::duration<float, std::milli>(t_end - t_start1).count();
        const float ms2 = std::chrono::duration<float, std::milli>(t_end - t_start2).count();
        std::cout << "image prepare + inference took: " << ms1 << " ms;" << std::endl;
        std::cout << "inference only: " << ms2 << " ms" << std::endl;
        if (i >= miss) {
            total1 += ms1;
            total2 += ms2;
        }
        printOutput(detections);
    }
    const auto t_end_total = std::chrono::high_resolution_clock::now();
    float ms_total = std::chrono::duration<float, std::milli>(t_end_total - t_start_total).count();
    total1 /= (filepaths.size() - miss);
    total2 /= (filepaths.size() - miss);
    ms_total /= filepaths.size();
    std::cout << "Average over " << (filepaths.size() - miss) <<
        " runs is " << total1 << " ms; (" << total2 << " ms - inference only)" << std::endl;
    std::cout << "Alternative average timing over all " << filepaths.size() << " processed files: " <<
        ms_total << " ms" << std::endl;
    return 0;
}
