#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <NvUffParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <plugin/amirInferPlugin.h>
#include "data.h"
#include "inout_transform.h"


//  Some required utilities
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
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

size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) size *= dims.d[i];
    return size;
}

//-----------------------------------------------------------------------------------------------------

// Load the model function
void parseUffModel(const std::string& model_path, int batch_size, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context)
{
    TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetworkV2(0U)};
    TRTUniquePtr< nvuffparser::IUffParser > parser{nvuffparser::createUffParser()};
    parser->registerInput("input", nvinfer1::Dims3(3,128,128), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput("final_training_ops/package_layer");
    // parse uff
    if (!parser->parse(model_path.c_str(), *network, nvinfer1::DataType::kFLOAT)) {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        return;
    }

    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // set batch size
    builder->setMaxBatchSize(batch_size);
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}

// read network output
template < class T >
unsigned getOutput(T* gpu_output, nvinfer1::Dims const& dims, int batch_size, std::vector< T >& cpu_output)
{
    // copy results from GPU to CPU
    size_t output_size = getSizeByDim(dims) * batch_size;
    cpu_output.resize(output_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(T), cudaMemcpyDeviceToHost);
    return cpu_output.size();
}

// read network outputs
unsigned getDetections(std::vector< void* >const& gpu_outputs, std::vector<nvinfer1::Dims>const& dims,
    int batch_size, ImageTransformer const& image_transformer, cv::Size const& orig_image_size,
    std::vector< Detection >& detections)
{
    assert(dims.size() == 4);
    std::vector<int> num_detections_vector;
    getOutput((int *) gpu_outputs[0], dims[0], batch_size, num_detections_vector);
    assert(num_detections_vector.size() == 1);
    int num_detections = num_detections_vector[0];
    std::cout << "Got num detections: " << num_detections << std::endl;
    std::vector<float> boxes, scores, labels;
    for (unsigned int i=1; i<dims.size(); i++) {
        switch(i) {
            case 1: getOutput((float *) gpu_outputs[i], dims[i], batch_size, boxes);  break;
            case 2: getOutput((float *) gpu_outputs[i], dims[i], batch_size, scores); break;
            case 3: getOutput((float *) gpu_outputs[i], dims[i], batch_size, labels); break;
        }
    }

    for (unsigned int i=0; i<num_detections; i++) {
        float left   = boxes[4*i];
        float top    = boxes[4*i+1];
        float right  = boxes[4*i+2];
        float bottom = boxes[4*i+3];
        float score  = scores[i];
        float label  = labels[i];
        Box scaled_back = image_transformer.transform_output_box(Box(left, top, right, bottom), orig_image_size);
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
bool processInput(std::string const& image_path, void* gpu_input, nvinfer1::Dims const& dims,
    ImageTransformer const& image_transformer, cv::Size& orig_image_size)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Input image " << image_path << " load failed " << std::endl;
        return false;
    }
    orig_image_size = cv::Size(image.cols, image.rows);
    auto channels     = dims.d[1];
    auto input_height = dims.d[2];
    auto input_width  = dims.d[3];
    auto input_size   = cv::Size(input_width, input_height);
    cv::Mat transformed;
    image_transformer.transform_input_image(image, transformed);
    // split RGB to vector of 2D tensors and put to 1D buffer
    size_t input_bindin_size = channels * input_width * input_height * sizeof(float);  //  8 or 16 bytes per elem?
    float* cpu_input = (float*) malloc(input_bindin_size);
    std::vector< cv::Mat > rgb;
    for (size_t i = 0; i < channels; ++i) {
        rgb.emplace_back(cv::Mat(input_size, CV_32FC1, cpu_input + i * input_width * input_height));
    }
    cv::split(transformed, rgb);
    // copy input image into GPU device
    cudaMemcpy(gpu_input, cpu_input, input_bindin_size, cudaMemcpyHostToDevice);
    free(cpu_input);
    return true;
}


int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <serialized model engine> <image(-s)>" << std::endl;
        return -1;
    }
    std::string model_path(argv[1]);
    int batch_size = 1;
    std::vector<std::string> filepaths;
    for (unsigned int i=2; i<argc; i++) filepaths.push_back(std::string(argv[i]));
    
    //  Parse the model and initialize the engine and the context
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
		return 1;
	}
	gieModelStream.read((char*)modelData, modelSize);
    //  create TRT engine in cuda device from model data
    initLibAmirstanInferPlugins();
    TRTUniquePtr< nvinfer1::IRuntime >    runtime{nvinfer1::createInferRuntime(gLogger)};
    TRTUniquePtr< nvinfer1::ICudaEngine > engine{runtime->deserializeCudaEngine(modelData, modelSize, nullptr)};
    free(modelData);
    if( !engine ) {
        std::cout << "failed to create CUDA engine" << std::endl;
        return 1;
    }
    TRTUniquePtr< nvinfer1::IExecutionContext > context{engine->createExecutionContext()};
    if( !context ) {
		std::cout << "failed to create execution context" << std::endl;
		return 1;
	}

    //  Specify (fix) model input layer size
    //  fill free to try different input layer size in range used while model convertion
    cv::Size input_layer_size(960, 544);
    context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, input_layer_size.height, input_layer_size.width));
    
    //  Create image transformer
    ImageTransformer image_transformer(input_layer_size);
    std::cout << "The model has been successfully parsed. Input dims set." << std::endl;

    //  Allocate input/output memory
    std::vector< nvinfer1::Dims > input_dims;   // we expect only one input
    std::vector< nvinfer1::Dims > output_dims;  // and one output
    std::vector< void* > inout_buffer_pointers; // buffers for input and output data
    std::vector< void* > input_buffer_pointers, output_buffer_pointers;
    for (size_t i = 0; i < engine->getNbBindings(); ++i) {
        auto binding_size = getSizeByDim(context->getBindingDimensions(i)) * batch_size * sizeof(float);
        void* buffer_pointer;
        cudaMalloc(&buffer_pointer, binding_size);
        inout_buffer_pointers.push_back(buffer_pointer);
        std::cout << "Allocated " << binding_size << " bytes in GPU, i = " << i << std::endl;
        if (engine->bindingIsInput(i)) {
            input_buffer_pointers.push_back(buffer_pointer);
            input_dims.emplace_back(context->getBindingDimensions(i));
            std::cout << "Allocated " << binding_size << " bytes in GPU for input" << std::endl;
        }
        else {
            output_buffer_pointers.push_back(buffer_pointer);
            output_dims.emplace_back(context->getBindingDimensions(i));
            std::cout << "Allocated " << binding_size << " bytes in GPU for output" << std::endl;
        }
    }
    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "Expect at least one input and one output for network" << std::endl;
        return -1;
    }
    if (input_buffer_pointers.size() > 1 || input_buffer_pointers.size() < 1) {
        std::cerr << "Expect exactly one input for network" << std::endl;
        return -1;
    }

    //  Run the test
    float total1 = 0;
    float total2 = 0;
    unsigned miss = 5;
    for (unsigned i=0; i<filepaths.size(); i++) {
        const auto t_start1 = std::chrono::high_resolution_clock::now();
        // preprocess input data
        cv::Size orig_image_size;
        if (!processInput(filepaths[i], input_buffer_pointers[0], input_dims[0], image_transformer, orig_image_size)) {
            std::cerr << "Error while pre-processing image and moving it to GPU device" << std::endl;
            return -1;
        }
        const auto t_start2 = std::chrono::high_resolution_clock::now();
        // run inference
        context->enqueueV2(inout_buffer_pointers.data(), 0, nullptr);
        // extract results
        std::vector< Detection > detections;       
        getDetections(output_buffer_pointers, output_dims, batch_size, image_transformer, orig_image_size, detections);
        const auto t_end = std::chrono::high_resolution_clock::now();
        const float ms1 = std::chrono::duration<float, std::milli>(t_end - t_start1).count();
        const float ms2 = std::chrono::duration<float, std::milli>(t_end - t_start2).count();
        std::cout << "image prepare + inference took: " << ms1 << " ms; only inference: " << ms2 << " ms" << std::endl;
        if (i >= miss) {
            total1 += ms1;
            total2 += ms2;
        }
        printOutput(detections);
    }
    // release cuda memory
    for (void* buf : inout_buffer_pointers) cudaFree(buf);
    
    total1 /= (filepaths.size() - miss);
    total2 /= (filepaths.size() - miss);
    std::cout << "Average over " << (filepaths.size() - miss) << 
        " runs is " << total1 << " ms (pre-processing + inference); (" << total2 << " ms - inference only)" << std::endl;

    return 0;
}
