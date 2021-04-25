#ifndef INOUT_TRANSFORM
#define INOUT_TRANSFORM

#include <stdexcept>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "data.h"

/*
normalize using mean=[123.675, 116.28, 103.53] and std=[58.395, 57.12, 57.375]
dict(
    img_scale=(1333, 800),
    transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
    ])
*/

//  cv::Size g_input_size(1333, 800);  //  WILL NOT do it - faster and doesn't degrade accuracy
cv::Scalar g_mean_RGB{123.675f, 116.28f, 103.53f};
cv::Scalar g_std_RGB{58.395f, 57.12f, 57.375f};
unsigned g_pad_divisor = 32;

class ImageTransformer {
  public:
    ImageTransformer() {}

    static cv::Size2d transform_input_image(cv::Mat const& input_image, cv::Size const& dst_size,
                                            cv::Mat& output_image) {
        //  0. Check dst size is of mod32
        if (dst_size.width % g_pad_divisor != 0 || dst_size.height % g_pad_divisor != 0)
            throw std::runtime_error(std::string("Error: input layer size must be mod of ") +
                                     std::to_string(g_pad_divisor));
        //  1. Resize before normalization - smaller image takes less time to be transformed
        cv::Size2d scale_factor = resize_keep_aspect_ratio(input_image, dst_size, output_image);
        //  2. To RGB, FP32
        cv::cvtColor(output_image, output_image, CV_BGR2RGB);
        //  3. To FP32 and normalize
        output_image.convertTo(output_image, CV_32FC3);
        cv::Mat rgb[3];
        cv::split(output_image, rgb);
        for (size_t i=0; i<3; i++) {
            rgb[i] = (rgb[i] - g_mean_RGB[i]) / g_std_RGB[i];
        }
        cv::merge(rgb, 3, output_image);
        return scale_factor;
    }

    static Box transform_output_box(Box const& box, cv::Size2d const& scale_factor) {
        Box scaled;
        scaled.left   = box.left   / scale_factor.width;
        scaled.top    = box.top    / scale_factor.height;
        scaled.right  = box.right  / scale_factor.width;
        scaled.bottom = box.bottom / scale_factor.height;
        return scaled;
    }

    static cv::Size2d resize_keep_aspect_ratio(cv::Mat const& input, cv::Size const& dst_size, cv::Mat& output)
    {
        cv::Mat temp_input = input;  // keep header (required if the function was called with same input and output)
        double h = dst_size.width  * (input.rows / (double) input.cols);
        double w = dst_size.height * (input.cols / (double) input.rows);
        if( h <= dst_size.height) w = dst_size.width;
        else                      h = dst_size.height;
        cv::resize(input, output, cv::Size(w, h));
        double fx = (double) output.cols / temp_input.cols;
        double fy = (double) output.rows / temp_input.rows;
        cv::Size2d scale_factors(fx, fy);
        int top  = 0;
        int left = 0;
        int bottom = dst_size.height - output.rows;
        int right  = dst_size.width  - output.cols;
        cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );
        return scale_factors;
    }
};

#endif
