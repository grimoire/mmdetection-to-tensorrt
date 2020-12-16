#ifndef INOUT_TRANSFORM
#define INOUT_TRANSFORM

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "data.h"

/*
according to DCNv2 config file input image must pass following transformation:

dict(
    img_scale=(1333, 800),
    transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
    ])
normalize using mean=[123.675, 116.28, 103.53] and std=[58.395, 57.12, 57.375]
*/

cv::Size g_input_size(1333, 800);
cv::Scalar g_mean_RGB{123.675f, 116.28f, 103.53f};
cv::Scalar g_std_RGB{58.395f, 57.12f, 57.375f};
unsigned g_pad_divisor = 32;


class ImageTransformer {
  public:
    ImageTransformer(cv::Size const& input_layer_size):m_input_layer_size(input_layer_size) {
        int pad_h = (int) (ceil(g_input_size.height / (float) g_pad_divisor)) * g_pad_divisor;
        int pad_w = (int) (ceil(g_input_size.width  / (float) g_pad_divisor)) * g_pad_divisor;
        m_padded_size = cv::Size(pad_w, pad_h);
    }
    
    void transform_input_image(cv::Mat const& input_image, cv::Mat& output_image) const {
        //  1. To RGB, FP32
        cv::cvtColor(input_image, output_image, CV_BGR2RGB);
        output_image.convertTo(output_image, CV_32FC3);
        //  2. Resize to config size
        cv::resize(output_image, output_image, g_input_size);
        //  3. Normalize    
        cv::Mat rgb[3];
        cv::split(output_image, rgb);
        for (size_t i=0; i<3; i++) {
            rgb[i] = (rgb[i] - g_mean_RGB[i]) / g_std_RGB[i];
        }
        cv::merge(rgb, 3, output_image);
        //  4. Pad to divisible size
        cv::copyMakeBorder(output_image,
                       output_image,
                       0,  //  top
                       m_padded_size.height - output_image.rows,  // bottom
                       0,  //  left
                       m_padded_size.width  - output_image.cols,  // right
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0,0,0));
        cv::resize(output_image, output_image, m_input_layer_size);
    }

    Box transform_output_box(Box const& box, cv::Size const& orig_image_size) const {
        float fx = (float) m_input_layer_size.width  / m_padded_size.width;
        float fy = (float) m_input_layer_size.height / m_padded_size.height;
        //Box scaled = scale(box, fx, fy);
        float fx2 = (float) g_input_size.width  / orig_image_size.width;
        float fy2 = (float) g_input_size.height / orig_image_size.height;
        Box scaled = scale(box, fx * fx2, fy * fy2);
        return scaled;
    }

  private:
    Box scale(Box const& box, float fx, float fy) const {
        Box scaled;
        scaled.left   = box.left / fx;
        scaled.top    = box.top / fy;
        scaled.right  = box.right / fx;
        scaled.bottom = box.bottom / fy;
        return scaled;
    }

    cv::Size m_input_layer_size;
    cv::Size m_padded_size;
};

#endif