#include<stdexcept>
#include<string>
#include<vector>
#include<array>
#include<filesystem>
#include<opencv2/opencv.hpp>
#include "image_process.h"

std::vector<float> preprocess(const std::string &image_path, const imageconfig &config){
    if(!std::filesystem::exists(image_path)){
        throw std::runtime_error("Image file does not exist: " + image_path);
    }

    cv::Mat image = cv::imread(image_path);
    if(image.empty()){
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    cv::Mat image_resize;
    cv::resize(image, image_resize, cv::Size(config.target_w, config.target_h));

    cv::Mat image_rgb;
    cv::cvtColor(image_resize, image_rgb, cv::COLOR_BGR2RGB);

    cv::Mat image_float;
    image_rgb.convertTo(image_float, CV_32FC3, 1.0 / 255.0);

    const std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    const std::array<float, 3> stddev = {0.229f, 0.224f, 0.225f};

    const int channels = 3;
    const int tensor_size = channels * config.target_h * config.target_w;

    std::vector<float> input_tensor_values(tensor_size);

    for(int c = 0; c < channels; c++){
        for(int h = 0; h < config.target_h; h++){
            for(int w = 0; w < config.target_w; w++){
                float pixel = image_float.at<cv::Vec3f>(h,w)[c];
                float normalizer = (pixel - mean[c]) / stddev[c];

                int chw_index = c * config.target_h * config.target_w + h * config.target_w + w;
                input_tensor_values[chw_index] = normalizer;
            }
        }
    }

    return input_tensor_values;
}