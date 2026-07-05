#include "ort_runner.h"
#include<filesystem>
#include<stdexcept>
#include<array>
#include<iostream>

static void print_shape(const std::vector<int64_t>& shape) {
    std::cout << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];

        if (i + 1 != shape.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]";
}

OrtRunner::OrtRunner(const std::string &model_path):
    env_(ORT_LOGGING_LEVEL_WARNING, "cpp_ort_runner"),
    session_options_(),
    session_(nullptr){
        if(!std::filesystem::exists(model_path)){
            throw std::runtime_error("Model file does not exist: " + model_path);
        }

        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(
            env_,
            model_path.c_str(),
            session_options_
        );

        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name_alloc = session_ -> GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session_ -> GetOutputNameAllocated(0, allocator);

        input_name_ = input_name_alloc.get();
        output_name_ = output_name_alloc.get();

    }

std::vector<float> OrtRunner::run(std::vector<float> &input_tensor_values, const std::vector<int64_t> &input_shape){
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const std::array<const char*, 1> input_names = {input_name_.c_str()};
    const std::array<const char*, 1> output_names = {output_name_.c_str()};

    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1
    );

    if (output_tensors.empty()) {
        throw std::runtime_error("No output tensor returned.");
    }

    Ort::Value& output_tensor = output_tensors[0];

    if (!output_tensor.IsTensor()) {
        throw std::runtime_error("Output is not a tensor.");
    }

    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_count = output_info.GetElementCount();

    float* output_data = output_tensor.GetTensorMutableData<float>();

    return std::vector<float>(output_data, output_data + output_count);

}

void OrtRunner::print_model_info(){
    auto input_type_info = session_ -> GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = input_tensor_info.GetShape();

    auto output_type_info = session_ -> GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_tensor_info.GetShape();

    std::cout << "[INFO] Input name: " << input_name_ << std::endl;
    std::cout << "[INFO] Output name: " << output_name_ << std::endl;

    std::cout << "[INFO] Model input shape: ";
    print_shape(input_shape);
    std::cout << std::endl;

    std::cout << "[INFO] Model output shape: ";
    print_shape(output_shape);
    std::cout << std::endl;
}

const std::string& OrtRunner::input_name() const{
    return input_name_;
}

const std::string& OrtRunner::output_name() const {
    return output_name_;
}