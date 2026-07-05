#pragma once

#include<memory>
#include<string>
#include<vector>
#include<onnxruntime_cxx_api.h>

class OrtRunner{
    public:
        explicit OrtRunner(const std::string &model_path);
        std::vector<float> run(std::vector<float> &input_tensor_values, const std::vector<int64_t> &input_shape);
        void print_model_info();
        const std::string &input_name() const;
        const std::string &output_name() const;

    private:
        Ort::Env env_;
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::Session> session_;
        std::string input_name_;
        std::string output_name_;
};
