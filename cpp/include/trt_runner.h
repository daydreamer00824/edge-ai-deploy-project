#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

class TrtRunner {
public:
    explicit TrtRunner(const std::string& engine_path);
    ~TrtRunner();

    std::vector<float> run(
        const std::vector<float>& input_tensor_values,
        int batch_size,
        int channels,
        int height,
        int width
    );

    void print_engine_info() const;

private:
    std::vector<char> load_engine_file(const std::string& engine_path);
    size_t volume(const nvinfer1::Dims& dims) const;
    void check_cuda(cudaError_t status, const std::string& message) const;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    std::string input_name_;
    std::string output_name_;

    cudaStream_t stream_ = nullptr;
};