#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>
#include <cstddef>

struct TrtStageTiming {
    float h2d_ms = 0.0F;
    float inference_ms = 0.0F;
    float d2h_ms = 0.0F;
    float gpu_total_ms = 0.0F;
};

class TrtRunner {
public:
    explicit TrtRunner(const std::string& engine_path);
    ~TrtRunner();

    std::vector<float> run(
        const std::vector<float>& input_tensor_values,
        int batch_size,
        int channels,
        int height,
        int width,
        TrtStageTiming* timing = nullptr
    );

    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    void print_engine_info() const;

private:
    std::vector<char> load_engine_file(const std::string& engine_path);
    std::size_t volume(const nvinfer1::Dims& dims) const;
    void check_cuda(cudaError_t status, const std::string& message) const;
    void ensure_device_buffer_capacity(std::size_t input_bytes, std::size_t output_bytes);
    void release_device_buffers() noexcept;
    void create_timing_events();
    void release_timing_events() noexcept;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    std::string input_name_;
    std::string output_name_;

    cudaStream_t stream_ = nullptr;

    cudaEvent_t event_start_ = nullptr;
    cudaEvent_t event_h2d_end_ = nullptr;
    cudaEvent_t event_inference_end_ = nullptr;
    cudaEvent_t event_d2h_end_ = nullptr;

    void* device_input_ = nullptr;
    void* device_output_ = nullptr;

    std::size_t device_input_capacity_bytes_ = 0;
    std::size_t device_output_capacity_bytes_ = 0;

};
