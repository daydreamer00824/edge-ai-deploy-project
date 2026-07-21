#include "trt_runner.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TrtLogger g_trt_logger;

TrtRunner::TrtRunner(const std::string& engine_path) {
    if (!std::filesystem::exists(engine_path)) {
        throw std::runtime_error(
            "TensorRT engine file does not exist: " + engine_path
        );
    }

    std::vector<char> engine_data = load_engine_file(engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));

    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    engine_.reset(
        runtime_->deserializeCudaEngine(
            engine_data.data(),
            engine_data.size()
        )
    );

    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine.");
    }

    context_.reset(engine_->createExecutionContext());

    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    const int nb_io_tensors = engine_->getNbIOTensors();

    for (int i = 0; i < nb_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);

        if (tensor_name == nullptr) {
            continue;
        }

        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = tensor_name;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = tensor_name;
        }
    }

    if (input_name_.empty()) {
        throw std::runtime_error("Failed to find TensorRT input tensor name.");
    }

    if (output_name_.empty()) {
        throw std::runtime_error("Failed to find TensorRT output tensor name.");
    }

    if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Only FP32 input tensor is supported in this demo.");
    }

    if (engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("Only FP32 output tensor is supported in this demo.");
    }

    check_cuda(
        cudaStreamCreate(&stream_),
        "Failed to create CUDA stream."
    );
}

TrtRunner::~TrtRunner() {
    release_device_buffers();

    if (stream_ != nullptr) {
        const cudaError_t status = cudaStreamDestroy(stream_);

        if (status != cudaSuccess) {
            std::cerr
                << "[WARN] Failed to destroy CUDA stream. CUDA error: "
                << cudaGetErrorString(status)
                << std::endl;
        }

        stream_ = nullptr;
    }
}

std::vector<char> TrtRunner::load_engine_file(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (file_size == 0) {
        throw std::runtime_error("Engine file is empty: " + engine_path);
    }

    std::vector<char> data(file_size);

    file.read(data.data(), static_cast<std::streamsize>(file_size));

    if (!file) {
        throw std::runtime_error("Failed to read engine file: " + engine_path);
    }

    return data;
}

std::size_t TrtRunner::volume(const nvinfer1::Dims& dims) const {
    size_t result = 1;

    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error(
                "Invalid tensor dimension. Tensor shape still contains dynamic or non-positive value."
            );
        }

        result *= static_cast<size_t>(dims.d[i]);
    }

    return result;
}

void TrtRunner::check_cuda(cudaError_t status, const std::string& message) const {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            message + " CUDA error: " + cudaGetErrorString(status)
        );
    }
}

void TrtRunner::release_device_buffers() noexcept {
    if (device_input_ != nullptr) {
        const cudaError_t status = cudaFree(device_input_);

        if (status != cudaSuccess) {
            std::cerr
                << "[WARN] Failed to free input device buffer. CUDA error: "
                << cudaGetErrorString(status)
                << std::endl;
        }
    }

    device_input_ = nullptr;
    device_input_capacity_bytes_ = 0;

    if (device_output_ != nullptr) {
        const cudaError_t status = cudaFree(device_output_);

        if (status != cudaSuccess) {
            std::cerr
                << "[WARN] Failed to free output device buffer. CUDA error: "
                << cudaGetErrorString(status)
                << std::endl;
        }
    }

    device_output_ = nullptr;
    device_output_capacity_bytes_ = 0;
}

void TrtRunner::ensure_device_buffer_capacity(
    std::size_t input_bytes,
    std::size_t output_bytes
) {
    if (input_bytes == 0 || output_bytes == 0) {
        throw std::runtime_error(
            "TensorRT device buffer size must be greater than 0."
        );
    }

    if (input_bytes > device_input_capacity_bytes_) {
        void* new_device_input = nullptr;

        check_cuda(
            cudaMalloc(&new_device_input, input_bytes),
            "Failed to allocate input device buffer."
        );

        if (device_input_ != nullptr) {
            const cudaError_t free_status = cudaFree(device_input_);

            if (free_status != cudaSuccess) {
                cudaFree(new_device_input);

                throw std::runtime_error(
                    std::string(
                        "Failed to free old input device buffer. CUDA error: "
                    ) +
                    cudaGetErrorString(free_status)
                );
            }
        }

        const std::size_t old_capacity =
            device_input_capacity_bytes_;

        device_input_ = new_device_input;
        device_input_capacity_bytes_ = input_bytes;

        std::cout
            << "[TRT] Input device buffer capacity: "
            << old_capacity
            << " -> "
            << device_input_capacity_bytes_
            << " bytes"
            << std::endl;
    }

    if (output_bytes > device_output_capacity_bytes_) {
        void* new_device_output = nullptr;

        check_cuda(
            cudaMalloc(&new_device_output, output_bytes),
            "Failed to allocate output device buffer."
        );

        if (device_output_ != nullptr) {
            const cudaError_t free_status = cudaFree(device_output_);

            if (free_status != cudaSuccess) {
                cudaFree(new_device_output);

                throw std::runtime_error(
                    std::string(
                        "Failed to free old output device buffer. CUDA error: "
                    ) +
                    cudaGetErrorString(free_status)
                );
            }
        }

        const std::size_t old_capacity =
            device_output_capacity_bytes_;

        device_output_ = new_device_output;
        device_output_capacity_bytes_ = output_bytes;

        std::cout
            << "[TRT] Output device buffer capacity: "
            << old_capacity
            << " -> "
            << device_output_capacity_bytes_
            << " bytes"
            << std::endl;
    }
}

std::vector<float> TrtRunner::run(
    const std::vector<float>& input_tensor_values,
    int batch_size,
    int channels,
    int height,
    int width
) {
    if (batch_size <= 0) {
        throw std::runtime_error("batch_size must be greater than 0.");
    }

    if (channels != 3) {
        throw std::runtime_error("Only 3-channel input is supported.");
    }

    nvinfer1::Dims4 input_dims(batch_size, channels, height, width);

    bool set_shape_ok = context_->setInputShape(
        input_name_.c_str(),
        input_dims
    );

    if (!set_shape_ok) {
        throw std::runtime_error(
            "Failed to set TensorRT input shape. "
            "Please check whether batch_size is within min/opt/max shape range."
        );
    }

    nvinfer1::Dims actual_input_dims =
        context_->getTensorShape(input_name_.c_str());

    nvinfer1::Dims actual_output_dims =
        context_->getTensorShape(output_name_.c_str());

    const std::size_t input_count = volume(actual_input_dims);
    const std::size_t output_count = volume(actual_output_dims);

    if (input_tensor_values.size() != input_count) {
        throw std::runtime_error(
            "Input tensor size mismatch. Expected " +
            std::to_string(input_count) +
            ", but got " +
            std::to_string(input_tensor_values.size())
        );
    }

    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);

    ensure_device_buffer_capacity(input_bytes, output_bytes);

    std::vector<float> output_tensor(output_count);

    check_cuda(
        cudaMemcpyAsync(
            device_input_,
            input_tensor_values.data(),
            input_bytes,
            cudaMemcpyHostToDevice,
            stream_
        ),
        "Failed to copy input from host to device."
    );

    bool set_input_ok = context_->setTensorAddress(
        input_name_.c_str(),
        device_input_
    );

    bool set_output_ok = context_->setTensorAddress(
        output_name_.c_str(),
        device_output_
    );

    if (!set_input_ok || !set_output_ok) {
        throw std::runtime_error("Failed to set TensorRT tensor address.");
    }

    bool enqueue_ok = context_->enqueueV3(stream_);

    if (!enqueue_ok) {
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    }

    check_cuda(
        cudaMemcpyAsync(
            output_tensor.data(),
            device_output_,
            output_bytes,
            cudaMemcpyDeviceToHost,
            stream_
        ),
        "Failed to copy output from device to host."
    );

    check_cuda(
        cudaStreamSynchronize(stream_),
        "Failed to synchronize CUDA stream."
    );

    return output_tensor;
}

void TrtRunner::print_engine_info() const {
    std::cout << "========== TensorRT Engine Info ==========" << std::endl;

    const int nb_io_tensors = engine_->getNbIOTensors();

    std::cout << "[TRT] Number of IO tensors: "
              << nb_io_tensors
              << std::endl;

    std::cout << "[TRT] Input name : "
              << input_name_
              << std::endl;

    std::cout << "[TRT] Output name: "
              << output_name_
              << std::endl;

    for (int i = 0; i < nb_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);

        if (tensor_name == nullptr) {
            continue;
        }

        nvinfer1::Dims dims = engine_->getTensorShape(tensor_name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

        std::cout << "[TRT] Tensor " << i
                  << " | name=" << tensor_name
                  << " | mode="
                  << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
                  << " | dims=[";

        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];

            if (j + 1 != dims.nbDims) {
                std::cout << ", ";
            }
        }

        std::cout << "]" << std::endl;
    }
}
