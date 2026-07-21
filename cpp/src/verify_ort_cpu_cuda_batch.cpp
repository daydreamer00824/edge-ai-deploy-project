#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_infer.h"
#include "ort_runner.h"

struct BatchConsistencyResult {
    int batch_size = 0;
    int num_classes = 0;
    int top1_match_count = 0;
    int top5_match_count = 0;
    float max_abs_error = 0.0F;
    double mean_abs_error = 0.0;
};

static std::vector<int> get_topk_indices(
    const std::vector<float>& logits,
    int k
) {
    if (k <= 0 || k > static_cast<int>(logits.size())) {
        throw std::invalid_argument(
            "k must be greater than 0 and not exceed logits size."
        );
    }

    std::vector<int> indices(logits.size());

    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&logits](int lhs, int rhs) {
            return logits[lhs] > logits[rhs];
        }
    );

    indices.resize(k);

    return indices;
}

static BatchConsistencyResult compare_batch_outputs(
    const std::vector<float>& cpu_logits,
    const std::vector<float>& cuda_logits,
    int batch_size
) {
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "batch_size must be greater than 0."
        );
    }

    if (cpu_logits.size() != cuda_logits.size()) {
        throw std::runtime_error(
            "CPU and CUDA output sizes do not match."
        );
    }

    if (cpu_logits.empty()) {
        throw std::runtime_error(
            "Output logits must not be empty."
        );
    }

    if (
        cpu_logits.size() %
        static_cast<size_t>(batch_size) != 0
    ) {
        throw std::runtime_error(
            "Output logits size is not divisible by batch size."
        );
    }

    const int num_classes = static_cast<int>(
        cpu_logits.size() /
        static_cast<size_t>(batch_size)
    );

    BatchConsistencyResult result;
    result.batch_size = batch_size;
    result.num_classes = num_classes;

    double abs_error_sum = 0.0;

    for (size_t i = 0; i < cpu_logits.size(); ++i) {
        const float abs_error = std::abs(
            cpu_logits[i] - cuda_logits[i]
        );

        result.max_abs_error = std::max(
            result.max_abs_error,
            abs_error
        );

        abs_error_sum += static_cast<double>(abs_error);
    }

    result.mean_abs_error =
        abs_error_sum /
        static_cast<double>(cpu_logits.size());

    for (int image_index = 0;
         image_index < batch_size;
         ++image_index) {
        const std::vector<float> cpu_single_logits =
            slice_logits_for_one_image(
                cpu_logits,
                image_index,
                num_classes
            );

        const std::vector<float> cuda_single_logits =
            slice_logits_for_one_image(
                cuda_logits,
                image_index,
                num_classes
            );

        const std::vector<int> cpu_top5 =
            get_topk_indices(cpu_single_logits, 5);

        const std::vector<int> cuda_top5 =
            get_topk_indices(cuda_single_logits, 5);

        if (cpu_top5.front() == cuda_top5.front()) {
            ++result.top1_match_count;
        }

        if (cpu_top5 == cuda_top5) {
            ++result.top5_match_count;
        }
    }

    return result;
}

static void save_results_csv(
    const std::filesystem::path& output_path,
    const std::vector<BatchConsistencyResult>& results
) {
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }

    std::ofstream ofs(output_path);

    if (!ofs.is_open()) {
        throw std::runtime_error(
            "Failed to open output CSV: " +
            output_path.string()
        );
    }

    ofs
        << "batch_size,"
        << "num_classes,"
        << "top1_match_count,"
        << "top5_match_count,"
        << "max_abs_error,"
        << "mean_abs_error\n";

    ofs << std::fixed << std::setprecision(10);

    for (const auto& result : results) {
        ofs
            << result.batch_size << ","
            << result.num_classes << ","
            << result.top1_match_count << ","
            << result.top5_match_count << ","
            << result.max_abs_error << ","
            << result.mean_abs_error << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr
            << "[ERROR] Usage: "
            << argv[0]
            << " <model.onnx> <image_dir> <output.csv>"
            << std::endl;

        return 1;
    }

    try {
        const std::string model_path = argv[1];
        const std::string image_dir = argv[2];
        const std::filesystem::path output_path = argv[3];

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(
                "Model file does not exist: " + model_path
            );
        }

        std::vector<std::string> image_paths =
            collect_image_path(image_dir);

        const std::vector<int> batch_sizes = {1, 8, 12, 16, 32};
        const int max_batch_size = *std::max_element(
            batch_sizes.begin(),
            batch_sizes.end()
        );

        if (
            static_cast<int>(image_paths.size()) <
            max_batch_size
        ) {
            throw std::runtime_error(
                "At least " +
                std::to_string(max_batch_size) +
                " images are required."
            );
        }

        imageconfig config;
        config.target_h = 224;
        config.target_w = 224;

        std::cout
            << "========== ORT CPU CUDA Batch Consistency Verify =========="
            << std::endl;

        std::cout
            << "[INFO] Model path : "
            << model_path
            << std::endl;

        std::cout
            << "[INFO] Image dir  : "
            << image_dir
            << std::endl;

        std::cout
            << "[INFO] Images     : "
            << image_paths.size()
            << std::endl;

        OrtRunner cpu_runner(model_path, "cpu");
        OrtRunner cuda_runner(model_path, "cuda");

        std::vector<BatchConsistencyResult> results;

        for (const int batch_size : batch_sizes) {
            std::cout
                << "\n---------- Batch "
                << batch_size
                << " ----------"
                << std::endl;

            std::vector<float> batch_input =
                build_batch_tensor(
                    image_paths,
                    0,
                    batch_size,
                    config
                );

            const std::vector<int64_t> input_shape = {
                static_cast<int64_t>(batch_size),
                3,
                config.target_h,
                config.target_w
            };

            std::cout
                << "[INFO] Input shape: ["
                << batch_size
                << ", 3, "
                << config.target_h
                << ", "
                << config.target_w
                << "]"
                << std::endl;

            const std::vector<float> cpu_logits =
                cpu_runner.run(
                    batch_input,
                    input_shape
                );

            const std::vector<float> cuda_logits =
                cuda_runner.run(
                    batch_input,
                    input_shape
                );

            const BatchConsistencyResult result =
                compare_batch_outputs(
                    cpu_logits,
                    cuda_logits,
                    batch_size
                );

            results.push_back(result);

            std::cout
                << "[RESULT] Output shape       : ["
                << batch_size
                << ", "
                << result.num_classes
                << "]"
                << std::endl;

            std::cout
                << "[RESULT] Top-1 matches      : "
                << result.top1_match_count
                << "/"
                << batch_size
                << std::endl;

            std::cout
                << "[RESULT] Top-5 matches      : "
                << result.top5_match_count
                << "/"
                << batch_size
                << std::endl;

            std::cout
                << std::fixed
                << std::setprecision(10)
                << "[RESULT] Max absolute error : "
                << result.max_abs_error
                << std::endl;

            std::cout
                << "[RESULT] Mean absolute error: "
                << result.mean_abs_error
                << std::endl;
        }

        save_results_csv(output_path, results);

        std::cout
            << "\n[INFO] Results saved: "
            << output_path.string()
            << std::endl;

        std::cout
            << "[INFO] Batch consistency verification finished."
            << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr
            << "[ERROR] ONNX Runtime exception: "
            << e.what()
            << std::endl;

        return 1;
    }
    catch (const std::exception& e) {
        std::cerr
            << "[ERROR] "
            << e.what()
            << std::endl;

        return 1;
    }

    return 0;
}
