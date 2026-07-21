#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch_infer.h"
#include "trt_runner.h"

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

    std::iota(
        indices.begin(),
        indices.end(),
        0
    );

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

static void print_topk(
    const std::vector<int>& topk
) {
    std::cout << "[";

    for (std::size_t i = 0; i < topk.size(); ++i) {
        std::cout << topk[i];

        if (i + 1 != topk.size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr
            << "[ERROR] Usage: "
            << argv[0]
            << " <model.engine> <image_dir>"
            << std::endl;

        return 1;
    }

    try {
        const std::string engine_path = argv[1];
        const std::string image_dir = argv[2];

        if (!std::filesystem::exists(engine_path)) {
            throw std::runtime_error(
                "TensorRT engine file does not exist: " +
                engine_path
            );
        }

        std::vector<std::string> image_paths =
            collect_image_path(image_dir);

        const std::vector<int> batch_sequence = {
            32,
            1,
            8,
            32
        };

        const int max_batch_size = *std::max_element(
            batch_sequence.begin(),
            batch_sequence.end()
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

        constexpr int channels = 3;
        constexpr int expected_num_classes = 1000;

        std::cout
            << "========== TensorRT Buffer Reuse Verify =========="
            << std::endl;

        std::cout
            << "[INFO] Engine path    : "
            << engine_path
            << std::endl;

        std::cout
            << "[INFO] Image dir      : "
            << image_dir
            << std::endl;

        std::cout
            << "[INFO] Images found   : "
            << image_paths.size()
            << std::endl;

        std::cout
            << "[INFO] Batch sequence : 32 -> 1 -> 8 -> 32"
            << std::endl;

        TrtRunner runner(engine_path);

        std::vector<int> baseline_top5;

        for (std::size_t step = 0;
             step < batch_sequence.size();
             ++step) {
            const int batch_size = batch_sequence[step];

            std::cout
                << "\n---------- Step "
                << step + 1
                << " | Batch "
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

            const std::vector<float> batch_logits =
                runner.run(
                    batch_input,
                    batch_size,
                    channels,
                    config.target_h,
                    config.target_w
                );

            const std::size_t expected_output_count =
                static_cast<std::size_t>(batch_size) *
                static_cast<std::size_t>(
                    expected_num_classes
                );

            if (
                batch_logits.size() !=
                expected_output_count
            ) {
                throw std::runtime_error(
                    "Output size mismatch for batch " +
                    std::to_string(batch_size) +
                    ". Expected " +
                    std::to_string(expected_output_count) +
                    ", but got " +
                    std::to_string(batch_logits.size()) +
                    "."
                );
            }

            const bool all_finite = std::all_of(
                batch_logits.begin(),
                batch_logits.end(),
                [](float value) {
                    return std::isfinite(value);
                }
            );

            if (!all_finite) {
                throw std::runtime_error(
                    "Output contains NaN or Inf for batch " +
                    std::to_string(batch_size) +
                    "."
                );
            }

            const std::vector<float> first_image_logits =
                slice_logits_for_one_image(
                    batch_logits,
                    0,
                    expected_num_classes
                );

            const std::vector<int> current_top5 =
                get_topk_indices(
                    first_image_logits,
                    5
                );

            std::cout
                << "[RESULT] Output shape: ["
                << batch_size
                << ", "
                << expected_num_classes
                << "]"
                << std::endl;

            std::cout
                << "[RESULT] All values finite: true"
                << std::endl;

            std::cout
                << "[RESULT] First image Top-5: ";

            print_topk(current_top5);

            std::cout << std::endl;

            if (step == 0) {
                baseline_top5 = current_top5;

                std::cout
                    << "[RESULT] Baseline Top-5 saved."
                    << std::endl;
            }
            else {
                const bool top1_match =
                    current_top5.front() ==
                    baseline_top5.front();

                const bool top5_match =
                    current_top5 ==
                    baseline_top5;

                std::cout
                    << "[RESULT] Top-1 matches baseline: "
                    << std::boolalpha
                    << top1_match
                    << std::endl;

                std::cout
                    << "[RESULT] Top-5 matches baseline: "
                    << std::boolalpha
                    << top5_match
                    << std::endl;

                if (!top1_match || !top5_match) {
                    throw std::runtime_error(
                        "First-image classification changed "
                        "after dynamic batch switching."
                    );
                }
            }
        }

        std::cout
            << "\n[PASS] TensorRT dynamic batch and "
            << "device buffer reuse verification passed."
            << std::endl;

        std::cout
            << "[PASS] Expected allocation behavior: "
            << "one input expansion and one output expansion."
            << std::endl;
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
