#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "batch_infer.h"
#include "image_process.h"
#include "label_map.h"
#include "postprocess.h"
#include "timer.h"
#include "trt_runner.h"

static int parse_int_arg(const char* value, const std::string& name) {
    try {
        return std::stoi(value);
    }
    catch (const std::exception& e) {
        throw std::invalid_argument(
            name + " must be an integer. Detail: " + e.what()
        );
    }
}

static void print_usage(const char* program_name) {
    std::cerr << "[ERROR] Usage: "
              << program_name
              << " <model.engine> <image_dir> <labels.txt> <batch_size>"
              << std::endl;

    std::cerr << "[EXAMPLE] "
              << program_name
              << " ../models/resnet18_fp16.engine ../data/images ../labels/imagenet_classes.txt 8"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string engine_path = argv[1];
        const std::string image_dir = argv[2];
        const std::string label_path = argv[3];
        const int batch_size = parse_int_arg(argv[4], "batch_size");

        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size must be greater than 0.");
        }

        const int channels = 3;

        imageconfig preprocess_config;
        preprocess_config.target_h = 224;
        preprocess_config.target_w = 224;

        Timerecorder timer;

        std::cout << "[INFO] TensorRT C++ inference started." << std::endl;
        std::cout << "[INFO] OpenCV version: " << CV_VERSION << std::endl;
        std::cout << "[INFO] Engine path: " << engine_path << std::endl;
        std::cout << "[INFO] Image dir  : " << image_dir << std::endl;
        std::cout << "[INFO] Label path : " << label_path << std::endl;
        std::cout << "[INFO] Batch size : " << batch_size << std::endl;

        std::vector<std::string> labels;

        {
            scopedTimer t(timer, "load_label");
            labels = load_labels(label_path);
        }

        std::vector<std::string> image_paths;

        {
            scopedTimer t(timer, "collect_images");
            image_paths = collect_image_path(image_dir);
        }

        std::cout << "[INFO] Labels loaded: " << labels.size() << std::endl;
        std::cout << "[INFO] Images found : " << image_paths.size() << std::endl;

        TrtRunner runner(engine_path);
        runner.print_engine_info();

        const int total_images = static_cast<int>(image_paths.size());

        int batch_index = 0;

        for (int start = 0; start < total_images; start += batch_size) {
            const int end = std::min(start + batch_size, total_images);
            const int current_batch_size = end - start;

            ++batch_index;

            std::cout << std::endl;
            std::cout << "========== TensorRT Batch "
                      << batch_index
                      << " | current_batch_size="
                      << current_batch_size
                      << " =========="
                      << std::endl;

            std::vector<float> batch_input_tensor;

            {
                scopedTimer t(timer, "preprocess");

                batch_input_tensor = build_batch_tensor(
                    image_paths,
                    start,
                    end,
                    preprocess_config
                );
            }

            const int single_image_tensor_size =
                channels *
                preprocess_config.target_h *
                preprocess_config.target_w;

            const int expected_input_size =
                current_batch_size * single_image_tensor_size;

            if (static_cast<int>(batch_input_tensor.size()) != expected_input_size) {
                throw std::runtime_error(
                    "Batch input tensor size mismatch. Expected " +
                    std::to_string(expected_input_size) +
                    ", but got " +
                    std::to_string(batch_input_tensor.size())
                );
            }

            std::vector<float> batch_logits;

            {
                scopedTimer t(timer, "infer");

                batch_logits = runner.run(
                    batch_input_tensor,
                    current_batch_size,
                    channels,
                    preprocess_config.target_h,
                    preprocess_config.target_w
                );
            }

            if (batch_logits.empty()) {
                throw std::runtime_error("TensorRT output logits is empty.");
            }

            if (batch_logits.size() % static_cast<size_t>(current_batch_size) != 0) {
                throw std::runtime_error(
                    "TensorRT output size is not divisible by current batch size."
                );
            }

            const int num_classes =
                static_cast<int>(
                    batch_logits.size() / static_cast<size_t>(current_batch_size)
                );

            std::cout << "[INFO] TensorRT output element count: "
                      << batch_logits.size()
                      << std::endl;

            std::cout << "[INFO] Num classes: "
                      << num_classes
                      << std::endl;

            {
                scopedTimer t(timer, "postprocess");

                for (int i = 0; i < current_batch_size; ++i) {
                    const int global_image_index = start + i;

                    std::vector<float> single_logits =
                        slice_logits_for_one_image(
                            batch_logits,
                            i,
                            num_classes
                        );

                    std::vector<topk_result> top5 =
                        get_topk(
                            single_logits,
                            5,
                            labels
                        );

                    std::cout << std::endl;
                    std::cout << "[IMAGE] "
                              << std::filesystem::path(
                                     image_paths[global_image_index]
                                 ).filename().string()
                              << std::endl;

                    if (!top5.empty()) {
                        std::cout << "[TOP1] index="
                                  << top5[0].index
                                  << ", label="
                                  << top5[0].label
                                  << ", score="
                                  << std::fixed
                                  << std::setprecision(6)
                                  << top5[0].score
                                  << std::endl;
                    }

                    std::cout << "[TOP5]" << std::endl;

                    for (size_t k = 0; k < top5.size(); ++k) {
                        std::cout << k + 1
                                  << ". index="
                                  << top5[k].index
                                  << ", label="
                                  << top5[k].label
                                  << ", score="
                                  << std::fixed
                                  << std::setprecision(6)
                                  << top5[k].score
                                  << std::endl;
                    }
                }
            }
        }

        std::cout << std::endl;
        std::cout << "========== TensorRT Time Summary ==========" << std::endl;

        std::cout << "[TIME] label load     : "
                  << timer.get("load_label")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] collect images : "
                  << timer.get("collect_images")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] preprocess total: "
                  << timer.get("preprocess")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] inference total : "
                  << timer.get("infer")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] postprocess total: "
                  << timer.get("postprocess")
                  << " ms"
                  << std::endl;

        std::cout << "[INFO] TensorRT C++ inference finished." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] "
                  << e.what()
                  << std::endl;
        return 1;
    }

    return 0;
}