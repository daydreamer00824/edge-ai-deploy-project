#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "benchmark.h"
#include "batch_infer.h"
#include "image_process.h"
#include "label_map.h"
#include "ort_runner.h"
#include "postprocess.h"
#include "timer.h"

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
              << " <model.onnx> <image_dir> <labels.txt> <batch_size> <warmup> <repeat> <csv_output>"
              << std::endl;

    std::cerr << "[EXAMPLE] "
              << program_name
              << " ../models/resnet18.onnx ../data/images ../data/labels.txt 8 5 50 ../results/benchmark_ort_cpp.csv"
              << std::endl;
}

static void run_one_batch(
    OrtRunner& runner,
    const std::vector<std::string>& image_paths,
    const std::vector<std::string>& labels,
    int start,
    int end,
    const imageconfig& preprocess_config,
    int repeat_index,
    int batch_index,
    bool record,
    std::vector<batchBenchmarkrecord>& records
) {
    const int channels = 3;
    const int current_batch_size = end - start;

    if (current_batch_size <= 0) {
        throw std::runtime_error(
            "current_batch_size must be greater than 0."
        );
    }

    const size_t single_image_tensor_size =
        static_cast<size_t>(channels) *
        static_cast<size_t>(preprocess_config.target_h) *
        static_cast<size_t>(preprocess_config.target_w);

    const size_t expected_input_size =
        static_cast<size_t>(current_batch_size) * single_image_tensor_size;

    Timerecorder batch_timer;

    std::vector<float> batch_input_tensor;

    {
        scopedTimer t(batch_timer, "preprocess");

        batch_input_tensor =
            build_batch_tensor(
                image_paths,
                start,
                end,
                preprocess_config
            );
    }

    if (batch_input_tensor.size() != expected_input_size) {
        throw std::runtime_error(
            "Batch input tensor size mismatch. Expected " +
            std::to_string(expected_input_size) +
            ", but got " +
            std::to_string(batch_input_tensor.size()) +
            "."
        );
    }

    std::vector<int64_t> input_shape = {
        static_cast<int64_t>(current_batch_size),
        static_cast<int64_t>(channels),
        static_cast<int64_t>(preprocess_config.target_h),
        static_cast<int64_t>(preprocess_config.target_w)
    };

    std::vector<float> batch_logits;

    {
        scopedTimer t(batch_timer, "infer");

        batch_logits =
            runner.run(
                batch_input_tensor,
                input_shape
            );
    }

    if (batch_logits.empty()) {
        throw std::runtime_error(
            "Batch output logits is empty."
        );
    }

    if (batch_logits.size() % static_cast<size_t>(current_batch_size) != 0) {
        throw std::runtime_error(
            "Batch output size is not divisible by current batch size."
        );
    }

    const int num_classes =
        static_cast<int>(
            batch_logits.size() / static_cast<size_t>(current_batch_size)
        );

    {
        scopedTimer t(batch_timer, "postprocess");

        for (int i = 0; i < current_batch_size; ++i) {
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

            if (top5.empty()) {
                throw std::runtime_error(
                    "TopK result is empty."
                );
            }
        }
    }

    if (record) {
        batchBenchmarkrecord r;

        r.repeat_index = repeat_index;
        r.batch_index = batch_index;
        r.current_batch_size = current_batch_size;

        r.preprocess_ms = batch_timer.get("preprocess");
        r.infer_ms = batch_timer.get("infer");
        r.postprocess_ms = batch_timer.get("postprocess");

        r.end_to_end_ms =
            r.preprocess_ms +
            r.infer_ms +
            r.postprocess_ms;

        records.push_back(r);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const std::string model_path = argv[1];
        const std::string image_dir = argv[2];
        const std::string label_path = argv[3];

        const int batch_size = parse_int_arg(argv[4], "batch_size");
        const int warmup = parse_int_arg(argv[5], "warmup");
        const int repeat = parse_int_arg(argv[6], "repeat");

        const std::string csv_output_path = argv[7];

        if (batch_size <= 0) {
            throw std::invalid_argument(
                "batch_size must be greater than 0."
            );
        }

        if (warmup < 0) {
            throw std::invalid_argument(
                "warmup must be greater than or equal to 0."
            );
        }

        if (repeat <= 0) {
            throw std::invalid_argument(
                "repeat must be greater than 0."
            );
        }

        const std::string backend = "ORT_CPU";

        imageconfig preprocess_config;
        preprocess_config.target_h = 224;
        preprocess_config.target_w = 224;

        std::cout << "[INFO] C++ ONNX Runtime benchmark started."
                  << std::endl;

        std::cout << "[INFO] Backend: "
                  << backend
                  << std::endl;

        std::cout << "[INFO] OpenCV version: "
                  << CV_VERSION
                  << std::endl;

        std::cout << "[INFO] Model path: "
                  << model_path
                  << std::endl;

        std::cout << "[INFO] Image dir: "
                  << image_dir
                  << std::endl;

        std::cout << "[INFO] Label path: "
                  << label_path
                  << std::endl;

        std::cout << "[INFO] Batch size: "
                  << batch_size
                  << std::endl;

        std::cout << "[INFO] Warmup: "
                  << warmup
                  << std::endl;

        std::cout << "[INFO] Repeat: "
                  << repeat
                  << std::endl;

        std::cout << "[INFO] CSV output: "
                  << csv_output_path
                  << std::endl;

        Timerecorder init_timer;

        std::vector<std::string> labels;

        {
            scopedTimer t(init_timer, "load_label");

            labels = load_labels(label_path);
        }

        if (labels.empty()) {
            throw std::runtime_error(
                "Labels are empty. Please check labels.txt."
            );
        }

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "[INFO] Labels loaded: "
                  << labels.size()
                  << " | cost: "
                  << init_timer.get("load_label")
                  << " ms"
                  << std::endl;

        std::vector<std::string> image_paths;

        {
            scopedTimer t(init_timer, "collect_images");

            image_paths = collect_image_path(image_dir);
        }

        if (image_paths.empty()) {
            throw std::runtime_error(
                "No valid images found. Please check image_dir."
            );
        }

        std::cout << "[INFO] Images found: "
                  << image_paths.size()
                  << " | cost: "
                  << init_timer.get("collect_images")
                  << " ms"
                  << std::endl;

        std::unique_ptr<OrtRunner> runner;

        {
            scopedTimer t(init_timer, "session_create");

            runner = std::make_unique<OrtRunner>(model_path);
        }

        std::cout << "[INFO] Session created | cost: "
                  << init_timer.get("session_create")
                  << " ms"
                  << std::endl;

        runner->print_model_info();

        const int total_images =
            static_cast<int>(image_paths.size());

        std::vector<batchBenchmarkrecord> records;

        std::cout << std::endl;
        std::cout << "========== Warmup =========="
                  << std::endl;

        for (int w = 0; w < warmup; ++w) {
            int batch_index = 0;

            for (int start = 0; start < total_images; start += batch_size) {
                const int end =
                    std::min(start + batch_size, total_images);

                ++batch_index;

                run_one_batch(
                    *runner,
                    image_paths,
                    labels,
                    start,
                    end,
                    preprocess_config,
                    w + 1,
                    batch_index,
                    false,
                    records
                );
            }

            std::cout << "[WARMUP] "
                      << w + 1
                      << "/"
                      << warmup
                      << " finished"
                      << std::endl;
        }

        std::cout << std::endl;
        std::cout << "========== Repeat Benchmark =========="
                  << std::endl;

        for (int r = 0; r < repeat; ++r) {
            int batch_index = 0;

            for (int start = 0; start < total_images; start += batch_size) {
                const int end =
                    std::min(start + batch_size, total_images);

                ++batch_index;

                run_one_batch(
                    *runner,
                    image_paths,
                    labels,
                    start,
                    end,
                    preprocess_config,
                    r + 1,
                    batch_index,
                    true,
                    records
                );
            }

            std::cout << "[REPEAT] "
                      << r + 1
                      << "/"
                      << repeat
                      << " finished"
                      << std::endl;
        }

        benchmarkSummary summary =
            summarize_benchmark(
                records,
                backend,
                batch_size,
                warmup,
                repeat,
                total_images
            );

        print_benchmark_summary(summary);

        append_benchmark_csv(
            csv_output_path,
            summary
        );

        std::cout << "[INFO] Benchmark CSV saved to: "
                  << csv_output_path
                  << std::endl;

        std::cout << "[INFO] C++ ONNX Runtime benchmark finished."
                  << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception: "
                  << e.what()
                  << std::endl;

        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] "
                  << e.what()
                  << std::endl;

        return 1;
    }

    return 0;
}