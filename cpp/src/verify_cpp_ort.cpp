#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include "image_process.h"
#include "ort_runner.h"
#include "label_map.h"
#include "postprocess.h"

static void save_logits_csv(
    const std::vector<float>& logits,
    const std::filesystem::path& save_path
) {
    std::ofstream ofs(save_path);

    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open logits save file: " + save_path.string());
    }

    ofs << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < logits.size(); ++i) {
        ofs << logits[i];

        if (i + 1 != logits.size()) {
            ofs << ",";
        }
    }

    ofs << "\n";
}

static void save_topk_csv(
    const std::vector<topk_result>& topk,
    const std::filesystem::path& save_path
) {
    std::ofstream ofs(save_path);

    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open topk save file: " + save_path.string());
    }

    ofs << "rank,index,score,label\n";

    ofs << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < topk.size(); ++i) {
        ofs << i + 1 << ","
            << topk[i].index << ","
            << topk[i].score << ","
            << topk[i].label
            << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "[ERROR] Usage: "
                  << argv[0]
                  << " <model.onnx> <image.jpg> <labels.txt> <output_dir>"
                  << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string label_path = argv[3];
    const std::filesystem::path output_dir = argv[4];

    try {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file does not exist: " + model_path);
        }

        if (!std::filesystem::exists(image_path)) {
            throw std::runtime_error("Image file does not exist: " + image_path);
        }

        if (!std::filesystem::exists(label_path)) {
            throw std::runtime_error("Label file does not exist: " + label_path);
        }

        std::filesystem::create_directories(output_dir);

        std::cout << "========== C++ ORT Consistency Verify ==========" << std::endl;
        std::cout << "[INFO] Model path : " << model_path << std::endl;
        std::cout << "[INFO] Image path : " << image_path << std::endl;
        std::cout << "[INFO] Label path : " << label_path << std::endl;
        std::cout << "[INFO] Output dir : " << output_dir.string() << std::endl;

        imageconfig config;
        config.target_h = 224;
        config.target_w = 224;

        std::vector<float> input_tensor = preprocess(image_path, config);

        std::vector<int64_t> input_shape = {
            1,
            3,
            config.target_h,
            config.target_w
        };

        std::cout << "[INFO] Input tensor size: "
                  << input_tensor.size()
                  << std::endl;

        std::cout << "[INFO] Input shape: [1, 3, "
                  << config.target_h
                  << ", "
                  << config.target_w
                  << "]"
                  << std::endl;

        OrtRunner runner(model_path);
        runner.print_model_info();

        std::vector<float> logits = runner.run(input_tensor, input_shape);

        std::cout << "[INFO] Output logits size: "
                  << logits.size()
                  << std::endl;

        std::vector<std::string> labels = load_labels(label_path);
        std::vector<topk_result> top5 = get_topk(logits, 5, labels);

        std::cout << "---------- C++ Top5 ----------" << std::endl;

        for (size_t i = 0; i < top5.size(); ++i) {
            std::cout << i + 1
                      << ". index=" << top5[i].index
                      << ", label=" << top5[i].label
                      << ", score=" << std::fixed << std::setprecision(6)
                      << top5[i].score
                      << std::endl;
        }

        const std::filesystem::path logits_path = output_dir / "cpp_logits.csv";
        const std::filesystem::path topk_path = output_dir / "cpp_top5.csv";

        save_logits_csv(logits, logits_path);
        save_topk_csv(top5, topk_path);

        std::cout << "[INFO] C++ logits saved: "
                  << logits_path.string()
                  << std::endl;

        std::cout << "[INFO] C++ top5 saved: "
                  << topk_path.string()
                  << std::endl;

        std::cout << "[INFO] C++ ORT consistency verify finished."
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