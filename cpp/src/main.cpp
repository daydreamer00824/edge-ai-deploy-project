#include<iostream>
#include<string>
#include<vector>
#include<memory>
#include<iomanip>
#include<opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include "timer.h"
#include "image_process.h"
#include "label_map.h"
#include "postprocess.h"
#include "ort_runner.h"
#include "batch_infer.h"

int main(int argc, char* argv[]){
    if (argc < 5) {
        std::cerr << "[ERROR] Usage: "
                  << argv[0]
                  << " <model.onnx> <image_path> <labels.txt> <batch_size>"
                  << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_dir = argv[2];
    const std::string label_path = argv[3];

    int batch_size = 0;

    try{
        batch_size = std::stoi(argv[4]);
    }
    catch(const std::exception &e){
        std::cerr << "[ERROR] batch_size must be an integer." << e.what() << std::endl;
        return 1;
    }

    if(batch_size <= 0){
        std::cerr << "[ERROR] batch_size must be > 0." << std::endl;
        return 1;
    }

    const int channels = 3;

    imageconfig preprocess_config;
    preprocess_config.target_h = 224;
    preprocess_config.target_w = 224;

    Timerecorder timer;

    try{
        std::cout << "[INFO] C++ ONNX Runtime single image inference started."
                  << std::endl;

        std::cout << "[INFO] OpenCV version: "
                  << CV_VERSION
                  << std::endl;

        std::cout << "[INFO] Model path: "
                  << model_path
                  << std::endl;

        std::cout << "[INFO] Image dir "
                  << image_dir
                  << std::endl;

        std::cout << "[INFO] Label path: "
                  << label_path
                  << std::endl;
        std::vector<std::string> labels;

        {
            scopedTimer t(timer, "load_label");
            labels = load_labels(label_path);
        }

        std::cout << "[INFO] Labels loaded: "
                  << labels.size()
                  << std::endl;

        std::vector<std::string> image_path;
        {
            scopedTimer t(timer, "collect_images");
            image_path = collect_image_path(image_dir);
        }
        std::cout << "[INFO] Images found: "
                  << image_path.size()
                  << std::endl;

        std::unique_ptr<OrtRunner> runner;
        {
            scopedTimer t(timer, "session_create");
            runner = std::make_unique<OrtRunner>(model_path);
        }
        runner->print_model_info();

        const int single_image_tensor_size = channels * preprocess_config.target_h * preprocess_config.target_w;

        const int total_images = static_cast<int>(image_path.size());

        int batch_index = 0;

        for (int start = 0; start < total_images; start += batch_size){
            const int end = std::min(start + batch_size, total_images);
            const int current_batch_size = end - start;

            batch_index++;

            std::cout << std::endl;
            std::cout << "========== Batch " << batch_index
                      << " | current_batch_size = "
                      << current_batch_size
                      << " =========="
                      << std::endl;

            std::vector<float> batch_input_tensor;
            {
                scopedTimer t(timer, "preprocess");
                batch_input_tensor = build_batch_tensor(image_path, start, end, preprocess_config);
            }

            const int expected_input_size =
                current_batch_size * single_image_tensor_size;

            std::cout << "[INFO] Batch input tensor size: "
                      << batch_input_tensor.size()
                      << std::endl;

            std::cout << "[INFO] Expected batch tensor size: "
                      << expected_input_size
                      << std::endl;

            if (static_cast<int>(batch_input_tensor.size()) != expected_input_size) {
                throw std::runtime_error("Batch input tensor size mismatch.");
            }

            std::vector<int64_t> input_shape = {
                current_batch_size,
                channels,
                preprocess_config.target_h,
                preprocess_config.target_w
            };

            std::vector<float> batch_logits;
            {
                scopedTimer t(timer, "infer");
                batch_logits = runner -> run(batch_input_tensor, input_shape);
            }

            std::cout << "[INFO] Batch output element count: "
                      << batch_logits.size()
                      << std::endl;

            if(batch_logits.size() % current_batch_size != 0){
                throw std::runtime_error("Batch output size is not divisible by current batch size.");
            }

            const int num_classes = static_cast<int>(batch_logits.size() / current_batch_size);
            std::cout << "[INFO] Num classes: "
                      << num_classes
                      << std::endl;

            {
                scopedTimer t(timer, "postprocess");

                for(int i = 0; i < current_batch_size; i++){
                    const int global_image_index = start + i;

                    std::vector<float> single_logits = slice_logits_for_one_image(batch_logits, i, num_classes);

                    std::vector<topk_result> top5 = get_topk(single_logits, 5, labels);

                    std::cout << std::endl;
                    std::cout << "[IMAGE] "
                              << std::filesystem::path(image_path[global_image_index]).filename().string()
                              << std::endl;

                     if (!top5.empty()) {
                        std::cout << "[TOP1] index=" << top5[0].index
                                  << ", label=" << top5[0].label
                                  << ", score=" << std::fixed << std::setprecision(6)
                                  << top5[0].score
                                  << std::endl;
                    }

                    std::cout << "[TOP5]" << std::endl;

                    for (size_t k = 0; k < top5.size(); ++k) {
                        std::cout << k + 1
                                  << ". index=" << top5[k].index
                                  << ", label=" << top5[k].label
                                  << ", score=" << std::fixed << std::setprecision(6)
                                  << top5[k].score
                                  << std::endl;
                    }
                }
            }
        }

        std::cout << std::endl;
        std::cout << "========== Time Summary ==========" << std::endl;

        std::cout << "[TIME] label load: "
                  << timer.get("load_label")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] collect images: "
                  << timer.get("collect_images")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] session create: "
                  << timer.get("session_create")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] preprocess total: "
                  << timer.get("preprocess")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] inference total: "
                  << timer.get("infer")
                  << " ms"
                  << std::endl;

        std::cout << "[TIME] postprocess total: "
                  << timer.get("postprocess")
                  << " ms"
                  << std::endl;

        std::cout << "[INFO] Batch inference finished." << std::endl;
        
    }
    catch(const Ort::Exception& e){
        std::cerr << "[ERROR] ONNX Runtime exception: "
                  << e.what()
                  << std::endl;
        return 1;
    }
    catch(const std::exception& e){
        std::cerr << "[ERROR] "
                  << e.what()
                  << std::endl;
        return 1;
    }

    return 0;
}