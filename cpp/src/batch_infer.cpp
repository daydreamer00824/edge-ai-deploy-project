#include "batch_infer.h"
#include<stdexcept>
#include<algorithm>
#include<cctype>

bool is_image_file(const std::filesystem::path &path){
    if(!std::filesystem::exists(path)){
        throw std::runtime_error("path not exist: " + path.string());
    }
    if(!path.has_extension()){
        throw std::runtime_error(path.string() + "not a image");
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

std::vector<std::string> collect_image_path(const std::string &image_dir){
    if(!std::filesystem::exists(image_dir)){
        throw std::runtime_error("Image directory does not exist: " + image_dir);
    }

    if(!std::filesystem::is_directory(image_dir)){
        throw std::runtime_error("Input path is not a directory: " + image_dir);
    }
    std::vector<std::string> image_path;

    for(const auto &p : std::filesystem::directory_iterator(image_dir)){
        if(!p.is_regular_file()){
            continue;
        }
        if(is_image_file(p.path())){
            image_path.push_back(p.path().string());
        }
    }

    std::sort(image_path.begin(), image_path.end());

    if(image_path.empty()){
        throw std::runtime_error("No image files found in directory: " + image_dir);
    }

    return image_path;
}

std::vector<float> build_batch_tensor(const std::vector<std::string> &image_path, int start, int end, const imageconfig &config){
    if(start < 0 || end < 0){
        throw std::runtime_error("start and end must be non-negative.");
    }
    if(start >= end){
        throw std::runtime_error("Invalid batch range: start must be smaller than end."); 
    }

    if(end > static_cast<int>(image_path.size())){
        throw std::runtime_error("Batch range exceeds image_paths size.");
    }

    const int channels = 3;
    const int single_image_tensor_size = channels * config.target_h * config.target_w;

    const int current_batch_size = end - start;

    std::vector<float> batch_tensor;
    batch_tensor.reserve(single_image_tensor_size * current_batch_size);

    for (int i = start; i < end; i++){
        std::vector<float> single_tensor = preprocess(image_path[i], config);

        if(static_cast<int>(single_tensor.size()) != single_image_tensor_size){
            throw std::runtime_error(
                "Single image tensor size mismatch: " + image_path[i]
            );
        }

        batch_tensor.insert(batch_tensor.end(), single_tensor.begin(), single_tensor.end());
    }

    return batch_tensor;
}

std::vector<float> slice_logits_for_one_image(const std::vector<float> &batch_logits, int image_index, int num_class){
    if(image_index < 0){
        throw std::runtime_error("image_index must be non-negative.");
    }
    if(num_class <= 0){
        throw std::runtime_error("num_classes must be greater than 0.");
    }

    const int start = image_index * num_class;
    const int end = start + num_class;

    if(end > static_cast<int>(batch_logits.size())){
        throw std::runtime_error("Logits slice range exceeds batch_logits size.");
    }

    return std::vector<float>(batch_logits.begin() + start, batch_logits.begin() + end);
}