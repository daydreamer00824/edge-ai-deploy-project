#include<string>
#include<vector>
#include<filesystem>
#include<stdexcept>
#include<fstream>
#include "label_map.h"

std::vector<std::string> load_labels(const std::string &label_path){
    if(!std::filesystem::exists(label_path)){
        throw std::runtime_error("Label file does not exist: " + label_path);
    }

    std::ifstream file(label_path);

    if(!file.is_open()){
        throw std::runtime_error("Failed to open label file: " + label_path);
    }

    std::vector<std::string> labels;
    std::string line;

    while(std::getline(file, line)){
        if(!line.empty()){
            labels.push_back(line);
        }
    }

    if(labels.empty()){
        throw std::runtime_error("Label file is empty: " +label_path);
    }

    return labels;
}

std::string index_to_label(const std::vector<std::string> &labels, int index){
    if(index < 0 || index >= static_cast<int>(labels.size())){
        return "unknown_" + std::to_string(index);
    }

    return labels[index];
    
}