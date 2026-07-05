#pragma once

#include<string>
#include<vector>

struct imageconfig{
    int target_h = 224;
    int target_w = 224;
};

std::vector<float> preprocess(const std::string &image_path, const imageconfig &config);
