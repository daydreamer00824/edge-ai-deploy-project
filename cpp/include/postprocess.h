#pragma once

#include <string>
#include <vector>

struct topk_result{
    int index;
    std::string label;
    float score;
};

std::vector<float> softmax(const std::vector<float> &logits);
std::vector<topk_result> get_topk(const std::vector<float> &logits, int k, const std::vector<std::string> &labels);
