#include<algorithm>
#include<vector>
#include<cmath>
#include <numeric>
#include<stdexcept>
#include "postprocess.h"
#include "label_map.h"

std::vector<float> softmax(const std::vector<float> &logits){
    if(logits.empty()){
        throw std::runtime_error("Softmax input is empty.");
    }

    float max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<float> probs(logits.size());
    float sum_exp = 0.0f;

    for(size_t i = 0; i < logits.size(); i++){
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    if (sum_exp == 0.0f){
        throw std::runtime_error("Softmax sum is zero.");
    }

    for (auto &f : probs){
        f /= sum_exp;
    }

    return probs;
    
}

std::vector<topk_result> get_topk(const std::vector<float> &logits, int k, const std::vector<std::string> &labels){
    std:: vector<float> probs = softmax(logits);
    k = std::min(k, static_cast<int>(probs.size()));

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(
        indices.begin(),
        indices.begin() + k, indices.end(),
        [&probs](int a, int b){
            return probs[a] > probs[b];
        }
    );

    std::vector<topk_result> results;
    results.reserve(k);

    for(int i = 0; i < k; i++){
        int idx = indices[i];

        results.push_back({
            idx,
            index_to_label(labels, idx),
            probs[idx]
        });
    }
    return results;
}