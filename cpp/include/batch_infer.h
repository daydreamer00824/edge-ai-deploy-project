#pragma once

#include<string>
#include<filesystem>
#include<vector>
#include "image_process.h"

bool is_image_file(const std::filesystem::path &path);
std::vector<std::string> collect_image_path(const std::string &image_dir);
std::vector<float> build_batch_tensor(const std::vector<std::string> &image_path, int start, int end, const imageconfig &config);
std::vector<float> slice_logits_for_one_image(const std::vector<float> &batch_logits, int image_index, int num_class);