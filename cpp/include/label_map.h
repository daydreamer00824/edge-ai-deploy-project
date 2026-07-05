#pragma once

#include<vector>
#include<string>
#include<filesystem>

std::vector<std::string> load_labels(const std::string &label_path);
std::string index_to_label(const std::vector<std::string> &labels, int index);