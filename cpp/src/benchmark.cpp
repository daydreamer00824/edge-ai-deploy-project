#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include "benchmark.h"

static double compute_average(const std::vector<double>& values){
    if (values.empty()){
        throw std::invalid_argument("compute_average received an empty vector.");
    }

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

static double compute_percentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        throw std::invalid_argument("compute_percentile received an empty vector.");
    }

    std::sort(values.begin(), values.end());

    double rank = percentile / 100.0 * static_cast<double>(values.size());
    int index = static_cast<int>(std::ceil(rank)) - 1;

    if (index < 0) {
        index = 0;
    }

    if (index >= static_cast<int>(values.size())) {
        index = static_cast<int>(values.size()) - 1;
    }

    return values[index];
}

benchmarkSummary summarize_benchmark(
    const std::vector<batchBenchmarkrecord>& records,
    const std::string& backend,
    int batch_size,
    int warmup,
    int repeat,
    int num_images
){
    benchmarkSummary summary;

    summary.backend = backend;
    summary.batch_size = batch_size;
    summary.warmup = warmup;
    summary.repeat = repeat;
    summary.num_images = num_images;
    summary.total_batches = static_cast<int>(records.size());

    if (records.empty()) {
        return summary;
    }

    std::vector<double> end_to_end_list;
    std::vector<double> preprocess_list;
    std::vector<double> infer_list;
    std::vector<double> postprocess_list;

    double total_end_to_end_ms = 0.0;
    int total_samples = 0;

    for (const auto& r : records) {
        end_to_end_list.push_back(r.end_to_end_ms);
        preprocess_list.push_back(r.preprocess_ms);
        infer_list.push_back(r.infer_ms);
        postprocess_list.push_back(r.postprocess_ms);

        total_end_to_end_ms += r.end_to_end_ms;
        total_samples += r.current_batch_size;
    }

    summary.total_samples = total_samples;

    summary.avg_latency_ms = compute_average(end_to_end_list);
    summary.p95_latency_ms = compute_percentile(end_to_end_list, 95.0);

    summary.min_latency_ms = *std::min_element(end_to_end_list.begin(), end_to_end_list.end());
    summary.max_latency_ms = *std::max_element(end_to_end_list.begin(), end_to_end_list.end());

    summary.avg_preprocess_ms = compute_average(preprocess_list);
    summary.avg_infer_ms = compute_average(infer_list);
    summary.avg_postprocess_ms = compute_average(postprocess_list);

    if (total_end_to_end_ms > 0.0) {
        summary.fps = static_cast<double>(total_samples) / (total_end_to_end_ms / 1000.0);
    }

    return summary;
}

void print_benchmark_summary(const benchmarkSummary& summary) {
    std::cout << std::endl;
    std::cout << "========== Benchmark Summary ==========" << std::endl;

    std::cout << "[BENCHMARK] backend: " << summary.backend << std::endl;
    std::cout << "[BENCHMARK] batch_size: " << summary.batch_size << std::endl;
    std::cout << "[BENCHMARK] warmup: " << summary.warmup << std::endl;
    std::cout << "[BENCHMARK] repeat: " << summary.repeat << std::endl;
    std::cout << "[BENCHMARK] num_images: " << summary.num_images << std::endl;
    std::cout << "[BENCHMARK] total_batches: " << summary.total_batches << std::endl;
    std::cout << "[BENCHMARK] total_samples: " << summary.total_samples << std::endl;

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "[BENCHMARK] avg_latency_ms: " << summary.avg_latency_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] p95_latency_ms: " << summary.p95_latency_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] min_latency_ms: " << summary.min_latency_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] max_latency_ms: " << summary.max_latency_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] fps: " << summary.fps << std::endl;
    std::cout << "[BENCHMARK] avg_preprocess_ms: " << summary.avg_preprocess_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] avg_infer_ms: " << summary.avg_infer_ms << " ms" << std::endl;
    std::cout << "[BENCHMARK] avg_postprocess_ms: " << summary.avg_postprocess_ms << " ms" << std::endl;
}

void append_benchmark_csv(
    const std::string& csv_path,
    const benchmarkSummary& summary
) {
    std::filesystem::path path(csv_path);

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    bool need_header = true;

    if (std::filesystem::exists(path)) {
        need_header = std::filesystem::file_size(path) == 0;
    }

    std::ofstream ofs(csv_path, std::ios::app);

    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_path);
    }

    if (need_header) {
        ofs << "backend,"
            << "batch_size,"
            << "warmup,"
            << "repeat,"
            << "num_images,"
            << "total_batches,"
            << "total_samples,"
            << "avg_latency_ms,"
            << "p95_latency_ms,"
            << "min_latency_ms,"
            << "max_latency_ms,"
            << "fps,"
            << "avg_preprocess_ms,"
            << "avg_infer_ms,"
            << "avg_postprocess_ms"
            << "\n";
    }

    ofs << summary.backend << ","
        << summary.batch_size << ","
        << summary.warmup << ","
        << summary.repeat << ","
        << summary.num_images << ","
        << summary.total_batches << ","
        << summary.total_samples << ","
        << std::fixed << std::setprecision(6)
        << summary.avg_latency_ms << ","
        << summary.p95_latency_ms << ","
        << summary.min_latency_ms << ","
        << summary.max_latency_ms << ","
        << summary.fps << ","
        << summary.avg_preprocess_ms << ","
        << summary.avg_infer_ms << ","
        << summary.avg_postprocess_ms
        << "\n";
}