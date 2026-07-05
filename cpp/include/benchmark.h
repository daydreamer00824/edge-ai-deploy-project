#pragma once
#include <string>
#include <vector>


struct batchBenchmarkrecord{
    int repeat_index = 0;
    int batch_index = 0;
    int current_batch_size = 0;

    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double end_to_end_ms = 0.0;
};

struct benchmarkSummary{
    std::string backend = "ort_cpu";

    int batch_size = 0;
    int warmup = 0;
    int repeat = 0;
    int num_images = 0;
    int total_batches = 0;
    int total_samples = 0;

    double avg_latency_ms = 0.0;
    double p95_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double fps = 0.0;

    double avg_preprocess_ms = 0.0;
    double avg_infer_ms = 0.0;
    double avg_postprocess_ms = 0.0;
};

benchmarkSummary summarize_benchmark(
    const std::vector<batchBenchmarkrecord>& records,
    const std::string& backend,
    int batch_size,
    int warmup,
    int repeat,
    int num_images
);

void print_benchmark_summary(const benchmarkSummary& summary);

void append_benchmark_csv(const std::string& csv_path, const benchmarkSummary& summary);