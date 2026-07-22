#pragma once
#include <string>
#include <vector>


struct batchBenchmarkrecord {
    int repeat_index = 0;
    int batch_index = 0;
    int current_batch_size = 0;

    double preprocess_ms = 0.0;

    // 原字段保留，避免破坏ORT benchmark和旧统计逻辑
    double infer_ms = 0.0;

    // TensorRT完整run()的CPU墙钟时间
    double trt_run_wall_ms = 0.0;

    // CUDA Event统计的GPU阶段时间
    double h2d_ms = 0.0;
    double gpu_inference_ms = 0.0;
    double d2h_ms = 0.0;
    double gpu_total_ms = 0.0;

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

    // TensorRT阶段计时
    double avg_trt_run_wall_ms = 0.0;
    double avg_h2d_ms = 0.0;
    double avg_gpu_inference_ms = 0.0;
    double avg_d2h_ms = 0.0;
    double avg_gpu_total_ms = 0.0;
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

void append_trt_stage_timing_csv(const std::string& csv_path, const benchmarkSummary& summary);
