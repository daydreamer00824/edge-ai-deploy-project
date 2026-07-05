import csv
from pathlib import Path
import logging

def save_benchmark(benchmark_result, output_dir:Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = output_dir / "benchmark_results.csv"

    benchmark_name = [
        "provider",
        "batch_size",
        "repeat_id",
        "image_count",
        "preprocess_time",
        "infer_time",
        "postprocess_time",
        "save_result_time",
        "avg_infer_ms",
        "throughput",
    ]

    file_exist = benchmark_path.exists()

    with open(benchmark_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, benchmark_name)

        if not file_exist:
            writer.writeheader()

        writer.writerow(benchmark_result)

    logging.info(f"benchmark 结果已保存: {benchmark_path}")

    return benchmark_path

def save_benchmark_summary(summary_resluts, output_dir:Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "benchmark_summary.csv"

    fieldnames = [
        "provider",
        "batch_size",
        "repeat_count",
        "image_count",
        "avg_infer_ms_mean",
        "avg_infer_ms_std",
        "throughput_mean",
        "throughput_std",
        "best_by_throughput",
    ]

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_resluts)

    logging.info(f"benchmark 汇总结果已保存: {result_path}")

    return result_path
