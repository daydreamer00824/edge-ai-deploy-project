#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple


RowKey = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TensorRT benchmark results before and after "
            "device-buffer reuse."
        )
    )

    parser.add_argument(
        "--pre",
        required=True,
        type=Path,
        help="Pre-reuse TensorRT stage timing CSV.",
    )

    parser.add_argument(
        "--post",
        required=True,
        type=Path,
        help="Post-reuse TensorRT stage timing CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output comparison CSV.",
    )

    return parser.parse_args()


def infer_precision(backend: str) -> str:
    if "FP32" in backend:
        return "FP32"

    if "FP16" in backend:
        return "FP16"

    raise ValueError(
        f"Cannot infer precision from backend: {backend}"
    )


def load_results(csv_path: Path) -> Dict[RowKey, dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV does not exist: {csv_path}"
        )

    required_fields = {
        "backend",
        "batch_size",
        "warmup",
        "repeat",
        "num_images",
        "avg_latency_ms",
        "p95_latency_ms",
        "fps",
        "avg_trt_run_wall_ms",
        "avg_gpu_inference_ms",
        "avg_gpu_total_ms",
        "wall_minus_gpu_total_ms",
    }

    results: Dict[RowKey, dict[str, str]] = {}

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {csv_path}"
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                f"CSV is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        for row in reader:
            precision = infer_precision(row["backend"])
            batch_size = int(row["batch_size"])
            key = (precision, batch_size)

            if key in results:
                raise ValueError(
                    f"Duplicate result for {key} in {csv_path}"
                )

            results[key] = row

    return results


def get_float(
    row: dict[str, str],
    field_name: str,
) -> float:
    return float(row[field_name])


def reduction_percent(
    before: float,
    after: float,
) -> float:
    if before == 0.0:
        return 0.0

    return (
        (before - after)
        / before
        * 100.0
    )


def gain_percent(
    before: float,
    after: float,
) -> float:
    if before == 0.0:
        return 0.0

    return (
        (after - before)
        / before
        * 100.0
    )


def validate_same_test_condition(
    pre_row: dict[str, str],
    post_row: dict[str, str],
    key: RowKey,
) -> None:
    fields = [
        "batch_size",
        "warmup",
        "repeat",
        "num_images",
    ]

    for field_name in fields:
        if pre_row[field_name] != post_row[field_name]:
            raise ValueError(
                f"Test condition mismatch for {key}: "
                f"{field_name}, "
                f"pre={pre_row[field_name]}, "
                f"post={post_row[field_name]}"
            )


def main() -> None:
    args = parse_args()

    pre_results = load_results(args.pre)
    post_results = load_results(args.post)

    if set(pre_results) != set(post_results):
        raise ValueError(
            "Pre-reuse and post-reuse CSV files "
            "do not contain the same precision/batch groups."
        )

    output_fields = [
        "precision",
        "batch_size",
        "warmup",
        "repeat",
        "num_images",
        "pre_trt_run_wall_ms",
        "post_trt_run_wall_ms",
        "trt_run_reduction_ms",
        "trt_run_reduction_pct",
        "pre_end_to_end_ms",
        "post_end_to_end_ms",
        "end_to_end_reduction_ms",
        "end_to_end_reduction_pct",
        "pre_p95_latency_ms",
        "post_p95_latency_ms",
        "p95_reduction_ms",
        "p95_reduction_pct",
        "pre_fps",
        "post_fps",
        "fps_gain",
        "fps_gain_pct",
        "pre_gpu_inference_ms",
        "post_gpu_inference_ms",
        "pre_gpu_total_ms",
        "post_gpu_total_ms",
        "pre_wall_minus_gpu_total_ms",
        "post_wall_minus_gpu_total_ms",
        "host_overhead_reduction_ms",
    ]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_rows: list[dict[str, object]] = []

    precision_order = {
        "FP32": 0,
        "FP16": 1,
    }

    sorted_keys = sorted(
        pre_results,
        key=lambda item: (
            precision_order[item[0]],
            item[1],
        ),
    )

    for key in sorted_keys:
        precision, batch_size = key

        pre_row = pre_results[key]
        post_row = post_results[key]

        validate_same_test_condition(
            pre_row,
            post_row,
            key,
        )

        pre_wall = get_float(
            pre_row,
            "avg_trt_run_wall_ms",
        )

        post_wall = get_float(
            post_row,
            "avg_trt_run_wall_ms",
        )

        pre_e2e = get_float(
            pre_row,
            "avg_latency_ms",
        )

        post_e2e = get_float(
            post_row,
            "avg_latency_ms",
        )

        pre_p95 = get_float(
            pre_row,
            "p95_latency_ms",
        )

        post_p95 = get_float(
            post_row,
            "p95_latency_ms",
        )

        pre_fps = get_float(
            pre_row,
            "fps",
        )

        post_fps = get_float(
            post_row,
            "fps",
        )

        pre_host_overhead = get_float(
            pre_row,
            "wall_minus_gpu_total_ms",
        )

        post_host_overhead = get_float(
            post_row,
            "wall_minus_gpu_total_ms",
        )

        comparison_rows.append(
            {
                "precision": precision,
                "batch_size": batch_size,
                "warmup": int(pre_row["warmup"]),
                "repeat": int(pre_row["repeat"]),
                "num_images": int(
                    pre_row["num_images"]
                ),
                "pre_trt_run_wall_ms": pre_wall,
                "post_trt_run_wall_ms": post_wall,
                "trt_run_reduction_ms": (
                    pre_wall - post_wall
                ),
                "trt_run_reduction_pct":
                    reduction_percent(
                        pre_wall,
                        post_wall,
                    ),
                "pre_end_to_end_ms": pre_e2e,
                "post_end_to_end_ms": post_e2e,
                "end_to_end_reduction_ms": (
                    pre_e2e - post_e2e
                ),
                "end_to_end_reduction_pct":
                    reduction_percent(
                        pre_e2e,
                        post_e2e,
                    ),
                "pre_p95_latency_ms": pre_p95,
                "post_p95_latency_ms": post_p95,
                "p95_reduction_ms": (
                    pre_p95 - post_p95
                ),
                "p95_reduction_pct":
                    reduction_percent(
                        pre_p95,
                        post_p95,
                    ),
                "pre_fps": pre_fps,
                "post_fps": post_fps,
                "fps_gain": post_fps - pre_fps,
                "fps_gain_pct": gain_percent(
                    pre_fps,
                    post_fps,
                ),
                "pre_gpu_inference_ms": get_float(
                    pre_row,
                    "avg_gpu_inference_ms",
                ),
                "post_gpu_inference_ms": get_float(
                    post_row,
                    "avg_gpu_inference_ms",
                ),
                "pre_gpu_total_ms": get_float(
                    pre_row,
                    "avg_gpu_total_ms",
                ),
                "post_gpu_total_ms": get_float(
                    post_row,
                    "avg_gpu_total_ms",
                ),
                "pre_wall_minus_gpu_total_ms":
                    pre_host_overhead,
                "post_wall_minus_gpu_total_ms":
                    post_host_overhead,
                "host_overhead_reduction_ms": (
                    pre_host_overhead
                    - post_host_overhead
                ),
            }
        )

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=output_fields,
        )

        writer.writeheader()

        for row in comparison_rows:
            formatted_row = {}

            for key, value in row.items():
                if isinstance(value, float):
                    formatted_row[key] = (
                        f"{value:.6f}"
                    )
                else:
                    formatted_row[key] = value

            writer.writerow(formatted_row)

    print(
        f"[INFO] Comparison CSV saved to: "
        f"{args.output}"
    )

    print()
    print(
        "precision batch "
        "run_reduction_pct "
        "e2e_reduction_pct "
        "fps_gain_pct"
    )

    for row in comparison_rows:
        print(
            f"{row['precision']:>9} "
            f"{row['batch_size']:>5} "
            f"{row['trt_run_reduction_pct']:>17.2f}% "
            f"{row['end_to_end_reduction_pct']:>17.2f}% "
            f"{row['fps_gain_pct']:>11.2f}%"
        )


if __name__ == "__main__":
    main()