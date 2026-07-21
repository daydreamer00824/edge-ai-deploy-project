#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$HOME/projects/edge_ai_deploy_project"

TRT_BENCHMARK="$PROJECT_ROOT/cpp/build/cpp_trt_benchmark"

IMAGE_DIR="$HOME/projects/python-onnx-demo/input"
LABELS="$PROJECT_ROOT/labels/imagenet_classes.txt"

FP32_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp32.engine"
FP16_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp16.engine"

OUTPUT_DIR="$PROJECT_ROOT/results/benchmark/trt_buffer_reuse"
CSV_OUTPUT="$OUTPUT_DIR/trt_buffer_reuse_benchmark.csv"

WARMUP=3
REPEAT=10
BATCH_SIZES=(1 8 32)

mkdir -p "$OUTPUT_DIR"

rm -f "$CSV_OUTPUT"
rm -f "$OUTPUT_DIR"/*.log

run_trt() {
    local engine_path="$1"
    local backend="$2"
    local batch_size="$3"
    local log_path="$OUTPUT_DIR/${backend}_batch${batch_size}.log"

    echo
    echo "=================================================="
    echo "Running ${backend}, batch=${batch_size}"
    echo "=================================================="

    "$TRT_BENCHMARK" \
        "$engine_path" \
        "$IMAGE_DIR" \
        "$LABELS" \
        "$batch_size" \
        "$WARMUP" \
        "$REPEAT" \
        "$backend" \
        "$CSV_OUTPUT" \
        2>&1 | tee "$log_path"
}

for batch_size in "${BATCH_SIZES[@]}"; do
    run_trt \
        "$FP32_ENGINE" \
        "TensorRT_FP32_BufferReuse" \
        "$batch_size"
done

for batch_size in "${BATCH_SIZES[@]}"; do
    run_trt \
        "$FP16_ENGINE" \
        "TensorRT_FP16_BufferReuse" \
        "$batch_size"
done

echo
echo "=================================================="
echo "TensorRT buffer-reuse benchmark completed"
echo "CSV: $CSV_OUTPUT"
echo "=================================================="

column -s, -t "$CSV_OUTPUT"
