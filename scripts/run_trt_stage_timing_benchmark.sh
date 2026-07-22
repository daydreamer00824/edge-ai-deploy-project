#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$HOME/projects/edge_ai_deploy_project"

TRT_BENCHMARK="$PROJECT_ROOT/cpp/build/cpp_trt_benchmark"

IMAGE_DIR="$HOME/projects/python-onnx-demo/input"
LABELS="$PROJECT_ROOT/labels/imagenet_classes.txt"

FP32_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp32.engine"
FP16_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp16.engine"

OUTPUT_DIR="$PROJECT_ROOT/results/benchmark/trt_stage_timing"

COMMON_CSV="$OUTPUT_DIR/trt_stage_timing_common.csv"
STAGE_CSV="$OUTPUT_DIR/trt_stage_timing.csv"

WARMUP=3
REPEAT=10

BATCH_SIZES=(1 8 32)

mkdir -p "$OUTPUT_DIR"

rm -f "$COMMON_CSV"
rm -f "$STAGE_CSV"
rm -f "$OUTPUT_DIR"/*.log

run_benchmark() {
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
        "$COMMON_CSV" \
        "$STAGE_CSV" \
        2>&1 | tee "$log_path"
}

for batch_size in "${BATCH_SIZES[@]}"; do
    run_benchmark \
        "$FP32_ENGINE" \
        "TensorRT_FP32_StageTiming" \
        "$batch_size"
done

for batch_size in "${BATCH_SIZES[@]}"; do
    run_benchmark \
        "$FP16_ENGINE" \
        "TensorRT_FP16_StageTiming" \
        "$batch_size"
done

echo
echo "=================================================="
echo "TensorRT stage timing benchmark completed"
echo "Common CSV: $COMMON_CSV"
echo "Stage CSV : $STAGE_CSV"
echo "Logs      : $OUTPUT_DIR"
echo "=================================================="

column -s, -t "$STAGE_CSV"
