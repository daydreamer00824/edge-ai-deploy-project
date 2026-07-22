#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$HOME/projects/edge_ai_deploy_project"

ORT_BENCHMARK="$PROJECT_ROOT/cpp/build/cpp_ort_benchmark"
TRT_BENCHMARK="$PROJECT_ROOT/cpp/build/cpp_trt_benchmark"

MODEL="$HOME/projects/python-onnx-demo/model/model.onnx"
IMAGE_DIR="$HOME/projects/python-onnx-demo/input"
LABELS="$PROJECT_ROOT/labels/imagenet_classes.txt"

FP32_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp32.engine"
FP16_ENGINE="$HOME/projects/python-onnx-demo/tensorrt/engines/model_fp16.engine"

OUTPUT_DIR="$PROJECT_ROOT/results/verify/resnet18_stage_acceptance"
CSV_OUTPUT="$OUTPUT_DIR/four_backend_recheck.csv"

WARMUP=3
REPEAT=10
BATCH_SIZES=(1 8 32)

mkdir -p "$OUTPUT_DIR"

rm -f "$CSV_OUTPUT"
rm -f "$OUTPUT_DIR"/ORT_*.log
rm -f "$OUTPUT_DIR"/TensorRT_*.log

check_file() {
    local file_path="$1"
    local description="$2"

    if [[ ! -f "$file_path" ]]; then
        echo "[ERROR] ${description} not found: ${file_path}" >&2
        exit 1
    fi
}

check_directory() {
    local directory_path="$1"
    local description="$2"

    if [[ ! -d "$directory_path" ]]; then
        echo "[ERROR] ${description} not found: ${directory_path}" >&2
        exit 1
    fi
}

check_file "$ORT_BENCHMARK" "ORT benchmark executable"
check_file "$TRT_BENCHMARK" "TensorRT benchmark executable"
check_file "$MODEL" "ONNX model"
check_file "$LABELS" "ImageNet label file"
check_file "$FP32_ENGINE" "TensorRT FP32 engine"
check_file "$FP16_ENGINE" "TensorRT FP16 engine"
check_directory "$IMAGE_DIR" "Input image directory"

run_ort() {
    local provider="$1"
    local backend="$2"
    local batch_size="$3"

    local log_path="$OUTPUT_DIR/${backend}_batch${batch_size}.log"

    echo
    echo "=================================================="
    echo "Running ${backend}, batch=${batch_size}"
    echo "=================================================="

    "$ORT_BENCHMARK" \
        "$MODEL" \
        "$IMAGE_DIR" \
        "$LABELS" \
        "$batch_size" \
        "$WARMUP" \
        "$REPEAT" \
        "$CSV_OUTPUT" \
        --provider "$provider" \
        2>&1 | tee "$log_path"
}

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
    run_ort "cpu" "ORT_CPU" "$batch_size"
done

for batch_size in "${BATCH_SIZES[@]}"; do
    run_ort "cuda" "ORT_CUDA" "$batch_size"
done

for batch_size in "${BATCH_SIZES[@]}"; do
    run_trt "$FP32_ENGINE" "TensorRT_FP32" "$batch_size"
done

for batch_size in "${BATCH_SIZES[@]}"; do
    run_trt "$FP16_ENGINE" "TensorRT_FP16" "$batch_size"
done

echo
echo "=================================================="
echo "ResNet18 stage acceptance benchmark completed"
echo "CSV : $CSV_OUTPUT"
echo "Logs: $OUTPUT_DIR"
echo "=================================================="

column -s, -t "$CSV_OUTPUT"