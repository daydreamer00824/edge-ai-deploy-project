import argparse
import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def use_argparse():
    parser = argparse.ArgumentParser(
        description="验证 Python ORT 与 C++ ORT 输出一致性"
    )

    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="ONNX 模型路径，例如: models/model.onnx"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="测试图片路径，例如: data/test.jpg"
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="ImageNet 标签文件路径，例如: labels/imagenet_classes.txt"
    )

    parser.add_argument(
        "--cpp-bin",
        type=str,
        required=True,
        help="C++ verify_cpp_ort 可执行文件路径，例如: cpp/build/verify_cpp_ort"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/verify_python_cpp_ort",
        help="验证结果输出目录"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="Python ONNX Runtime Provider"
    )

    return parser.parse_args()


def preprocess_single_image(image_path: Path, size=(224, 224)) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"图片读取失败: {image_path}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_resize = cv2.resize(img, size)
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - mean) / std
    img_chw = np.transpose(img_norm, (2, 0, 1))

    input_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
    input_batch = np.ascontiguousarray(input_batch)

    return input_batch


def run_python_ort(onnx_path: Path, input_batch: np.ndarray, provider: str):
    available_providers = ort.get_available_providers()

    if provider not in available_providers:
        raise RuntimeError(
            f"指定 provider 不可用: {provider}, 当前可用 provider: {available_providers}"
        )

    session = ort.InferenceSession(str(onnx_path), providers=[provider])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output = session.run(
        [output_name],
        {input_name: input_batch}
    )[0]

    return output, input_name, output_name


def run_cpp_verify(
    cpp_bin: Path,
    onnx_path: Path,
    image_path: Path,
    label_path: Path,
    output_dir: Path
):
    if not cpp_bin.exists():
        raise FileNotFoundError(f"C++ 可执行文件不存在: {cpp_bin}")

    cmd = [
        str(cpp_bin),
        str(onnx_path),
        str(image_path),
        str(label_path),
        str(output_dir)
    ]

    logging.info("开始调用 C++ verify_cpp_ort")
    logging.info(" ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    cpp_stdout_path = output_dir / "cpp_stdout.log"
    cpp_stderr_path = output_dir / "cpp_stderr.log"

    cpp_stdout_path.write_text(result.stdout, encoding="utf-8")
    cpp_stderr_path.write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"C++ verify_cpp_ort 执行失败，returncode={result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )

    logging.info("C++ verify_cpp_ort 执行成功")


def load_cpp_logits(logits_path: Path) -> np.ndarray:
    if not logits_path.exists():
        raise FileNotFoundError(f"C++ logits 文件不存在: {logits_path}")

    cpp_logits = np.loadtxt(str(logits_path), delimiter=",", dtype=np.float32)

    if cpp_logits.ndim == 1:
        cpp_logits = np.expand_dims(cpp_logits, axis=0)

    return cpp_logits


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def get_topk(logits: np.ndarray, k: int = 5):
    probs = softmax(logits)

    topk_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
    topk_scores = np.take_along_axis(probs, topk_indices, axis=1)

    return topk_indices, topk_scores


def compare_outputs(python_logits: np.ndarray, cpp_logits: np.ndarray):
    if python_logits.shape != cpp_logits.shape:
        raise ValueError(
            f"Python / C++ 输出 shape 不一致: "
            f"python={python_logits.shape}, cpp={cpp_logits.shape}"
        )

    abs_error = np.abs(python_logits - cpp_logits)

    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))

    python_topk_indices, python_topk_scores = get_topk(python_logits, k=5)
    cpp_topk_indices, cpp_topk_scores = get_topk(cpp_logits, k=5)

    python_topk_indices = python_topk_indices[0].tolist()
    cpp_topk_indices = cpp_topk_indices[0].tolist()

    python_topk_scores = python_topk_scores[0].tolist()
    cpp_topk_scores = cpp_topk_scores[0].tolist()

    top1_same = python_topk_indices[0] == cpp_topk_indices[0]
    top5_same_set = set(python_topk_indices) == set(cpp_topk_indices)

    passed = bool(
        max_abs_error < 1e-4
        and mean_abs_error < 1e-5
        and top1_same
    )

    return {
        "python_shape": list(python_logits.shape),
        "cpp_shape": list(cpp_logits.shape),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "python_top5_indices": python_topk_indices,
        "cpp_top5_indices": cpp_topk_indices,
        "python_top5_scores": python_topk_scores,
        "cpp_top5_scores": cpp_topk_scores,
        "top1_same": top1_same,
        "top5_same_set": top5_same_set,
        "passed": passed
    }


def main():
    args = use_argparse()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    onnx_path = Path(args.onnx)
    image_path = Path(args.image)
    label_path = Path(args.labels)
    cpp_bin = Path(args.cpp_bin)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    logging.info("=" * 60)
    logging.info("开始 Python ORT / C++ ORT 一致性验证")
    logging.info(f"ONNX 模型: {onnx_path}")
    logging.info(f"测试图片: {image_path}")
    logging.info(f"标签文件: {label_path}")
    logging.info(f"C++ 可执行文件: {cpp_bin}")
    logging.info(f"Python ORT Provider: {args.provider}")
    logging.info("=" * 60)

    run_cpp_verify(
        cpp_bin=cpp_bin,
        onnx_path=onnx_path,
        image_path=image_path,
        label_path=label_path,
        output_dir=output_dir
    )

    cpp_logits_path = output_dir / "cpp_logits.csv"
    cpp_logits = load_cpp_logits(cpp_logits_path)

    input_batch = preprocess_single_image(image_path)

    logging.info(f"Python input shape: {input_batch.shape}")
    logging.info(f"Python input dtype : {input_batch.dtype}")

    python_logits, input_name, output_name = run_python_ort(
        onnx_path=onnx_path,
        input_batch=input_batch,
        provider=args.provider
    )

    result = compare_outputs(python_logits, cpp_logits)

    save_result = {
        "onnx_model": str(onnx_path),
        "image": str(image_path),
        "labels": str(label_path),
        "cpp_bin": str(cpp_bin),
        "python_provider": args.provider,
        "onnx_input_name": input_name,
        "onnx_output_name": output_name,
        **result
    }

    result_path = output_dir / "verify_python_cpp_ort.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)

    logging.info("-" * 60)
    logging.info(f"Python output shape: {result['python_shape']}")
    logging.info(f"C++ output shape   : {result['cpp_shape']}")
    logging.info(f"max_abs_error     : {result['max_abs_error']:.10f}")
    logging.info(f"mean_abs_error    : {result['mean_abs_error']:.10f}")
    logging.info(f"Python Top5       : {result['python_top5_indices']}")
    logging.info(f"C++ Top5          : {result['cpp_top5_indices']}")
    logging.info(f"Top1 same         : {result['top1_same']}")
    logging.info(f"Top5 same set     : {result['top5_same_set']}")
    logging.info("-" * 60)

    if result["passed"]:
        logging.info("PASS: Python ORT 与 C++ ORT 输出一致")
    else:
        logging.warning("WARNING: Python ORT 与 C++ ORT 输出可能不一致，需要排查")

    logging.info(f"验证结果已保存: {result_path}")


if __name__ == "__main__":
    main()