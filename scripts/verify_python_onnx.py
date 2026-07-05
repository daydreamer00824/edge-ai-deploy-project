import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import onnxruntime as ort

from postprocess import get_topk


def use_argparse():
    parser = argparse.ArgumentParser(
        description="验证 PyTorch ResNet18 与 ONNX Runtime 输出一致性"
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
        "--output",
        type=str,
        default="results/verify_pytorch_onnx.json",
        help="验证结果保存路径"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch 推理设备"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        help="ONNX Runtime Execution Provider"
    )

    return parser.parse_args()


def preprocess_single_image(image_path: Path, size=(224, 224)):
    """
    和你现有 image_process.py 保持一致：
    BGR -> resize -> RGB -> /255 -> normalize -> HWC to CHW -> NCHW
    """

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


def load_pytorch_model(device):
    """
    和 export_model.py 对齐：
    torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    """

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(device)

    return model


def run_pytorch(model, input_batch, device):
    input_tensor = torch.from_numpy(input_batch).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output.cpu().numpy()


def run_onnxruntime(onnx_path: Path, input_batch, provider):
    available_providers = ort.get_available_providers()

    if provider not in available_providers:
        raise RuntimeError(
            f"指定 provider 不可用: {provider}, 当前可用 provider: {available_providers}"
        )

    session = ort.InferenceSession(str(onnx_path), providers=[provider])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    output = session.run(
        [output_name],
        {input_name: input_batch}
    )[0]

    return output, input_name, output_name, input_shape, output_shape


def compare_outputs(pytorch_output, onnx_output):
    abs_error = np.abs(pytorch_output - onnx_output)

    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))

    pytorch_topk_indices, pytorch_topk_scores = get_topk(pytorch_output, k=5)
    onnx_topk_indices, onnx_topk_scores = get_topk(onnx_output, k=5)

    pytorch_topk_indices = pytorch_topk_indices[0].tolist()
    onnx_topk_indices = onnx_topk_indices[0].tolist()

    pytorch_topk_scores = pytorch_topk_scores[0].tolist()
    onnx_topk_scores = onnx_topk_scores[0].tolist()

    top1_same = pytorch_topk_indices[0] == onnx_topk_indices[0]
    top5_same_set = set(pytorch_topk_indices) == set(onnx_topk_indices)

    passed = bool(
        max_abs_error < 1e-4
        and mean_abs_error < 1e-5
        and top1_same
    )

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "pytorch_top5_indices": pytorch_topk_indices,
        "onnx_top5_indices": onnx_topk_indices,
        "pytorch_top5_scores": pytorch_topk_scores,
        "onnx_top5_scores": onnx_topk_scores,
        "top1_same": top1_same,
        "top5_same_set": top5_same_set,
        "passed": passed
    }


def main():
    args = use_argparse()

    onnx_path = Path(args.onnx)
    image_path = Path(args.image)
    output_path = Path(args.output)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("设置了 --device cuda，但当前 PyTorch CUDA 不可用")

    device = torch.device(args.device)

    logging.info("=" * 60)
    logging.info("开始 PyTorch / ONNX Runtime 一致性验证")
    logging.info(f"ONNX 模型: {onnx_path}")
    logging.info(f"测试图片: {image_path}")
    logging.info(f"PyTorch device: {device}")
    logging.info(f"ONNX Runtime provider: {args.provider}")
    logging.info("=" * 60)

    input_batch = preprocess_single_image(image_path)

    logging.info(f"输入 tensor shape: {input_batch.shape}")
    logging.info(f"输入 tensor dtype : {input_batch.dtype}")

    model = load_pytorch_model(device)
    pytorch_output = run_pytorch(model, input_batch, device)

    onnx_output, input_name, output_name, input_shape, output_shape = run_onnxruntime(
        onnx_path,
        input_batch,
        args.provider
    )

    result = compare_outputs(pytorch_output, onnx_output)

    save_result = {
        "onnx_model": str(onnx_path),
        "image": str(image_path),
        "pytorch_model": "torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)",
        "pytorch_device": str(device),
        "onnx_provider": args.provider,
        "onnx_input_name": input_name,
        "onnx_output_name": output_name,
        "onnx_input_shape": [str(x) for x in input_shape],
        "onnx_output_shape": [str(x) for x in output_shape],
        "input_tensor_shape": list(input_batch.shape),
        "input_tensor_dtype": str(input_batch.dtype),
        **result
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)

    logging.info("-" * 60)
    logging.info(f"ONNX input name : {input_name}")
    logging.info(f"ONNX output name: {output_name}")
    logging.info(f"ONNX input shape : {input_shape}")
    logging.info(f"ONNX output shape: {output_shape}")
    logging.info("-" * 60)
    logging.info(f"max_abs_error : {result['max_abs_error']:.8f}")
    logging.info(f"mean_abs_error: {result['mean_abs_error']:.8f}")
    logging.info(f"PyTorch Top5: {result['pytorch_top5_indices']}")
    logging.info(f"ONNX Top5   : {result['onnx_top5_indices']}")
    logging.info(f"Top1 same   : {result['top1_same']}")
    logging.info(f"Top5 same   : {result['top5_same_set']}")
    logging.info("-" * 60)

    if result["passed"]:
        logging.info("PASS: PyTorch 与 ONNX Runtime 输出一致")
    else:
        logging.warning("WARNING: PyTorch 与 ONNX Runtime 输出可能不一致，需要排查")

    logging.info(f"验证结果已保存: {output_path}")


if __name__ == "__main__":
    main()