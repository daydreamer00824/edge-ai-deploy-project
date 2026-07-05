import onnx
from pathlib import Path
import logging

def check_onnx(model_path : Path):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not exist:{model_path}")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    logging.info(f"ONNX model check passed:{model_path}")
