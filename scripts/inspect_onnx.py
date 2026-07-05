import onnxruntime as ort
from pathlib import Path
import logging

def inspect_onnx(onnx_path : Path):
    session = ort.InferenceSession(onnx_path)

    for x in session.get_inputs():
        logging.info(f"name:{x.name}")
        logging.info(f"shape:{x.shape}")
        logging.info(f"type:{x.type}")
        logging.info("*" * 30)

    for y in session.get_outputs():
        logging.info(f"name:{y.name}")
        logging.info(f"shape:{y.shape}")
        logging.info(f"type:{y.type}")
        logging.info("*" * 30)