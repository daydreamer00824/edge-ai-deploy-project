import cv2
from pathlib import Path
import numpy as np
import logging

def collect_image(input_dir : Path, output_dir : Path, extensions, size):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    image = []
    image_path = []  #!#!

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    if not input_dir.exists() or not input_dir.is_dir():
        # logging.error("输入文件不存在或不是目录")
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")       #!#!

    for i in input_dir.iterdir():
        if i.is_file() and i.suffix.lower() in extensions:
            img = cv2.imread(str(i))
            if img is None:
                logging.warning(f"图片读取失败:{i}")
                continue        #!#!
            img_reise = cv2.resize(img, size)
            save_path = output_dir / i.name
            cv2.imwrite(str(save_path), img_reise)
            img_color = cv2.cvtColor(img_reise, cv2.COLOR_BGR2RGB)
            img_float = img_color.astype(np.float32) / 255.0
            img_norm = (img_float - mean) / std
            img_c = np.transpose(img_norm, (2, 0, 1))
            image.append(img_c)
            image_path.append(i)  #!#!
        else:
            logging.warning(f"{i}不是图片")

    return image, image_path
