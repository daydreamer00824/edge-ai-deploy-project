from pathlib import Path
import logging

def load_labels(label_path : Path):
    if not label_path.exists():
        raise ValueError(f"类别文件不存在: {label_path}")
    
    labels = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            label = line.strip()
            if label:
                labels.append(label)
    if len(labels) == 0:
        raise ValueError(f"类别文件为空: {label_path}")
    
    logging.info(f"类别文件加载成功，共 {len(labels)} 个类别: {label_path}")
    return labels

def index_to_label(index : int, labels):
    if index < 0 or index>= len(labels):
        return f"unknow{index}"
    
    return labels[index]

def indices_to_labels(indices : int, labels):
    return [index_to_label(int(i), labels) for i in indices]