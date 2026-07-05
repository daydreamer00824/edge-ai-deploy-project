import csv
from pathlib import Path
import logging

def save_result(results, output_dir : Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "prediction_results.csv"
    fieldnames = [
        "image_name",
        "image_path",
        "top1_index",
        "top1_label",
        "top1_score",
        "top5_indices",
        "top5_labels",
        "top5_scores"
        ]

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logging.info(f"推理结果已保存: {result_path}")
    return result_path
    