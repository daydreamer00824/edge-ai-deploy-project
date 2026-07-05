from pathlib import Path
from torchvision.models import ResNet18_Weights


def generate_imagenet_labels(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    label_path = save_dir / "imagenet_classes.txt"

    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]

    with open(label_path, "w", encoding="utf-8") as f:
        for name in categories:
            f.write(name + "\n")

    print(f"ImageNet 类别文件已生成: {label_path}")
    print(f"类别数量: {len(categories)}")

    return label_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    save_dir = project_root / "labels"
    generate_imagenet_labels(save_dir)