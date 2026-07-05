import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def use_argparse():
    parser = argparse.ArgumentParser(description="绘制 benchmark 可视化曲线")
    parser.add_argument(
        "--summary",
        type=str,
        default="/home/daydreamer/Desktop/study/python-onnx-demo/output/benchmark_summary.csv",
        help="benchmark_summary.csv 路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/daydreamer/Desktop/study/python-onnx-demo/output",
        help="图片输出目录"
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"benchmark summary 文件不存在: {summary_path}")

    df = pd.read_csv(summary_path)

    required_columns = [
        "provider",
        "batch_size",
        "avg_infer_ms_mean",
        "avg_infer_ms_std",
        "throughput_mean",
        "throughput_std",
        "best_by_throughput"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"benchmark summary 缺少字段: {missing_columns}")

    df = df.sort_values("batch_size")

    return df


def plot_throughput(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = df["provider"].iloc[0]
    save_path = output_dir / "benchmark_throughput.png"

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df["batch_size"],
        df["throughput_mean"],
        yerr=df["throughput_std"],
        marker="o",
        capsize=4
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (image/s)")
    plt.title(f"Batch Size vs Throughput ({provider})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def plot_avg_infer_ms(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = df["provider"].iloc[0]
    save_path = output_dir / "benchmark_avg_infer_ms.png"

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df["batch_size"],
        df["avg_infer_ms_mean"],
        yerr=df["avg_infer_ms_std"],
        marker="o",
        capsize=4
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Average Inference Time (ms/image)")
    plt.title(f"Batch Size vs Avg Inference Time ({provider})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def main():
    args = use_argparse()

    summary_path = Path(args.summary)
    output_dir = Path(args.output)

    try:
        df = load_summary(summary_path)
        throughput_path = plot_throughput(df, output_dir)
        avg_infer_path = plot_avg_infer_ms(df, output_dir)

        print(f"throughput 图已保存: {throughput_path}")
        print(f"avg infer ms 图已保存: {avg_infer_path}")

    except Exception as e:
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    main()