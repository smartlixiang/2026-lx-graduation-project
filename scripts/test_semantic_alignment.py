"""运行语义对齐度 (SA) 计算与可视化。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP  # noqa: E402
from scoring import SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试语义对齐度 (SA) 指标并可视化结果")
    parser.add_argument("--data-root", type=str, default="./data", help="数据存放路径")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader 的并行线程数")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt",
        help="本地已训练好的 adapter 权重路径，默认指向 CIFAR-10 模型",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="计算设备，例如 cuda 或 cpu，默认跟随全局配置"
    )
    parser.add_argument("--seed", type=int, default=CONFIG.global_seed, help="随机种子")
    return parser.parse_args()


def build_loader(args: argparse.Namespace, preprocess, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=preprocess,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def load_adapter(adapter_path: Path, embed_dim: int, device: torch.device) -> AdapterMLP:
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter 权重: {adapter_path}")

    adapter = AdapterMLP(input_dim=embed_dim)
    state_dict = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()
    return adapter


def visualize_results(
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_means: list[float],
    save_dir: Path,
    class_names: list[str],
) -> None:
    font = ImageFont.load_default()

    def draw_histogram() -> None:
        hist = torch.histc(scores, bins=50, min=float(scores.min()), max=float(scores.max()))
        width, height, margin = 1000, 500, 50
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        max_count = hist.max() or 1
        bar_width = (width - 2 * margin) / len(hist)
        for i, count in enumerate(hist):
            x0 = margin + i * bar_width
            x1 = x0 + bar_width * 0.9
            bar_height = (float(count) / float(max_count)) * (height - 2 * margin)
            y0 = height - margin - bar_height
            draw.rectangle([x0, y0, x1, height - margin], fill="#4C72B0")
        draw.text((margin, margin / 2), "SA 分值分布", font=font, fill="black")
        img.save(save_dir / "sa_histogram.png")

    def draw_boxplot() -> None:
        width, height, margin = 1200, 600, 80
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        n = len(class_names)
        box_space = (width - 2 * margin) / n
        for idx in range(n):
            cls_scores = scores[labels == idx]
            if cls_scores.numel() == 0:
                continue
            q1, q2, q3 = torch.quantile(cls_scores, torch.tensor([0.25, 0.5, 0.75]))
            whisker_low, whisker_high = torch.quantile(cls_scores, torch.tensor([0.05, 0.95]))
            x_center = margin + idx * box_space + box_space / 2
            scale = (height - 2 * margin)
            score_min = float(scores.min())
            score_range = float(scores.max() - scores.min()) + 1e-6
            y = lambda v: height - margin - (float(v) - score_min) / score_range * scale
            draw.line(
                [x_center, y(whisker_low), x_center, y(whisker_high)],
                fill="#222222",
            )
            box_width = box_space * 0.4
            draw.rectangle(
                [x_center - box_width / 2, y(q3), x_center + box_width / 2, y(q1)],
                outline="#000000",
                fill="#55A868",
            )
            draw.line(
                [x_center - box_width / 2, y(q2), x_center + box_width / 2, y(q2)],
                fill="#000000",
            )
            label_y = height - margin + 10
            draw.text((x_center - box_width / 2, label_y), class_names[idx], font=font, fill="black")
        draw.text((margin, margin / 2), "各类别 SA 箱线图", font=font, fill="black")
        img.save(save_dir / "sa_boxplot.png")

    def draw_class_mean() -> None:
        width, height, margin = 1000, 500, 70
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        n = len(class_means)
        step = (width - 2 * margin) / max(n - 1, 1)
        min_v, max_v = min(class_means), max(class_means)
        scale = (height - 2 * margin) / (max_v - min_v + 1e-6)
        points = []
        for i, m in enumerate(class_means):
            x = margin + i * step
            y = height - margin - (m - min_v) * scale
            points.append((x, y))
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="#C44E52")
            draw.text((x - 10, height - margin + 10), class_names[i], font=font, fill="black")
        if len(points) > 1:
            draw.line(points, fill="#C44E52", width=2)
        draw.text((margin, margin / 2), "各类别平均 SA", font=font, fill="black")
        img.save(save_dir / "sa_class_mean.png")

    def draw_scatter() -> None:
        width, height, margin = 1000, 500, 60
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        max_points = 5000
        total = scores.numel()
        step = max(1, total // max_points)
        subset = scores[::step]
        x_scale = (width - 2 * margin) / max(subset.numel() - 1, 1)
        min_s, max_s = float(subset.min()), float(subset.max())
        y_scale = (height - 2 * margin) / (max_s - min_s + 1e-6)
        for i, val in enumerate(subset):
            x = margin + i * x_scale
            y = height - margin - (float(val) - min_s) * y_scale
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="#8172B2")
        draw.text((margin, margin / 2), "SA 散点分布 (抽样)", font=font, fill="black")
        img.save(save_dir / "sa_scatter.png")

    draw_histogram()
    draw_boxplot()
    draw_class_mean()
    draw_scatter()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = PROJECT_ROOT / "test_SA_with_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    adapter_path = (PROJECT_ROOT / args.adapter_path).resolve()

    # 读取类别名并初始化 SA
    dataset_for_names = datasets.CIFAR10(root=args.data_root, train=True, download=True)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )

    adapter = load_adapter(adapter_path=adapter_path, embed_dim=sa_metric.extractor.embed_dim, device=device)

    # 使用 CLIP 的官方预处理重新构建 dataloader
    loader = build_loader(args, preprocess=sa_metric.extractor.preprocess, device=device)

    start = time.perf_counter()
    result = sa_metric.score_dataset(loader, adapter=adapter)
    elapsed = time.perf_counter() - start

    scores = result.scores
    class_means = result.classwise_mean()

    torch.save(scores, output_dir / "sa_scores.pt")
    torch.save(result.labels, output_dir / "sa_labels.pt")

    summary = {
        "num_samples": int(scores.numel()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "class_means": {name: float(m) for name, m in zip(class_names, class_means)},
        "elapsed_seconds": elapsed,
        "adapter_path": str(adapter_path),
        "device": str(device),
    }
    with open(output_dir / "sa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    visualize_results(scores, result.labels, class_means, output_dir, class_names)

    print(f"全部样本 SA 计算完成，用时 {elapsed:.2f} 秒。")
    print(f"使用的 Adapter 权重: {adapter_path}")
    print(f"结果文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
