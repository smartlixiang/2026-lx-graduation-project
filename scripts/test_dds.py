"""运行类内难度方向得分 (DDS) 计算并输出可视化对比图。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP  # noqa: E402
from scoring import DifficultyDirection  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试类内难度方向得分 (DDS) 并输出可视化结果")
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
        "--k",
        type=float,
        default=10,
        help="低方差方向数量。整数表示固定方向数，小数表示比例。",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="计算设备，例如 cuda 或 cpu，默认跟随全局配置"
    )
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


def build_raw_dataset(args: argparse.Namespace) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=None)


def save_class_top_bottom(
    scores: torch.Tensor,
    labels: torch.Tensor,
    raw_dataset: datasets.CIFAR10,
    class_names: list[str],
    output_dir: Path,
    topk: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        if not class_mask.any():
            continue

        class_indices = torch.where(class_mask)[0]
        class_scores = scores[class_indices]
        order = torch.argsort(class_scores)

        bottom_indices = class_indices[order[:topk]].tolist()
        top_indices = class_indices[order[-topk:]].tolist()

        images = []
        for idx in top_indices + bottom_indices:
            img = raw_dataset[int(idx)][0]
            images.append(transforms.ToTensor()(img))

        grid = make_grid(images, nrow=topk, padding=2, pad_value=1.0)
        save_path = output_dir / f"{class_idx:02d}_{class_name}_dds_top_bottom.png"
        save_image(grid, save_path)


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "test_DDS_result"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    adapter_path = (PROJECT_ROOT / args.adapter_path).resolve()

    dataset_for_names = datasets.CIFAR10(root=args.data_root, train=True, download=True)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    dds_metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        k=args.k,
        device=device,
    )

    adapter = load_adapter(
        adapter_path=adapter_path, embed_dim=dds_metric.extractor.embed_dim, device=device
    )
    loader = build_loader(args, preprocess=dds_metric.extractor.preprocess, device=device)

    start = time.perf_counter()
    result = dds_metric.score_dataset(loader, adapter=adapter)
    elapsed = time.perf_counter() - start

    torch.save(result.scores, output_dir / "dds_scores.pt")
    torch.save(result.labels, output_dir / "dds_labels.pt")

    summary = {
        "num_samples": int(result.scores.numel()),
        "score_mean": float(result.scores.mean()),
        "score_std": float(result.scores.std()),
        "score_min": float(result.scores.min()),
        "score_max": float(result.scores.max()),
        "class_means": {
            name: float(m) for name, m in zip(class_names, result.classwise_mean())
        },
        "elapsed_seconds": elapsed,
        "adapter_path": str(adapter_path),
        "device": str(device),
        "k_input": args.k,
        "k_resolved": result.resolved_k,
    }
    with open(output_dir / "dds_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    raw_dataset = build_raw_dataset(args)
    save_class_top_bottom(
        result.scores, result.labels, raw_dataset, class_names, output_dir, topk=4
    )

    print(f"全部样本 DDS 计算完成，用时 {elapsed:.2f} 秒。")
    print(f"使用的 Adapter 权重: {adapter_path}")
    print(f"结果文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
