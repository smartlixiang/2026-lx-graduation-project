from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10  # noqa: E402
from model.adapter import AdapterMLP  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw histograms of CIFAR-10 static scoring distributions."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_scoring_debug",
        help="输出目录",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt",
        help="adapter 权重路径",
    )
    parser.add_argument(
        "--weight-seed",
        type=str,
        default="22",
        help="学习权重的随机种子（scoring_weights.json 键名）",
    )
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--bins", type=int, default=80, help="直方图 bins 数量")
    return parser.parse_args()


def ensure_scoring_weights(path: Path, dataset_name: str) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    updated = False
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
    dataset_entry = data.get(dataset_name)
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        updated = True
    naive = dataset_entry.get("naive")
    if not isinstance(naive, dict):
        naive = {}
        updated = True
    for key in ("dds", "div", "sa"):
        if key not in naive:
            naive[key] = 1.0
            updated = True
    dataset_entry["naive"] = naive
    data[dataset_name] = dataset_entry
    if updated or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        group_name: group
        for group_name, group in dataset_entry.items()
        if isinstance(group, dict)
    }


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    group: str,
) -> dict[str, float]:
    selected = all_weights.get(group)
    if selected is None:
        raise KeyError(f"未找到权重组: {group}")
    if not isinstance(selected, dict):
        raise ValueError(f"权重组 {group} 格式不正确。")
    required = {"dds", "div", "sa"}
    missing = required - selected.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"权重组 {group} 缺少必要键: {missing_str}")
    weights: dict[str, float] = {}
    for key in sorted(required):
        value = selected[key]
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group} 的 {key} 无法转换为 float。") from exc
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"权重组 {group} 的权重和必须为正数。")
    if not np.isclose(total, 1.0):
        weights = {key: value / total for key, value in weights.items()}
    return weights


def build_score_loader(
    preprocess, data_root: str, device: torch.device, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def main() -> None:
    args = parse_args()
    set_seed(22)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, CIFAR10)
    naive_weights = load_scoring_weights(all_weights, "naive")
    learned_weights = load_scoring_weights(all_weights, str(args.weight_seed))

    dataset_for_names = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=None
    )
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter 权重: {adapter_path}")
    adapter = AdapterMLP(input_dim=dds_metric.extractor.embed_dim)
    state_dict = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()

    batch_size = 128
    num_workers = 8

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )

    dds_scores = dds_metric.score_dataset(dds_loader, adapter=adapter).scores
    div_scores = div_metric.score_dataset(div_loader, adapter=adapter).scores
    sa_scores = sa_metric.score_dataset(sa_loader, adapter=adapter).scores

    if not (len(dds_scores) == len(div_scores) == len(sa_scores)):
        raise RuntimeError("三个指标的样本数不一致，无法合并。")

    total_naive = (
        naive_weights["dds"] * dds_scores
        + naive_weights["div"] * div_scores
        + naive_weights["sa"] * sa_scores
    )
    total_learned = (
        learned_weights["dds"] * dds_scores
        + learned_weights["div"] * div_scores
        + learned_weights["sa"] * sa_scores
    )

    score_sets = {
        "SA": np.asarray(sa_scores),
        "Div": np.asarray(div_scores),
        "DDS": np.asarray(dds_scores),
        "Naive Total": np.asarray(total_naive),
        f"Learned Total (seed {args.weight_seed})": np.asarray(total_learned),
    }

    for name, scores in score_sets.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores, bins=args.bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        output_path = output_dir / f"cifar10_{safe_name}_hist_seed_{args.weight_seed}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to: {output_path}")


if __name__ == "__main__":
    main()
