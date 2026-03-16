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
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402

DATASET_REGISTRY = {
    CIFAR10: datasets.CIFAR10,
    CIFAR100: datasets.CIFAR100,
}


class TinyImageNetTrain(datasets.ImageFolder):
    def __init__(self, root: str, train: bool, download: bool, transform):
        _ = (train, download)
        train_root = Path(root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        super().__init__(root=str(train_root), transform=transform)


DATASET_REGISTRY[TINY_IMAGENET] = TinyImageNetTrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 CIFAR-10/100 learned 与 naive 权重在归一化综合分上的重叠分布图。"
    )
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--output-dir", type=str, default="result_scoring_debug", help="输出目录")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument("--adapter-seed", type=int, default=22, help="adapter 种子")
    parser.add_argument(
        "--weight-seed",
        type=str,
        default="22",
        help="学习权重随机种子（weights/scoring_weights.json 键名）",
    )
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--bins", type=int, default=80, help="直方图 bins 数量")
    return parser.parse_args()


def load_scoring_weights(path: Path, dataset_name: str, group: str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    dataset_data = all_data.get(dataset_name)
    if not isinstance(dataset_data, dict):
        raise KeyError(f"weights 文件中不存在数据集: {dataset_name}")
    selected = dataset_data.get(group)
    if not isinstance(selected, dict):
        raise KeyError(f"weights 文件中不存在权重组: {group}")

    required = ("dds", "div", "sa")
    weights = {key: float(selected[key]) for key in required}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("综合权重和必须为正数。")
    if not np.isclose(total, 1.0):
        weights = {k: v / total for k, v in weights.items()}
    return weights


def build_score_loader(
    dataset_name: str,
    preprocess,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(root=data_root, train=True, download=True, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    if np.isclose(max_val, min_val):
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - min_val) / (max_val - min_val)


def _save_overlap_hist(
    learned_scores: np.ndarray,
    naive_scores: np.ndarray,
    dataset_name: str,
    weight_seed: str,
    bins: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        learned_scores,
        bins=bins,
        range=(0.0, 1.0),
        color="#4C72B0",
        alpha=0.55,
        label=f"Learned (seed {weight_seed})",
    )
    ax.hist(
        naive_scores,
        bins=bins,
        range=(0.0, 1.0),
        color="#DD8452",
        alpha=0.55,
        label="Naive (uniform 1/3)",
    )
    ax.set_title(f"{dataset_name.upper()} Normalized Total Score Distribution")
    ax.set_xlabel("Normalized Total Score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlap histogram to: {output_path}")


def _compute_dataset_scores(
    dataset_name: str,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> None:
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset_for_names = dataset_cls(root=args.data_root, train=True, download=True, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=dataset_name,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.adapter_seed,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()
    print(f"[{dataset_name}] Resolved adapter image path: {adapter_paths['image_path']}")
    print(f"[{dataset_name}] Resolved adapter text path: {adapter_paths['text_path']}")

    batch_size = 128
    num_workers = 4
    dds_loader = build_score_loader(dataset_name, dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)
    div_loader = build_score_loader(dataset_name, div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)
    sa_loader = build_score_loader(dataset_name, sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)

    num_samples = len(dataset_for_names)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc=f"[{dataset_name}] Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc=f"[{dataset_name}] Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc=f"[{dataset_name}] Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {
            "sa": np.asarray(sa_scores_local),
            "div": np.asarray(div_scores_local),
            "dds": np.asarray(dds_scores_local),
            "labels": np.asarray(dataset_for_names.targets),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=dataset_name,
        seed=args.adapter_seed,
        clip_model=args.clip_model,
        adapter_image_path=str(adapter_paths["image_path"]),
        adapter_text_path=str(adapter_paths["text_path"]),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    sa_scores = static_scores["sa"]
    div_scores = static_scores["div"]
    dds_scores = static_scores["dds"]

    if not (len(sa_scores) == len(div_scores) == len(dds_scores)):
        raise RuntimeError(f"[{dataset_name}] 三个指标样本数不一致，无法计算综合分。")

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    learned_weights = load_scoring_weights(weights_path, dataset_name, str(args.weight_seed))
    naive_weights = {"sa": 1.0 / 3.0, "div": 1.0 / 3.0, "dds": 1.0 / 3.0}

    total_scores_learned = (
        learned_weights["sa"] * sa_scores
        + learned_weights["div"] * div_scores
        + learned_weights["dds"] * dds_scores
    )
    total_scores_naive = (
        naive_weights["sa"] * sa_scores + naive_weights["div"] * div_scores + naive_weights["dds"] * dds_scores
    )

    total_scores_learned_norm = _minmax_normalize(np.asarray(total_scores_learned, dtype=np.float64))
    total_scores_naive_norm = _minmax_normalize(np.asarray(total_scores_naive, dtype=np.float64))

    output_path = output_dir / f"{dataset_name}_total_hist_overlap_minmax_{args.weight_seed}.png"
    _save_overlap_hist(
        total_scores_learned_norm,
        total_scores_naive_norm,
        dataset_name=dataset_name,
        weight_seed=str(args.weight_seed),
        bins=args.bins,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.adapter_seed)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in (CIFAR10, CIFAR100):
        _compute_dataset_scores(dataset_name, args, device, output_dir)


if __name__ == "__main__":
    main()
