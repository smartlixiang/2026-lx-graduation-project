"""Visualize high-score vs low-score CIFAR100 samples under learned static weights.

This script computes SA/Div/DDS static metrics over CIFAR100 train split,
optionally loads metric caches, then aggregates them with learned weights and
plots paired high/low scoring samples for selected classes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.global_config import CONFIG
from utils.seed import set_seed
from utils.static_score_cache import get_or_compute_static_scores

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_CLASS_IDS = [0, 20, 40, 60, 80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CIFAR100 static metrics with learned weights and visualize class-wise extremes.",
    )
    parser.add_argument("--data-root", type=str, default=str(CONFIG.data_root))
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100"])
    parser.add_argument("--seed", type=int, default=CONFIG.global_seed, help="Adapter/weight seed.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=CONFIG.default_batch_size)
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument("--dds-k", type=int, default=5)
    parser.add_argument("--dds-eigval-lower-bound", type=float, default=0.02)
    parser.add_argument("--dds-eigval-upper-bound", type=float, default=0.2)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.8)
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    parser.add_argument("--weights-json", type=str, default="weights/scoring_weights.json")
    parser.add_argument("--static-cache-root", type=str, default="static_scores")
    parser.add_argument("--output", type=str, default="picture/cifar100_static_high_low_pairs.png")
    return parser.parse_args()


def _make_loader(transform, data_root: str, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _load_learned_weights(weights_json_path: Path, dataset_name: str, seed: int) -> tuple[float, float, float, float]:
    if not weights_json_path.is_file():
        raise FileNotFoundError(
            f"Learned weights json not found: {weights_json_path}. "
            "Run learn_scoring_weights.py first or pass --weights-json."
        )

    with weights_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entry: dict[str, object] | None = None
    if isinstance(payload, dict):
        dataset_node = payload.get(dataset_name)
        if isinstance(dataset_node, dict):
            by_seed = dataset_node.get(str(seed))
            if isinstance(by_seed, dict):
                entry = by_seed
        if entry is None and all(k in payload for k in ("sa", "div", "dds")):
            entry = payload

    if entry is None:
        raise KeyError(
            f"Cannot find learned weights for dataset='{dataset_name}', seed={seed} in {weights_json_path}."
        )

    return (
        float(entry["sa"]),
        float(entry["div"]),
        float(entry["dds"]),
        float(entry.get("bias", 0.0)),
    )


def _stable_argsort_desc(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(values.shape[0])
    rng.shuffle(order)
    shuffled = values[order]
    sorted_in_shuffled = np.argsort(-shuffled, kind="mergesort")
    return order[sorted_in_shuffled]


def _stable_argsort_asc(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(values.shape[0])
    rng.shuffle(order)
    shuffled = values[order]
    sorted_in_shuffled = np.argsort(shuffled, kind="mergesort")
    return order[sorted_in_shuffled]


def _select_extreme_indices_per_class(
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
    rng: np.random.Generator,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    high: dict[int, list[int]] = {}
    low: dict[int, list[int]] = {}

    for class_id in class_ids:
        mask = labels == class_id
        class_indices = np.where(mask)[0]
        if class_indices.shape[0] < 3:
            raise ValueError(f"Class {class_id} has fewer than 3 samples.")

        class_scores = scores[class_indices]
        desc_order = _stable_argsort_desc(class_scores, rng)
        asc_order = _stable_argsort_asc(class_scores, rng)

        high[class_id] = class_indices[desc_order[:3]].tolist()
        low[class_id] = class_indices[asc_order[:3]].tolist()
    return high, low


def _plot_pairs(
    raw_dataset: datasets.CIFAR100,
    class_names: list[str],
    final_scores: np.ndarray,
    high_pairs: dict[int, list[int]],
    low_pairs: dict[int, list[int]],
    output_path: Path,
) -> None:
    rows = len(TARGET_CLASS_IDS)

    # 整体宽度收紧，避免右侧多余留白
    fig = plt.figure(figsize=(14.2, 12.6), dpi=200)

    # 重新调整网格：
    # - 左侧标签列略窄，但给文字更合理的锚点
    # - 中间分隔列缩小
    # - 整体子图间距更紧凑
    gs = fig.add_gridspec(
        rows,
        8,
        width_ratios=[0.82, 1.0, 1.0, 1.0, 0.12, 1.0, 1.0, 1.0],
        wspace=0.10,
        hspace=0.18,
    )

    high_axes_first_row = []
    low_axes_first_row = []

    for row, class_id in enumerate(TARGET_CLASS_IDS):
        # 左侧类别标签
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.92,          # 向右靠近图片组，但不要太贴边
            0.50,          # 严格垂直居中
            class_names[class_id],
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        # High-score group
        for col, sample_index in enumerate(high_pairs[class_id]):
            ax = fig.add_subplot(gs[row, 1 + col])
            if row == 0:
                high_axes_first_row.append(ax)

            image, _ = raw_dataset[sample_index]
            ax.imshow(np.asarray(image))
            ax.axis("off")
            ax.set_title(
                f"{final_scores[sample_index]:.3f}",
                fontsize=9,
                pad=6,
            )

        # Low-score group
        for col, sample_index in enumerate(low_pairs[class_id]):
            ax = fig.add_subplot(gs[row, 5 + col])
            if row == 0:
                low_axes_first_row.append(ax)

            image, _ = raw_dataset[sample_index]
            ax.imshow(np.asarray(image))
            ax.axis("off")
            ax.set_title(
                f"{final_scores[sample_index]:.3f}",
                fontsize=9,
                pad=6,
            )

    # 用实际子图位置来计算组标题中心，而不是手写经验坐标
    if high_axes_first_row:
        high_left = high_axes_first_row[0].get_position().x0
        high_right = high_axes_first_row[-1].get_position().x1
        high_center = (high_left + high_right) / 2
    else:
        high_center = 0.35

    if low_axes_first_row:
        low_left = low_axes_first_row[0].get_position().x0
        low_right = low_axes_first_row[-1].get_position().x1
        low_center = (low_left + low_right) / 2
    else:
        low_center = 0.78

    fig.text(
        high_center,
        0.965,
        "High-score group",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        low_center,
        0.965,
        "Low-score group",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
    )

    # 手动控制边距，比默认 tight 更稳定
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.93,
        bottom=0.05,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(CONFIG.global_seed)
    rng = np.random.default_rng(CONFIG.global_seed)

    device = CONFIG.global_device
    raw_dataset = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=None)
    class_names = [str(c) for c in raw_dataset.classes]

    dds_metric = DifficultyDirection(
        class_names=class_names,
        k=args.dds_k,
        clip_model=args.clip_model,
        device=device,
        eigval_lower_bound=args.dds_eigval_lower_bound,
        eigval_upper_bound=args.dds_eigval_upper_bound,
        important_eigval_ratio=args.dds_important_eigval_ratio,
    )
    div_metric = Div(
        class_names=class_names,
        k=args.div_k,
        clip_model=args.clip_model,
        device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=args.data_root,
        prompt_template=args.prompt_template,
    )

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    dds_loader = _make_loader(
        dds_metric.extractor.preprocess,
        args.data_root,
        args.batch_size,
        args.num_workers,
        device,
    )
    div_loader = _make_loader(
        div_metric.extractor.preprocess,
        args.data_root,
        args.batch_size,
        args.num_workers,
        device,
    )
    sa_loader = _make_loader(
        sa_metric.extractor.preprocess,
        args.data_root,
        args.batch_size,
        args.num_workers,
        device,
    )

    num_samples = len(raw_dataset)

    def _compute_scores() -> dict[str, np.ndarray]:
        with tqdm(total=3, desc="Computing static metrics", unit="metric") as metric_bar:
            dds_scores = dds_metric.score_dataset(
                tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
                adapter=image_adapter,
            )
            metric_bar.update(1)

            div_scores = div_metric.score_dataset(
                tqdm(div_loader, desc="Scoring Div", unit="batch"),
                adapter=image_adapter,
            )
            metric_bar.update(1)

            sa_scores = sa_metric.score_dataset(
                tqdm(sa_loader, desc="Scoring SA", unit="batch"),
                adapter_image=image_adapter,
                adapter_text=text_adapter,
            )
            metric_bar.update(1)
        return {
            "sa": sa_scores.scores.numpy(),
            "div": div_scores.scores.numpy(),
            "dds": dds_scores.scores.numpy(),
            "labels": np.asarray(raw_dataset.targets),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / args.static_cache_root,
        dataset=args.dataset,
        seed=args.seed,
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

    labels = static_scores["labels"].astype(np.int64)
    if not np.array_equal(labels, np.asarray(raw_dataset.targets, dtype=np.int64)):
        raise ValueError("Static score labels mismatch CIFAR100 raw dataset order.")

    w_sa, w_div, w_dds, bias = _load_learned_weights(Path(args.weights_json), args.dataset, args.seed)
    final_scores = (
        w_sa * static_scores["sa"].astype(np.float64)
        + w_div * static_scores["div"].astype(np.float64)
        + w_dds * static_scores["dds"].astype(np.float64)
        + bias
    )

    high_pairs, low_pairs = _select_extreme_indices_per_class(final_scores, labels, TARGET_CLASS_IDS, rng)
    _plot_pairs(raw_dataset, class_names, final_scores, high_pairs, low_pairs, Path(args.output))

    print(f"Saved figure to: {args.output}")
    print(
        "Weights:",
        f"sa={w_sa:.6f}, div={w_div:.6f}, dds={w_dds:.6f}, bias={bias:.6f}",
    )


if __name__ == "__main__":
    main()
