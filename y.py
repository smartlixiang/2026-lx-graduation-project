from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 CIFAR-10 默认静态指标及综合得分分布直方图。")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--output-dir", type=str, default="result_scoring_debug", help="输出目录")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-image-path",
        type=str,
        default=None,
        help="图像 adapter 权重路径（默认按 dataset/seed 自动构造）",
    )
    parser.add_argument(
        "--adapter-text-path",
        type=str,
        default=None,
        help="文本 adapter 权重路径（默认按 dataset/seed 自动构造）",
    )
    parser.add_argument("--adapter-seed", type=int, default=22, help="adapter 种子（用于按规则构造路径）")
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
    preprocess,
    data_root: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _save_hist(scores: np.ndarray, title: str, bins: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(scores, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to: {output_path}")


def _save_dual_hist(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    left_title: str,
    right_title: str,
    bins: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axes[0].hist(left_scores, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
    axes[0].set_title(left_title)
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0.0, 1.0)

    axes[1].hist(right_scores, bins=bins, range=(0.0, 1.0), color="#DD8452", alpha=0.85)
    axes[1].set_title(right_title)
    axes[1].set_xlabel("Score")
    axes[1].set_xlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison histogram to: {output_path}")


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))
    if np.isclose(max_val, min_val):
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - min_val) / (max_val - min_val)


def _save_div_dds_grid_hist(
    div_scores: np.ndarray,
    dds_scores_by_combo: list[tuple[tuple[int, int], np.ndarray]],
    bins: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for idx, ((k, upper_pct), dds_scores) in enumerate(dds_scores_by_combo):
        ax = flat_axes[idx]
        ax.hist(div_scores, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.55, label="Div (default)")
        ax.hist(dds_scores, bins=bins, range=(0.0, 1.0), color="#DD8452", alpha=0.55, label="DDS")
        ax.set_title(f"DDS(k={k}, upper={upper_pct}%) vs Div")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved Div-vs-DDS grid histogram to: {output_path}")


def _save_divk_dds_grid_hist(
    dds_scores: np.ndarray,
    div_scores_by_k: list[tuple[float, np.ndarray]],
    bins: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    flat_axes = axes.flatten()

    for idx, (div_k, div_scores_variant) in enumerate(div_scores_by_k):
        ax = flat_axes[idx]
        ax.hist(dds_scores, bins=bins, range=(0.0, 1.0), color="#DD8452", alpha=0.55, label="DDS (default)")
        ax.hist(div_scores_variant, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.55, label="Div")
        ax.set_title(f"Div(k={div_k * 100:.1f}%) vs DDS")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved Div(k)-vs-DDS grid histogram to: {output_path}")


def _compute_div_dds_overlap_rows(
    div_scores: np.ndarray,
    dds_scores_by_combo: list[tuple[tuple[int, int], np.ndarray]],
    cut_ratios: list[int],
) -> tuple[list[str], list[list[str]]]:
    header = ["DDS Params"] + [f"Top {ratio}%" for ratio in cut_ratios]
    rows: list[list[str]] = []

    for (k, upper_pct), dds_scores in dds_scores_by_combo:
        row = [f"k={k}, upper={upper_pct}%"]
        for ratio in cut_ratios:
            div_idx = _select_topk_indices(div_scores, ratio)
            dds_idx = _select_topk_indices(dds_scores, ratio)
            row.append(f"{_overlap_rate(div_idx, dds_idx):.4f}")
        rows.append(row)
    return header, rows


def _compute_divk_dds_overlap_rows(
    dds_scores: np.ndarray,
    div_scores_by_k: list[tuple[float, np.ndarray]],
    cut_ratios: list[int],
) -> tuple[list[str], list[list[str]]]:
    header = ["Div k"] + [f"Top {ratio}%" for ratio in cut_ratios]
    rows: list[list[str]] = []

    for div_k, div_scores_variant in div_scores_by_k:
        row = [f"k={div_k * 100:.1f}%"]
        for ratio in cut_ratios:
            dds_idx = _select_topk_indices(dds_scores, ratio)
            div_idx = _select_topk_indices(div_scores_variant, ratio)
            row.append(f"{_overlap_rate(dds_idx, div_idx):.4f}")
        rows.append(row)
    return header, rows


def _select_topk_indices(scores: np.ndarray, cut_ratio: int) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("scores 必须为一维数组。")
    if not (0 < cut_ratio <= 100):
        raise ValueError("cut_ratio 必须在 (0, 100] 范围内。")
    total = scores.shape[0]
    k = max(1, int(round(total * cut_ratio / 100.0)))
    selected = np.argpartition(-scores, kth=k - 1)[:k]
    return np.sort(selected)


def _overlap_rate(indices_a: np.ndarray, indices_b: np.ndarray) -> float:
    inter_size = np.intersect1d(indices_a, indices_b).shape[0]
    base = max(indices_a.shape[0], 1)
    return inter_size / base


def _print_aligned_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    all_rows = [header] + rows
    col_widths = [max(len(r[col_idx]) for r in all_rows) for col_idx in range(len(header))]

    def _format_row(items: list[str]) -> str:
        return "│ " + " │ ".join(item.ljust(col_widths[i]) for i, item in enumerate(items)) + " │"

    top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    mid_border = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    print(f"\n{title}")
    print(top_border)
    print(_format_row(header))
    print(mid_border)
    for row in rows:
        print(_format_row(row))
    print(bottom_border)


def _print_overlap_table(score_map: dict[str, np.ndarray], cut_ratios: list[int]) -> None:
    pairs = [
        ("SA", "Div", "SA vs Div"),
        ("SA", "DDS", "SA vs DDS"),
        ("Div", "DDS", "Div vs DDS"),
        ("Total Learned", "Total Naive", "Total Learned vs Naive"),
    ]

    header = ["Pair"] + [f"Top {ratio}%" for ratio in cut_ratios]
    rows: list[list[str]] = []

    for left_key, right_key, row_name in pairs:
        row = [row_name]
        for ratio in cut_ratios:
            left_idx = _select_topk_indices(score_map[left_key], ratio)
            right_idx = _select_topk_indices(score_map[right_key], ratio)
            row.append(f"{_overlap_rate(left_idx, right_idx):.4f}")
        rows.append(row)

    all_rows = [header] + rows
    col_widths = [max(len(r[col_idx]) for r in all_rows) for col_idx in range(len(header))]

    def _format_row(items: list[str]) -> str:
        return "│ " + " │ ".join(item.ljust(col_widths[i]) for i, item in enumerate(items)) + " │"

    top_border = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    mid_border = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    bottom_border = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    print("\nSubset overlap rate table (intersection / selected_size):")
    print(top_border)
    print(_format_row(header))
    print(mid_border)
    for row in rows:
        print(_format_row(row))
    print(bottom_border)


def main() -> None:
    args = parse_args()
    set_seed(args.adapter_seed)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    learned_weights = load_scoring_weights(weights_path, CIFAR10, str(args.weight_seed))

    dataset_for_names = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=None,
    )
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=CIFAR10,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.adapter_seed,
        map_location=device,
        adapter_image_path=args.adapter_image_path,
        adapter_text_path=args.adapter_text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()
    print(f"Resolved adapter image path: {adapter_paths['image_path']}")
    print(f"Resolved adapter text path: {adapter_paths['text_path']}")

    batch_size = 128
    num_workers = 4
    dds_loader = build_score_loader(dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers)

    num_samples = len(dataset_for_names)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
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
        dataset=CIFAR10,
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

    sa_scores = torch.from_numpy(static_scores["sa"])
    div_scores = torch.from_numpy(static_scores["div"])
    dds_scores = torch.from_numpy(static_scores["dds"])

    if not (len(sa_scores) == len(div_scores) == len(dds_scores)):
        raise RuntimeError("三个指标样本数不一致，无法计算综合分。")

    naive_weights = {"sa": 1.0 / 3.0, "div": 1.0 / 3.0, "dds": 1.0 / 3.0}
    total_scores_learned = (
        learned_weights["sa"] * sa_scores
        + learned_weights["div"] * div_scores
        + learned_weights["dds"] * dds_scores
    )
    total_scores_naive = (
        naive_weights["sa"] * sa_scores
        + naive_weights["div"] * div_scores
        + naive_weights["dds"] * dds_scores
    )

    _save_hist(np.asarray(sa_scores), "SA", args.bins, output_dir / "cifar10_sa_hist_default.png")
    _save_hist(np.asarray(div_scores), "Div", args.bins, output_dir / "cifar10_div_hist_default.png")
    _save_hist(np.asarray(dds_scores), "DDS", args.bins, output_dir / "cifar10_dds_hist_default.png")
    _save_hist(
        np.asarray(total_scores_learned),
        f"Total (learned seed {args.weight_seed})",
        args.bins,
        output_dir / f"cifar10_total_hist_learned_{args.weight_seed}.png",
    )

    _save_dual_hist(
        np.asarray(total_scores_learned),
        np.asarray(total_scores_naive),
        f"Total Learned (seed {args.weight_seed})",
        "Total Naive (uniform 1/3)",
        args.bins,
        output_dir / f"cifar10_total_hist_compare_learned_vs_naive_{args.weight_seed}.png",
    )

    total_scores_learned_norm = _minmax_normalize(np.asarray(total_scores_learned, dtype=np.float64))
    total_scores_naive_norm = _minmax_normalize(np.asarray(total_scores_naive, dtype=np.float64))
    _save_dual_hist(
        total_scores_learned_norm,
        total_scores_naive_norm,
        f"Total Learned Min-Max Norm (seed {args.weight_seed})",
        "Total Naive Min-Max Norm",
        args.bins,
        output_dir / f"cifar10_total_hist_compare_learned_vs_naive_minmax_{args.weight_seed}.png",
    )

    dds_param_combos = [(2, 20), (2, 30), (1, 10), (1, 20), (5, 30), (5, 50)]
    dds_scores_by_combo: list[tuple[tuple[int, int], np.ndarray]] = []
    for k, upper_pct in dds_param_combos:
        dds_metric.k = k
        dds_metric.eigval_upper_bound = upper_pct / 100.0
        dds_scores_variant = dds_metric.score_dataset(
            tqdm(dds_loader, desc=f"Scoring DDS (k={k}, upper={upper_pct}%)", unit="batch"),
            adapter=image_adapter,
        ).scores
        dds_scores_by_combo.append(((k, upper_pct), np.asarray(dds_scores_variant)))

    div_dds_header, div_dds_rows = _compute_div_dds_overlap_rows(
        np.asarray(div_scores),
        dds_scores_by_combo,
        cut_ratios=list(range(20, 100, 10)),
    )
    _print_aligned_table(
        "Div(default) vs DDS(parameter combos) overlap rate table:",
        div_dds_header,
        div_dds_rows,
    )

    _save_div_dds_grid_hist(
        np.asarray(div_scores),
        dds_scores_by_combo,
        args.bins,
        output_dir / "cifar10_div_vs_dds_param_grid_hist.png",
    )

    div_k_values = [0.002, 0.005, 0.01, 0.03, 0.05, 0.10]
    div_scores_by_k: list[tuple[float, np.ndarray]] = []
    for div_k in div_k_values:
        div_metric.k = div_k
        div_scores_variant = div_metric.score_dataset(
            tqdm(div_loader, desc=f"Scoring Div (k={div_k * 100:.1f}%)", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores_by_k.append((div_k, np.asarray(div_scores_variant)))

    divk_dds_header, divk_dds_rows = _compute_divk_dds_overlap_rows(
        np.asarray(dds_scores),
        div_scores_by_k,
        cut_ratios=list(range(20, 100, 10)),
    )
    _print_aligned_table(
        "DDS(default) vs Div(k sweep) overlap rate table:",
        divk_dds_header,
        divk_dds_rows,
    )

    _save_divk_dds_grid_hist(
        np.asarray(dds_scores),
        div_scores_by_k,
        args.bins,
        output_dir / "cifar10_div_k_sweep_vs_dds_grid_hist.png",
    )

    overlap_scores = {
        "SA": np.asarray(sa_scores),
        "Div": np.asarray(div_scores),
        "DDS": np.asarray(dds_scores),
        "Total Learned": np.asarray(total_scores_learned),
        "Total Naive": np.asarray(total_scores_naive),
    }
    _print_overlap_table(overlap_scores, cut_ratios=list(range(20, 100, 10)))


if __name__ == "__main__":
    main()
