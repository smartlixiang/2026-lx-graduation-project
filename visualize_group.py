from __future__ import annotations

import argparse
import csv
import json
import sys
from math import ceil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.score_utils import standard_zscore  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402

# Reuse the exact helper logic from calculate_my_mask.py where possible.
from calculate_my_mask import (  # noqa: E402
    _get_or_compute_group_mean_stats,
    _mean_stats_cache_path,
    ensure_scoring_weights,
    load_scoring_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize local one-step decisions of group selection under different "
            "distribution-correction weights."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight-group", type=str, default="learned", choices=["naive", "learned"])
    parser.add_argument("--keep-ratio", type=int, default=50)
    parser.add_argument(
        "--dist-weight",
        type=str,
        default=None,
        help="Deprecated; group visualization now uses linear decay by class progress with max 0.6.",
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated class ids to visualize. "
            "If omitted, choose low/median/high classes by initial static score mean."
        ),
    )
    parser.add_argument("--num-auto-classes", type=int, default=3)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument(
        "--late-fraction",
        type=float,
        default=0.9,
        help=(
            "For late snapshot, simulate this fraction of class budget before "
            "visualizing the next add step. 0.9 means near the end."
        ),
    )
    parser.add_argument("--top-candidates", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/group",
        help="Directory for PNG/CSV/JSON outputs.",
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print tiny-imagenet prompt debugging info.",
    )
    return parser.parse_args()


def parse_float_list(text: str | None, default_value: float) -> list[float]:
    if text is None or not text.strip():
        return [float(default_value)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_dataset(dataset_name: str, transform) -> datasets.VisionDataset:
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = data_root / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_score_loader(
    preprocess,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = build_dataset(dataset_name, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def allocate_class_budgets(labels: np.ndarray, num_classes: int, keep_ratio: int) -> np.ndarray:
    sr = float(keep_ratio) / 100.0
    target_size = int(round(sr * labels.shape[0]))
    target_size = min(labels.shape[0], max(1, target_size))

    class_sizes = np.asarray([np.sum(labels == c) for c in range(num_classes)], dtype=np.int64)
    raw = class_sizes.astype(np.float64) * sr
    floor_budget = np.floor(raw).astype(np.int64)
    floor_budget = np.minimum(floor_budget, class_sizes)

    need = int(target_size - np.sum(floor_budget))
    if need <= 0:
        return floor_budget

    frac = raw - floor_budget.astype(np.float64)
    order = np.lexsort((np.arange(num_classes, dtype=np.int64), -frac))
    budgets = floor_budget.copy()
    for class_id in order:
        if need <= 0:
            break
        if budgets[class_id] >= class_sizes[class_id]:
            continue
        budgets[class_id] += 1
        need -= 1

    if need != 0:
        raise RuntimeError("Failed to allocate class budgets.")
    return budgets


def choose_auto_classes(
    labels: np.ndarray,
    num_classes: int,
    static_init_score: np.ndarray,
    num_auto_classes: int,
) -> list[int]:
    class_means: list[tuple[int, float]] = []
    for class_id in range(num_classes):
        idx = np.flatnonzero(labels == class_id)
        if idx.size == 0:
            continue
        class_means.append((class_id, float(np.mean(static_init_score[idx]))))

    if not class_means:
        raise RuntimeError("No non-empty classes found.")

    class_means.sort(key=lambda item: item[1])
    if num_auto_classes <= 1:
        return [class_means[len(class_means) // 2][0]]

    positions = np.linspace(0, len(class_means) - 1, num=min(num_auto_classes, len(class_means)))
    chosen: list[int] = []
    for pos in positions:
        class_id = class_means[int(round(float(pos)))][0]
        if class_id not in chosen:
            chosen.append(class_id)
    return chosen


def init_selected_for_class(
    class_indices: np.ndarray,
    budget: int,
    sa_raw_scores: np.ndarray,
    init_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.zeros(class_indices.shape[0], dtype=bool)
    if class_indices.size == 0 or budget <= 0:
        return selected

    count = min(init_count, budget, int(class_indices.size))
    top_pool_size = max(count, int(np.ceil(0.5 * class_indices.size)))
    top_pool_size = min(int(class_indices.size), max(1, top_pool_size))
    ranked_local = np.argsort(-sa_raw_scores[class_indices], kind="mergesort")[:top_pool_size]
    if ranked_local.size <= count:
        chosen_local = ranked_local
    else:
        chosen_local = rng.choice(ranked_local, size=count, replace=False).astype(np.int64)
    selected[chosen_local] = True
    return selected


def compute_candidate_table(
    *,
    class_id: int,
    class_indices: np.ndarray,
    selected_local: np.ndarray,
    div_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    div_metric: Div,
    device: torch.device,
    sa_scores: np.ndarray,
    dds_scores: np.ndarray,
    weights: dict[str, float],
    class_budget: int,
    top_candidates: int,
) -> tuple[list[dict[str, float | int]], int | None]:
    candidate_indices = class_indices[~selected_local]
    reference_indices = class_indices[selected_local]

    if candidate_indices.size == 0 or reference_indices.size == 0:
        return [], None

    current_count = int(reference_indices.size)
    current_sum = np.sum(div_features_np[reference_indices], axis=0, dtype=np.float32)
    mu_full = full_class_mean[class_id].astype(np.float32, copy=False)
    mu_sub = current_sum / float(current_count)
    old_dist = float(np.linalg.norm(mu_sub - mu_full))

    dynamic_k = max(3, int(ceil(0.05 * current_count)))

    candidate_features_t = torch.as_tensor(
        div_features_np[candidate_indices],
        dtype=torch.float32,
        device=device,
    )
    reference_features_t = torch.as_tensor(
        div_features_np[reference_indices],
        dtype=torch.float32,
        device=device,
    )

    div_raw = div_metric._knn_mean_distance_to_reference(
        query_features=candidate_features_t,
        reference_features=reference_features_t,
        k=float(dynamic_k),
        query_indices=torch.as_tensor(candidate_indices, dtype=torch.long, device=device),
        reference_indices=torch.as_tensor(reference_indices, dtype=torch.long, device=device),
    ).detach().cpu().numpy().astype(np.float32)

    div_z = standard_zscore(div_raw)

    candidate_features_np = div_features_np[candidate_indices]
    mu_new = (current_sum[None, :] + candidate_features_np) / float(current_count + 1)
    new_dist = np.linalg.norm(mu_new - mu_full[None, :], axis=1)
    dist_improve = (old_dist - new_dist).astype(np.float32)
    dist_z = standard_zscore(dist_improve)
    sa_z = standard_zscore(sa_scores[candidate_indices])
    dds_z = standard_zscore(dds_scores[candidate_indices])
    progress = current_count / float(class_budget) if class_budget > 0 else 1.0
    progress = float(np.clip(progress, 0.0, 1.0))
    dist_weight_t = 0.6 * (1.0 - progress)

    sa_component = float(weights["sa"]) * sa_z
    dds_component = float(weights["dds"]) * dds_z
    div_component = float(weights["div"]) * div_z
    dist_component = float(dist_weight_t) * dist_z
    total = sa_component + dds_component + div_component + dist_component

    order = np.argsort(-total, kind="mergesort")
    top_n = min(top_candidates, candidate_indices.size)

    rows: list[dict[str, float | int]] = []
    for rank, local_pos in enumerate(order[:top_n], start=1):
        sample_index = int(candidate_indices[local_pos])
        rows.append(
            {
                "rank": int(rank),
                "sample_index": sample_index,
                "class_id": int(class_id),
                "selected_count_before_add": int(current_count),
                "dynamic_k": int(dynamic_k),
                "old_mean_shift": float(old_dist),
                "new_mean_shift": float(new_dist[local_pos]),
                "dist_improve_raw": float(dist_improve[local_pos]),
                "div_raw": float(div_raw[local_pos]),
                "sa_z": float(sa_z[local_pos]),
                "dds_z": float(dds_z[local_pos]),
                "div_z": float(div_z[local_pos]),
                "dist_z": float(dist_z[local_pos]),
                "sa_component": float(sa_component[local_pos]),
                "dds_component": float(dds_component[local_pos]),
                "div_component": float(div_component[local_pos]),
                "dist_weight_t": float(dist_weight_t),
                "dist_component": float(dist_component[local_pos]),
                "total": float(total[local_pos]),
            }
        )

    picked_idx = int(candidate_indices[order[0]]) if order.size > 0 else None
    return rows, picked_idx


def greedy_advance_one_class(
    *,
    class_id: int,
    class_indices: np.ndarray,
    selected_local: np.ndarray,
    target_count: int,
    div_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    div_metric: Div,
    device: torch.device,
    sa_scores: np.ndarray,
    dds_scores: np.ndarray,
    weights: dict[str, float],
    class_budget: int,
) -> np.ndarray:
    selected = selected_local.copy()

    while int(np.sum(selected)) < target_count:
        rows, picked_idx = compute_candidate_table(
            class_id=class_id,
            class_indices=class_indices,
            selected_local=selected,
            div_features_np=div_features_np,
            full_class_mean=full_class_mean,
            div_metric=div_metric,
            device=device,
            sa_scores=sa_scores,
            dds_scores=dds_scores,
            weights=weights,
            class_budget=class_budget,
            top_candidates=1,
        )
        if picked_idx is None:
            break
        local_pick = np.where(class_indices == picked_idx)[0]
        if local_pick.size == 0:
            break
        selected[int(local_pick[0])] = True

    return selected


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_candidate_contributions(path: Path, title: str, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [f"#{int(row['rank'])}\nidx {int(row['sample_index'])}" for row in rows]
    component_names = ["sa_component", "dds_component", "div_component", "dist_component", "total"]
    display_names = ["SA", "DDS", "DynDiv", "Dist", "Total"]

    x = np.arange(len(rows), dtype=np.float32)
    width = 0.15

    fig_width = max(10, 0.55 * len(rows) + 5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    for offset, (component, display_name) in enumerate(zip(component_names, display_names)):
        values = np.asarray([float(row[component]) for row in rows], dtype=np.float32)
        ax.bar(x + (offset - 2) * width, values, width=width, label=display_name)

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Contribution")
    ax.set_title(title)
    ax.legend(loc="best", ncols=5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else CONFIG.global_device
    if args.dist_weight is not None:
        print("[warn] --dist-weight is ignored; using linear decay by class progress with max 0.6.")
    dist_weights = [0.6]

    output_root = (
        Path(args.output_dir)
        / dataset_name
        / f"seed_{args.seed}"
        / f"kr_{args.keep_ratio}"
        / args.weight_group
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_for_names = build_dataset(dataset_name, transform=None)
    labels = np.asarray(dataset_for_names.targets, dtype=np.int64)
    class_names = resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=PROJECT_ROOT / "data",
        class_names=dataset_for_names.classes,  # type: ignore[attr-defined]
    )
    num_classes = len(class_names)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)
    weights = load_scoring_weights(all_weights, args.weight_group, args.seed)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=str(PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=dataset_name,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    dds_loader = build_score_loader(dds_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)

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
            "sa": np.asarray(sa_scores_local, dtype=np.float32),
            "div": np.asarray(div_scores_local, dtype=np.float32),
            "dds": np.asarray(dds_scores_local, dtype=np.float32),
            "labels": labels,
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=dataset_name,
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

    sa_scores = np.asarray(static_scores["sa"], dtype=np.float32)
    dds_scores = np.asarray(static_scores["dds"], dtype=np.float32)
    static_init_score = sa_scores
    rng = np.random.default_rng(args.seed)

    print("Encoding Div features for group visualization...")
    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = div_features.detach().cpu().numpy().astype(np.float32)

    mean_stats_cache_path = _mean_stats_cache_path(
        dataset_name=dataset_name,
        clip_model=args.clip_model,
        adapter_image_path=str(adapter_paths["image_path"]),
    )
    full_class_mean, _ = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels,
        num_classes=num_classes,
    )

    class_budgets = allocate_class_budgets(labels, num_classes, args.keep_ratio)
    class_indices_list = [np.flatnonzero(labels == c).astype(np.int64) for c in range(num_classes)]

    class_ids = parse_int_list(args.class_ids)
    if class_ids is None:
        class_ids = choose_auto_classes(
            labels=labels,
            num_classes=num_classes,
            static_init_score=static_init_score,
            num_auto_classes=args.num_auto_classes,
        )

    summary: list[dict[str, object]] = []

    for dist_weight in dist_weights:
        for class_id in class_ids:
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"class_id out of range: {class_id}")

            class_indices = class_indices_list[class_id]
            budget = int(class_budgets[class_id])
            if class_indices.size == 0 or budget <= 0:
                continue

            init_selected = init_selected_for_class(
                class_indices=class_indices,
                budget=budget,
                sa_raw_scores=sa_scores,
                init_count=args.group_init_count,
                rng=rng,
            )
            init_selected_count = int(np.sum(init_selected))
            if init_selected_count <= 0:
                continue

            late_target = int(round(init_selected_count + args.late_fraction * max(0, budget - init_selected_count)))
            late_target = min(max(init_selected_count, late_target), max(init_selected_count, budget - 1))

            snapshots = {
                "early": init_selected,
                "late": greedy_advance_one_class(
                    class_id=class_id,
                    class_indices=class_indices,
                    selected_local=init_selected,
                    target_count=late_target,
                    div_features_np=div_features_np,
                    full_class_mean=full_class_mean,
                    div_metric=div_metric,
                    device=device,
                    sa_scores=sa_scores,
                    dds_scores=dds_scores,
                    weights=weights,
                    class_budget=budget,
                ),
            }

            class_label = str(class_names[class_id]) if class_id < len(class_names) else str(class_id)

            for phase, selected_local in snapshots.items():
                rows, picked_idx = compute_candidate_table(
                    class_id=class_id,
                    class_indices=class_indices,
                    selected_local=selected_local,
                    div_features_np=div_features_np,
                    full_class_mean=full_class_mean,
                    div_metric=div_metric,
                    device=device,
                    sa_scores=sa_scores,
                    dds_scores=dds_scores,
                    weights=weights,
                    class_budget=budget,
                    top_candidates=args.top_candidates,
                )

                base_name = f"class_{class_id}_{phase}_dist_linear_decay"
                csv_path = output_root / f"{base_name}.csv"
                png_path = output_root / f"{base_name}.png"

                write_csv(csv_path, rows)

                title = (
                    f"{dataset_name} | class {class_id} ({class_label}) | {phase} | "
                    f"selected={int(np.sum(selected_local))}/{budget} | dist schedule=linear_decay(max=0.6)"
                )
                plot_candidate_contributions(png_path, title, rows)

                summary.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(args.seed),
                        "keep_ratio": int(args.keep_ratio),
                        "weight_group": args.weight_group,
                        "class_id": int(class_id),
                        "class_name": class_label,
                        "phase": phase,
                        "dist_weight_schedule": "linear_decay_by_class_progress",
                        "dist_weight_max": 0.6,
                        "dist_weight_min": 0.0,
                        "selected_count_before_add": int(np.sum(selected_local)),
                        "class_budget": int(budget),
                        "picked_sample_index": int(picked_idx) if picked_idx is not None else None,
                        "csv": str(csv_path),
                        "png": str(png_path),
                    }
                )

                print(f"[saved] {png_path}")
                print(f"[saved] {csv_path}")

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": weights,
                "dist_weight_schedule": "linear_decay_by_class_progress",
                "dist_weight_max": 0.6,
                "dist_weight_min": 0.0,
                "class_ids": [int(c) for c in class_ids],
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()