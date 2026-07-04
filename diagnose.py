from __future__ import annotations

import argparse
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learn_scoring_weights import (  # noqa: E402
    _build_dataset,
    build_score_loader,
)
from noise_exp.cal_noise_mask import (  # noqa: E402
    _get_or_compute_group_mean_stats,
    _mean_stats_cache_path,
    ensure_scoring_weights,
    load_scoring_weights,
)
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR100  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.score_utils import standard_zscore  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare distribution shift of random and group selection for each keep ratio. "
            "The group selection can use naive or learned static metric weights."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        choices=["naive", "learned"],
        help=(
            "Static metric weight mode for group selection. "
            "'naive' uses equal SA/Div/DDS weights; 'learned' loads weights/scoring_weights.json "
            "for the selected dataset and seed."
        ),
    )
    parser.add_argument("--kr", type=str, default="20,30,40,50,60,70,80,90")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--group-candidate-pool-size", type=int, default=5)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument(
        "--dist-min-ratio",
        type=float,
        default=0.5,
        help="Initial coefficient ratio relative to kr-dependent maximum. Current strategy: 0.5.",
    )
    parser.add_argument(
        "--dist-power",
        type=float,
        default=1.0,
        help="Progress exponent. 1.0 means linear increase.",
    )
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args()


def parse_ratio_list(ratio_text: str) -> list[int]:
    items = [item.strip() for item in ratio_text.split(",") if item.strip()]
    return [int(item) for item in items]


def load_group_weights(dataset_name: str, mode: str, seed: int) -> dict[str, float]:
    """Load SA/Div/DDS weights for group selection.

    mode='naive' returns equal weights through calculate_my_mask's weight loader.
    mode='learned' requires weights/scoring_weights.json to contain the entry for this dataset and seed.
    """
    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)
    weights = load_scoring_weights(all_weights, mode, seed)
    total = float(weights["sa"] + weights["div"] + weights["dds"])
    if total <= 0:
        raise ValueError(f"Invalid {mode} weights: {weights}")
    return {key: float(value) / total for key, value in weights.items()}


def allocate_class_budgets(labels: np.ndarray, num_classes: int, keep_ratio: int) -> np.ndarray:
    sr = float(keep_ratio) / 100.0
    num_samples = int(labels.shape[0])
    target_size = min(num_samples, max(1, int(round(sr * num_samples))))

    class_sizes = np.asarray([np.sum(labels == c) for c in range(num_classes)], dtype=np.int64)
    raw = class_sizes.astype(np.float64) * sr
    budgets = np.minimum(np.floor(raw).astype(np.int64), class_sizes)

    need = int(target_size - np.sum(budgets))
    if need <= 0:
        return budgets

    frac = raw - budgets.astype(np.float64)
    order = np.lexsort((np.arange(num_classes, dtype=np.int64), -frac))
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


# ---------------------------------------------------------------------
# Distribution correction coefficient strategy.
#
# Edit this function to test another strategy.
#
# Current strategy:
#   dist_weight_max = max(0, 0.8 - 0.005 * keep_ratio)
#   dist_weight_t = dist_weight_max * (min_ratio + (1 - min_ratio) * progress^power)
# where progress = selected_count_in_class / class_budget.
#
# With default min_ratio=0.5 and power=1.0, the coefficient linearly increases
# from half of the kr-dependent maximum to the full kr-dependent maximum.
# ---------------------------------------------------------------------
def compute_dist_weight(
    *,
    keep_ratio: int,
    current_count: int,
    class_budget: int,
    min_ratio: float = 0.5,
    power: float = 1.0,
) -> tuple[float, float, float]:
    dist_weight_max = max(0.0, 0.8 - 0.005 * float(keep_ratio))
    min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
    progress = 1.0 if class_budget <= 0 else float(np.clip(current_count / float(class_budget), 0.0, 1.0))
    power = max(float(power), 1e-8)
    dist_weight_t = dist_weight_max * (min_ratio + (1.0 - min_ratio) * (progress ** power))
    return float(dist_weight_t), float(dist_weight_max), float(progress)


def make_random_mask(labels: np.ndarray, num_classes: int, keep_ratio: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    budgets = allocate_class_budgets(labels, num_classes, keep_ratio)
    rng = np.random.default_rng(seed + 1000003 * int(keep_ratio))
    mask = np.zeros(labels.shape[0], dtype=np.uint8)

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id).astype(np.int64)
        budget = int(budgets[class_id])
        if class_indices.size == 0 or budget <= 0:
            continue
        chosen = class_indices if budget >= class_indices.size else rng.choice(class_indices, size=budget, replace=False)
        mask[chosen.astype(np.int64)] = 1
    return mask, budgets


def compute_distribution_shift(
    selected_mask: np.ndarray,
    labels: np.ndarray,
    div_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    num_classes: int,
) -> float:
    selected_bool = selected_mask.astype(bool)
    shifts = []
    for class_id in range(num_classes):
        class_selected = (labels == class_id) & selected_bool
        if not np.any(class_selected):
            continue
        mu_sub = np.mean(div_features_np[class_selected], axis=0, dtype=np.float32)
        mu_full = full_class_mean[class_id].astype(np.float32, copy=False)
        shifts.append(float(np.linalg.norm(mu_sub - mu_full)))
    return float(np.mean(shifts)) if shifts else 0.0


def make_group_mask(
    *,
    labels: np.ndarray,
    num_classes: int,
    keep_ratio: int,
    seed: int,
    weights: dict[str, float],
    sa_raw_np: np.ndarray,
    dds_raw_np: np.ndarray,
    div_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    div_metric: Div,
    device: torch.device,
    group_candidate_pool_size: int,
    group_init_count: int,
    dist_min_ratio: float,
    dist_power: float,
) -> tuple[np.ndarray, dict[str, object]]:
    num_samples = int(labels.shape[0])
    budgets = allocate_class_budgets(labels, num_classes, keep_ratio)
    class_indices_list = [np.flatnonzero(labels == c).astype(np.int64) for c in range(num_classes)]
    rng = np.random.default_rng(seed + 9176 * int(keep_ratio))

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    class_selected_counts = np.zeros(num_classes, dtype=np.int64)
    class_selected_sum = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    init_per_class = np.zeros(num_classes, dtype=np.int64)

    requested_init_count = max(0, int(group_init_count))
    for class_id, class_indices in enumerate(class_indices_list):
        budget = int(budgets[class_id])
        if class_indices.size == 0 or budget <= 0 or requested_init_count <= 0:
            continue
        init_count = min(requested_init_count, budget, int(class_indices.size))
        init_per_class[class_id] = init_count

        top_pool_size = max(init_count, int(np.ceil(0.5 * class_indices.size)))
        top_pool_size = min(int(class_indices.size), max(1, top_pool_size))
        ranked_by_sa = np.argsort(-sa_raw_np[class_indices], kind="mergesort")[:top_pool_size]
        init_pool = class_indices[ranked_by_sa]
        init_indices = init_pool if init_pool.size <= init_count else rng.choice(init_pool, size=init_count, replace=False)

        init_indices = init_indices.astype(np.int64)
        selected_mask[init_indices] = 1
        class_selected_counts[class_id] = init_count
        class_selected_sum[class_id] = np.sum(div_features_np[init_indices], axis=0, dtype=np.float32)

    candidate_pool_size = max(1, int(group_candidate_pool_size))
    total_to_add = int(np.sum(budgets) - np.sum(init_per_class))
    pbar = tqdm(total=total_to_add, desc=f"[group:{keep_ratio}]", unit="sample", leave=False)

    dist_weights_used = []
    progress_used = []

    while True:
        remaining_by_class = budgets - class_selected_counts
        active_classes = np.flatnonzero(remaining_by_class > 0).astype(np.int64)
        if active_classes.size == 0:
            break

        remain_total = int(np.sum(remaining_by_class))
        if remain_total < active_classes.size:
            chosen_classes = np.sort(rng.choice(active_classes, size=remain_total, replace=False).astype(np.int64))
        else:
            chosen_classes = active_classes

        for class_id in chosen_classes:
            class_indices = class_indices_list[int(class_id)]
            candidate_indices = class_indices[selected_mask[class_indices] == 0]
            if candidate_indices.size == 0:
                continue

            current_count = int(class_selected_counts[class_id])
            if current_count <= 0:
                continue

            class_budget = int(budgets[class_id])
            dist_weight_t, dist_weight_max, progress = compute_dist_weight(
                keep_ratio=keep_ratio,
                current_count=current_count,
                class_budget=class_budget,
                min_ratio=dist_min_ratio,
                power=dist_power,
            )
            dist_weights_used.append(dist_weight_t)
            progress_used.append(progress)

            current_sum = class_selected_sum[class_id]
            mu_full = full_class_mean[class_id].astype(np.float32, copy=False)
            mu_sub = current_sum / float(current_count)
            old_dist = float(np.linalg.norm(mu_sub - mu_full))

            dynamic_k = max(3, int(ceil(0.05 * current_count)))
            reference_indices = class_indices[selected_mask[class_indices] > 0]

            candidate_features_t = torch.as_tensor(div_features_np[candidate_indices], dtype=torch.float32, device=device)
            reference_features_t = torch.as_tensor(div_features_np[reference_indices], dtype=torch.float32, device=device)

            div_raw = div_metric._knn_mean_distance_to_reference(
                query_features=candidate_features_t,
                reference_features=reference_features_t,
                k=float(dynamic_k),
                query_indices=torch.as_tensor(candidate_indices, dtype=torch.long, device=device),
                reference_indices=torch.as_tensor(reference_indices, dtype=torch.long, device=device),
            ).detach().cpu().numpy().astype(np.float32)
            div_local = standard_zscore(div_raw)

            candidate_features_np = div_features_np[candidate_indices]
            mu_new = (current_sum[None, :] + candidate_features_np) / float(current_count + 1)
            new_dist = np.linalg.norm(mu_new - mu_full[None, :], axis=1)
            dist_improve = (old_dist - new_dist).astype(np.float32)
            dist_local = standard_zscore(dist_improve)

            sa_local = standard_zscore(sa_raw_np[candidate_indices])
            dds_local = standard_zscore(dds_raw_np[candidate_indices])

            combined_scores = (
                weights["sa"] * sa_local
                + weights["dds"] * dds_local
                + weights["div"] * div_local
                + dist_weight_t * dist_local
            ).astype(np.float32)

            rank = np.argsort(-combined_scores, kind="mergesort")
            pool_n = min(candidate_pool_size, candidate_indices.size)
            pool_indices = candidate_indices[rank[:pool_n]]
            picked_idx = int(pool_indices[0]) if pool_n == 1 else int(rng.choice(pool_indices, size=1, replace=False)[0])

            selected_mask[picked_idx] = 1
            class_selected_counts[class_id] += 1
            class_selected_sum[class_id] += div_features_np[picked_idx]
            pbar.update(1)

    pbar.close()

    dist_weight_max = max(0.0, 0.8 - 0.005 * float(keep_ratio))
    stats = {
        "dist_weight_strategy": "increase_from_min_ratio_to_kr_linear_max",
        "dist_weight_max": float(dist_weight_max),
        "dist_weight_min_ratio": float(dist_min_ratio),
        "dist_weight_power": float(dist_power),
        "dist_weight_mean_used": float(np.mean(dist_weights_used)) if dist_weights_used else 0.0,
        "progress_mean_used": float(np.mean(progress_used)) if progress_used else 0.0,
        "selected_count": int(selected_mask.sum()),
    }
    return selected_mask.astype(np.uint8), stats


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    keep_ratios = parse_ratio_list(args.kr)
    if not keep_ratios:
        raise ValueError("--kr must not be empty.")

    set_seed(args.seed)
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    group_weights = load_group_weights(dataset_name, args.mode, args.seed)
    print(
        f"[Weights] mode={args.mode} | "
        f"SA={group_weights['sa']:.6f}, Div={group_weights['div']:.6f}, DDS={group_weights['dds']:.6f}"
    )

    start_time = time.perf_counter()

    dataset_for_names = _build_dataset(dataset_name, str(PROJECT_ROOT / "data"), transform=None)
    labels = np.asarray(dataset_for_names.targets, dtype=np.int64)
    class_names = resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=PROJECT_ROOT / "data",
        class_names=dataset_for_names.classes,  # type: ignore[attr-defined]
    )
    num_classes = len(class_names)

    print(f"[Init] dataset={dataset_name} | samples={labels.shape[0]} | classes={num_classes} | seed={args.seed}")

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

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        str(PROJECT_ROOT / "data"),
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        str(PROJECT_ROOT / "data"),
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        str(PROJECT_ROOT / "data"),
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )

    num_samples = int(labels.shape[0])

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"), adapter=image_adapter
        ).scores
        div_scores = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"), adapter=image_adapter
        ).scores
        sa_scores = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {"sa": np.asarray(sa_scores), "div": np.asarray(div_scores), "dds": np.asarray(dds_scores), "labels": labels}

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

    sa_raw_np = np.asarray(static_scores["sa"], dtype=np.float32)
    dds_raw_np = np.asarray(static_scores["dds"], dtype=np.float32)

    print("[Init] encoding Div features once...")
    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = (
        div_features.detach().cpu().numpy()
        if isinstance(div_features, torch.Tensor)
        else np.asarray(div_features)
    ).astype(np.float32)

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

    rows = []
    print("\n[Run] computing random and group distribution shifts...")
    for keep_ratio in keep_ratios:
        random_mask, _ = make_random_mask(labels, num_classes, keep_ratio, args.seed)
        random_shift = compute_distribution_shift(random_mask, labels, div_features_np, full_class_mean, num_classes)

        group_mask, group_stats = make_group_mask(
            labels=labels,
            num_classes=num_classes,
            keep_ratio=keep_ratio,
            seed=args.seed,
            weights=group_weights,
            sa_raw_np=sa_raw_np,
            dds_raw_np=dds_raw_np,
            div_features_np=div_features_np,
            full_class_mean=full_class_mean,
            div_metric=div_metric,
            device=device,
            group_candidate_pool_size=args.group_candidate_pool_size,
            group_init_count=args.group_init_count,
            dist_min_ratio=args.dist_min_ratio,
            dist_power=args.dist_power,
        )
        group_shift = compute_distribution_shift(group_mask, labels, div_features_np, full_class_mean, num_classes)

        ratio = group_shift / random_shift if random_shift > 0 else float("nan")
        delta = group_shift - random_shift
        row = {
            "kr": int(keep_ratio),
            "dist_weight_max": float(group_stats["dist_weight_max"]),
            "dist_weight_mean_used": float(group_stats["dist_weight_mean_used"]),
            "random_shift": float(random_shift),
            "group_shift": float(group_shift),
            "group/random": float(ratio),
            "group-random": float(delta),
        }
        rows.append(row)
        print(
            f"kr={keep_ratio:>2d} | dist_max={row['dist_weight_max']:.4f} | "
            f"dist_mean={row['dist_weight_mean_used']:.4f} | random={random_shift:.6f} | "
            f"group={group_shift:.6f} | group/random={ratio:.4f}"
        )

    print("\n=== Distribution shift comparison ===")
    print(f"dataset={dataset_name} | mode={args.mode} | seed={args.seed}")
    print(
        f"weights: SA={group_weights['sa']:.6f}, "
        f"Div={group_weights['div']:.6f}, DDS={group_weights['dds']:.6f}"
    )

    headers = ["kr", "dist_max", "dist_mean", "random_shift", "group_shift", "group/random", "group-random"]
    table_rows = [
        [
            str(int(row["kr"])),
            f"{float(row['dist_weight_max']):.4f}",
            f"{float(row['dist_weight_mean_used']):.4f}",
            f"{float(row['random_shift']):.6f}",
            f"{float(row['group_shift']):.6f}",
            f"{float(row['group/random']):.4f}",
            f"{float(row['group-random']):+.6f}",
        ]
        for row in rows
    ]

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    print("  ".join(headers[i].rjust(widths[i]) for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[i].rjust(widths[i]) for i in range(len(headers))))

    print(f"\n[Done] elapsed={time.perf_counter() - start_time:.2f}s")
    print("[Note] Smaller distribution_shift means the selected subset mean is closer to the full class mean in Div feature space.")
    print("[Edit point] To adjust the coefficient strategy, edit compute_dist_weight() near the top of this script.")


if __name__ == "__main__":
    main()
