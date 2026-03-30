from __future__ import annotations

import argparse
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import calculate_my_mask as cmm
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.static_score_cache import get_or_compute_static_scores
from utils.seed import set_seed


DEFAULT_KR = "20,30,40,50,60,70,80,90"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare overlap between naive-group and learned-group masks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=cmm.CIFAR100,
        choices=cmm.AVAILABLE_DATASETS,
        help="目标数据集名称，默认 cifar100",
    )
    parser.add_argument(
        "--kr",
        type=str,
        default=DEFAULT_KR,
        help="keep_ratio 列表（百分比），支持逗号分隔或单值；默认 20 到 90。",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument(
        "--group-candidate-pool-size",
        type=int,
        default=1,
        help="group 模式类内贪心候选池大小，默认 1。",
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="打印 tiny-imagenet 前几个最终英文 prompt（调试用）。",
    )
    return parser.parse_args()


def _quantile_minmax(values: np.ndarray, low_q: float = 0.002, high_q: float = 0.998) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if abs(high - low) <= 1e-12:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _allocate_class_budgets(class_indices_list: list[np.ndarray], sr: float, target_size: int) -> np.ndarray:
    num_classes = len(class_indices_list)
    class_sizes = np.asarray([idx.size for idx in class_indices_list], dtype=np.int64)
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
        raise RuntimeError("类别预算分配失败，无法满足目标总样本数。")
    return budgets


def select_group_mask_with_linear_dist_weight(
    *,
    sa_scores: np.ndarray,
    dds_static_scores: np.ndarray,
    div_metric: Div,
    div_loader,
    image_adapter,
    labels: np.ndarray,
    weights: dict[str, float],
    num_classes: int,
    keep_ratio: int,
    device: torch.device,
    dataset_name: str,
    seed: int,
    clip_model: str,
    adapter_image_path: str,
    group_candidate_pool_size: int = 1,
    progress: tqdm | None = None,
    progress_prefix: str = "",
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")

    num_samples = sa_scores.shape[0]
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    dds_static_np = np.asarray(dds_static_scores, dtype=np.float32)
    if labels_np.shape[0] != num_samples or dds_static_np.shape[0] != num_samples:
        raise ValueError("样本数不一致，无法执行 group。")

    sr = float(keep_ratio) / 100.0
    target_size = int(round(sr * num_samples))
    target_size = min(num_samples, max(1, target_size)) if num_samples > 0 else 0
    if target_size <= 0:
        raise ValueError("target_size 必须大于 0。")

    class_indices_list = [np.flatnonzero(labels_np == c).astype(np.int64) for c in range(num_classes)]
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = (
        div_features.detach().cpu().numpy()
        if isinstance(div_features, torch.Tensor)
        else np.asarray(div_features)
    ).astype(np.float32)

    mean_stats_cache_path = cmm._mean_stats_cache_path(
        dataset_name=dataset_name,
        clip_model=clip_model,
        adapter_image_path=adapter_image_path,
    )
    full_class_mean, _ = cmm._get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    full_class_mean_f32 = full_class_mean.astype(np.float32, copy=False)

    class_budgets = _allocate_class_budgets(class_indices_list, sr, target_size)
    candidate_pool_size = max(1, int(group_candidate_pool_size))
    dist_weight = max(0.0, 1.0 - 0.01 * float(keep_ratio))

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    class_selected_counts = np.zeros(num_classes, dtype=np.int64)
    class_selected_sum = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    init_per_class = np.zeros(num_classes, dtype=np.int64)
    static_init_score = (weights["sa"] * sa_scores_np + weights["dds"] * dds_static_np).astype(np.float32)

    for class_id, class_indices in enumerate(class_indices_list):
        budget = int(class_budgets[class_id])
        if class_indices.size == 0 or budget <= 0:
            continue
        init_count = min(3, budget, int(class_indices.size))
        init_per_class[class_id] = init_count
        class_static = static_init_score[class_indices]
        ranked_local = np.argsort(-class_static, kind="mergesort")[:init_count]
        init_indices = class_indices[ranked_local]
        selected_mask[init_indices] = 1
        class_selected_counts[class_id] = init_count
        class_selected_sum[class_id] = np.sum(div_features_np[init_indices], axis=0, dtype=np.float32)

    selected_count_history: list[int] = [int(np.sum(selected_mask))]
    round_id = 0
    total_score_acc = 0.0

    while True:
        remaining_by_class = class_budgets - class_selected_counts
        active_classes = np.flatnonzero(remaining_by_class > 0).astype(np.int64)
        if active_classes.size == 0:
            break
        round_id += 1
        remain_total = int(np.sum(remaining_by_class))
        if remain_total < active_classes.size:
            chosen_classes = np.sort(rng.choice(active_classes, size=remain_total, replace=False).astype(np.int64))
        else:
            chosen_classes = active_classes

        for class_id in chosen_classes:
            class_indices = class_indices_list[int(class_id)]
            unselected_mask = selected_mask[class_indices] == 0
            candidate_indices = class_indices[unselected_mask]
            if candidate_indices.size == 0:
                continue

            current_count = int(class_selected_counts[class_id])
            if current_count <= 0:
                continue
            current_sum = class_selected_sum[class_id]
            mu_full = full_class_mean_f32[class_id]
            mu_sub = current_sum / float(current_count)
            old_dist = float(np.linalg.norm(mu_sub - mu_full))

            dynamic_k = max(3, int(ceil(0.05 * current_count)))

            candidate_features_t = torch.as_tensor(div_features_np[candidate_indices], dtype=torch.float32, device=device)
            reference_indices = class_indices[selected_mask[class_indices] > 0]
            reference_features_t = torch.as_tensor(div_features_np[reference_indices], dtype=torch.float32, device=device)
            div_raw = div_metric._knn_mean_distance_to_reference(
                query_features=candidate_features_t,
                reference_features=reference_features_t,
                k=float(dynamic_k),
                query_indices=torch.as_tensor(candidate_indices, dtype=torch.long, device=device),
                reference_indices=torch.as_tensor(reference_indices, dtype=torch.long, device=device),
            ).detach().cpu().numpy().astype(np.float32)
            div_scores = _quantile_minmax(div_raw)

            candidate_features_np = div_features_np[candidate_indices]
            mu_new = (current_sum[None, :] + candidate_features_np) / float(current_count + 1)
            new_dist = np.linalg.norm(mu_new - mu_full[None, :], axis=1)
            dist_improve = (old_dist - new_dist).astype(np.float32)
            dist_scores = _quantile_minmax(dist_improve)

            combined_scores = (
                weights["sa"] * sa_scores_np[candidate_indices]
                + weights["dds"] * dds_static_np[candidate_indices]
                + weights["div"] * div_scores
                + dist_weight * dist_scores
            ).astype(np.float32)
            rank = np.argsort(-combined_scores, kind="mergesort")
            pool_n = min(candidate_pool_size, candidate_indices.size)
            pool_indices = candidate_indices[rank[:pool_n]]
            if pool_n == 1:
                picked_idx = int(pool_indices[0])
            else:
                picked_idx = int(rng.choice(pool_indices, size=1, replace=False)[0])

            selected_mask[picked_idx] = 1
            class_selected_counts[class_id] += 1
            class_selected_sum[class_id] += div_features_np[picked_idx]
            total_score_acc += float(np.max(combined_scores))
            selected_count_history.append(int(np.sum(selected_mask)))
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(
                    f"{progress_prefix}kr={keep_ratio} selected={int(np.sum(selected_mask))}/{int(np.sum(class_budgets))} active={int(active_classes.size)}"
                )

    final_mask = selected_mask.astype(np.uint8)
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    final_div_scores = np.asarray(
        div_metric.score_dataset_dynamic(
            div_loader,
            adapter=image_adapter,
            selected_mask=final_mask,
            image_features=div_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    selected_bool = final_mask.astype(bool)
    subset_comprehensive_score = float(
        np.sum(
            (
                weights["sa"] * sa_scores_np
                + weights["dds"] * dds_static_np
                + weights["div"] * final_div_scores
            )[selected_bool],
            dtype=np.float64,
        )
    )

    class_shift_values: list[float] = []
    for class_id in range(num_classes):
        if class_selected_counts[class_id] <= 0:
            continue
        mu_sub = class_selected_sum[class_id] / float(class_selected_counts[class_id])
        mu_full = full_class_mean_f32[class_id]
        class_shift_values.append(float(np.linalg.norm(mu_sub - mu_full)))
    distribution_shift = float(np.mean(class_shift_values)) if class_shift_values else 0.0

    stats: dict[str, object] = {
        "solver": "group_classwise_greedy_add_test",
        "sr": float(sr),
        "final_rate": float(final_mask.mean()),
        "selected_by_class": selected_by_class,
        "class_budgets": {int(c): int(v) for c, v in enumerate(class_budgets.tolist())},
        "init_per_class": {int(c): int(v) for c, v in enumerate(init_per_class.tolist())},
        "candidate_pool_size": int(candidate_pool_size),
        "selected_count_history": selected_count_history,
        "accumulated_greedy_score": float(total_score_acc),
        "subset_comprehensive_score": subset_comprehensive_score,
        "distribution_shift": distribution_shift,
        "dist_weight": float(dist_weight),
    }
    return final_mask, selected_by_class, stats


def mask_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(mask_a, dtype=np.uint8).astype(bool)
    b = np.asarray(mask_b, dtype=np.uint8).astype(bool)
    if a.shape != b.shape:
        raise ValueError("mask 形状不一致。")
    same = float(np.mean(a == b))
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    a_count = int(np.sum(a))
    b_count = int(np.sum(b))
    overlap_selected = float(inter / a_count) if a_count > 0 else 0.0
    jaccard = float(inter / union) if union > 0 else 1.0
    return {
        "same_ratio": same,
        "selected_overlap_ratio": overlap_selected,
        "jaccard": jaccard,
        "intersection": float(inter),
        "union": float(union),
        "naive_count": float(a_count),
        "learned_count": float(b_count),
    }


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    keep_ratios = cmm.parse_ratio_list(args.kr)
    if not keep_ratios:
        raise ValueError("kr 参数不能为空。")

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    seed = int(args.seed)
    set_seed(seed)

    weights_path = cmm.PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = cmm.ensure_scoring_weights(weights_path, dataset_name)

    dataset_for_names = cmm._build_dataset(dataset_name, transform=None)
    class_names = resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=str(cmm.PROJECT_ROOT / "data"),
        class_names=dataset_for_names.classes,  # type: ignore[attr-defined]
    )
    labels = np.asarray(dataset_for_names.targets)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=str(cmm.PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )

    batch_size = 128
    num_workers = 4
    dds_loader = cmm.build_score_loader(dds_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    div_loader = cmm.build_score_loader(div_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    sa_loader = cmm.build_score_loader(sa_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=dataset_name,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=seed,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    num_samples = len(dataset_for_names)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(dds_loader, adapter=image_adapter).scores
        div_scores_local = div_metric.score_dataset(div_loader, adapter=image_adapter).scores
        sa_scores_local = sa_metric.score_dataset(
            sa_loader,
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
        cache_root=cmm.PROJECT_ROOT / "static_scores",
        dataset=dataset_name,
        seed=seed,
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

    sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)
    dds_scores_np = np.asarray(static_scores["dds"], dtype=np.float32)

    total_budget_to_add = 0
    for kr in keep_ratios:
        sr = float(kr) / 100.0
        target_size = int(round(sr * num_samples))
        class_indices_list = [np.flatnonzero(labels == c).astype(np.int64) for c in range(len(class_names))]
        budgets = _allocate_class_budgets(class_indices_list, sr, target_size)
        init_counts = sum(min(3, int(b), int(class_indices_list[c].size)) for c, b in enumerate(budgets))
        total_budget_to_add += 2 * max(0, int(np.sum(budgets)) - init_counts)

    pbar = tqdm(total=total_budget_to_add, desc="[test] compare naive vs learned group masks", unit="sample")

    results: list[dict[str, float | int | str]] = []
    for kr in keep_ratios:
        naive_weights = cmm.load_scoring_weights(all_weights, "naive", seed)
        learned_weights = cmm.load_scoring_weights(all_weights, "learned", seed)

        naive_mask, _, naive_stats = select_group_mask_with_linear_dist_weight(
            sa_scores=sa_scores_np,
            dds_static_scores=dds_scores_np,
            div_metric=div_metric,
            div_loader=div_loader,
            image_adapter=image_adapter,
            labels=labels,
            weights=naive_weights,
            num_classes=len(class_names),
            keep_ratio=kr,
            device=device,
            dataset_name=dataset_name,
            seed=seed,
            clip_model=args.clip_model,
            adapter_image_path=str(adapter_paths["image_path"]),
            group_candidate_pool_size=args.group_candidate_pool_size,
            progress=pbar,
            progress_prefix="naive ",
        )
        learned_mask, _, learned_stats = select_group_mask_with_linear_dist_weight(
            sa_scores=sa_scores_np,
            dds_static_scores=dds_scores_np,
            div_metric=div_metric,
            div_loader=div_loader,
            image_adapter=image_adapter,
            labels=labels,
            weights=learned_weights,
            num_classes=len(class_names),
            keep_ratio=kr,
            device=device,
            dataset_name=dataset_name,
            seed=seed,
            clip_model=args.clip_model,
            adapter_image_path=str(adapter_paths["image_path"]),
            group_candidate_pool_size=args.group_candidate_pool_size,
            progress=pbar,
            progress_prefix="learned ",
        )

        overlap = mask_overlap(naive_mask, learned_mask)
        result = {
            "dataset": dataset_name,
            "seed": seed,
            "kr": int(kr),
            "dist_weight": float(naive_stats["dist_weight"]),
            "same_ratio": overlap["same_ratio"],
            "selected_overlap_ratio": overlap["selected_overlap_ratio"],
            "jaccard": overlap["jaccard"],
            "intersection": overlap["intersection"],
            "union": overlap["union"],
            "naive_count": overlap["naive_count"],
            "learned_count": overlap["learned_count"],
            "naive_subset_score": float(naive_stats["subset_comprehensive_score"]),
            "learned_subset_score": float(learned_stats["subset_comprehensive_score"]),
            "naive_shift": float(naive_stats["distribution_shift"]),
            "learned_shift": float(learned_stats["distribution_shift"]),
        }
        results.append(result)

    pbar.close()

    print("\n=== naive vs learned group mask overlap ===")
    header = (
        f"{'kr':>4} | {'w_dist':>6} | {'same':>8} | {'sel_overlap':>11} | {'jaccard':>8} | "
        f"{'naive_shift':>11} | {'learned_shift':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{int(row['kr']):>4} | "
            f"{float(row['dist_weight']):>6.3f} | "
            f"{float(row['same_ratio']):>8.4f} | "
            f"{float(row['selected_overlap_ratio']):>11.4f} | "
            f"{float(row['jaccard']):>8.4f} | "
            f"{float(row['naive_shift']):>11.6f} | "
            f"{float(row['learned_shift']):>13.6f}"
        )

    elapsed = time.perf_counter() - total_start
    print(f"\nDone. dataset={dataset_name} seed={seed} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
