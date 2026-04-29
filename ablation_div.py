from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calculate_my_mask import (  # noqa: E402
    _build_dataset,
    build_score_loader,
    parse_ratio_list,
    select_topk_mask,
)
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR100  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402

ABLATION_NAME = "ablation_div"
ALL_COMPONENTS = ("dds", "div", "sa")
ACTIVE_COMPONENTS = ("dds", "sa")


class RatioBandDifficultyDirection(DifficultyDirection):
    """DDS variant selecting PCA directions with eigenvalue ratio in [2%, 20%]."""

    def _select_difficulty_dirs(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_dim = eigenvalues.shape[0]
        if feat_dim == 0:
            return eigenvectors[:, :0], eigenvalues[:0]
        eigvals = torch.clamp(eigenvalues, min=0.0).flip(0)
        eigvecs = eigenvectors.flip(1)
        total = eigvals.sum()
        if total.item() <= 0:
            return eigvecs[:, :1], eigvals[:1]
        ratios = eigvals / total
        keep = (ratios >= self.eigval_lower_bound) & (ratios <= self.eigval_upper_bound)
        if not bool(keep.any()):
            lower_gap = torch.clamp(self.eigval_lower_bound - ratios, min=0.0)
            upper_gap = torch.clamp(ratios - self.eigval_upper_bound, min=0.0)
            nearest = torch.argmin(lower_gap + upper_gap)
            keep = torch.zeros_like(ratios, dtype=torch.bool)
            keep[nearest] = True
        return eigvecs[:, keep], eigvals[keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate Div and calculate masks.")
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--kr", type=str, default="20,30,40,50,60,70,80,90")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=str, default="22,42,96")
    parser.add_argument("--weight-group", type=str, default="learned", choices=("naive", "learned"))
    parser.add_argument("--method", type=str, default="group", choices=("topk", "group"))
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args()


def _normalize_active(weights: dict[str, object]) -> dict[str, float]:
    out = {k: 0.0 for k in ALL_COMPONENTS}
    active_sum = 0.0
    for key in ACTIVE_COMPONENTS:
        try:
            value = max(float(weights.get(key, 0.0)), 0.0)
        except (TypeError, ValueError):
            value = 0.0
        out[key] = value
        active_sum += value
    if active_sum <= 1e-12:
        for key in ACTIVE_COMPONENTS:
            out[key] = 1.0 / len(ACTIVE_COMPONENTS)
    else:
        for key in ACTIVE_COMPONENTS:
            out[key] /= active_sum
    out["div"] = 0.0
    return out


def _load_original_weights(dataset_name: str) -> dict[str, dict[str, object]]:
    path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(dataset_name, {})
    return entry if isinstance(entry, dict) else {}


def load_ablation_weights(dataset_name: str, weight_group: str, seed: int) -> dict[str, float]:
    path = PROJECT_ROOT / "weights" / "ablation_weights.json"
    data: dict[str, object] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
    dataset_entry = data.setdefault(dataset_name, {})
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        data[dataset_name] = dataset_entry
    ablation_entry = dataset_entry.setdefault(ABLATION_NAME, {})
    if not isinstance(ablation_entry, dict):
        ablation_entry = {}
        dataset_entry[ABLATION_NAME] = ablation_entry

    key = "naive" if weight_group == "naive" else str(seed)
    selected = ablation_entry.get(key)
    if isinstance(selected, dict):
        weights = _normalize_active(selected)
    elif weight_group == "naive":
        weights = {"dds": 0.5, "div": 0.0, "sa": 0.5}
    else:
        original = _load_original_weights(dataset_name)
        fallback = original.get(str(seed)) or original.get("learned") or original.get("naive") or {}
        weights = _normalize_active(fallback if isinstance(fallback, dict) else {})
        print(
            f"[Warning] weights/ablation_weights.json missing {dataset_name}/{ABLATION_NAME}/{key}; "
            "use renormalized original weights as fallback."
        )
    ablation_entry[key] = weights
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return weights


def select_group_mask_without_div_component(
    sa_scores: np.ndarray,
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
    dds_static_scores: np.ndarray,
    group_candidate_pool_size: int = 1,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    """Group solver with the Div component removed from the greedy objective.

    The original group mode recomputes a candidate-to-selected-set Div term, so
    removing Div is not equivalent to only ignoring the cached static Div array.
    This function removes that dynamic Div term as well. It still uses CLIP image
    features for the existing mean-matching correction, which is treated as group
    optimization machinery rather than the static Div component itself.
    """
    from calculate_my_mask import _get_or_compute_group_mean_stats, _mean_stats_cache_path

    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    dds_static_np = np.asarray(dds_static_scores, dtype=np.float32)
    num_samples = sa_scores_np.shape[0]
    if labels_np.shape[0] != num_samples or dds_static_np.shape[0] != num_samples:
        raise ValueError("样本数不一致，无法执行 group。")

    sr = float(keep_ratio) / 100.0
    target_size = min(num_samples, max(1, int(round(sr * num_samples))))
    class_indices_list = [np.flatnonzero(labels_np == c).astype(np.int64) for c in range(num_classes)]
    rng = np.random.default_rng(seed)

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = div_features.detach().cpu().numpy().astype(np.float32)
    mean_stats_cache_path = _mean_stats_cache_path(dataset_name, clip_model, adapter_image_path)
    full_class_mean, _ = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    full_class_mean = full_class_mean.astype(np.float32, copy=False)

    def _quantile_minmax(values: np.ndarray, low_q: float = 0.002, high_q: float = 0.998) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        low = float(np.quantile(values, low_q))
        high = float(np.quantile(values, high_q))
        if abs(high - low) <= 1e-12:
            return np.full(values.shape, 0.5, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    class_sizes = np.asarray([idx.size for idx in class_indices_list], dtype=np.int64)
    raw = class_sizes.astype(np.float64) * sr
    class_budgets = np.minimum(np.floor(raw).astype(np.int64), class_sizes)
    need = int(target_size - np.sum(class_budgets))
    if need > 0:
        frac = raw - class_budgets.astype(np.float64)
        for class_id in np.lexsort((np.arange(num_classes, dtype=np.int64), -frac)):
            if need <= 0:
                break
            if class_budgets[class_id] < class_sizes[class_id]:
                class_budgets[class_id] += 1
                need -= 1
    if need != 0:
        raise RuntimeError("类别预算分配失败，无法满足目标总样本数。")

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
        init_indices = class_indices[np.argsort(-static_init_score[class_indices], kind="mergesort")[:init_count]]
        selected_mask[init_indices] = 1
        class_selected_counts[class_id] = init_count
        class_selected_sum[class_id] = np.sum(div_features_np[init_indices], axis=0, dtype=np.float32)

    candidate_pool_size = max(1, int(group_candidate_pool_size))
    dist_weight = max(0.0, 1.0 - 0.01 * float(keep_ratio))
    selected_count_history = [int(np.sum(selected_mask))]
    total_to_add = int(np.sum(class_budgets) - np.sum(init_per_class))
    total_score_acc = 0.0
    pbar = tqdm(total=total_to_add, desc="[group-no-div] classwise greedy add", unit="sample")
    while True:
        remaining_by_class = class_budgets - class_selected_counts
        active_classes = np.flatnonzero(remaining_by_class > 0).astype(np.int64)
        if active_classes.size == 0:
            break
        remain_total = int(np.sum(remaining_by_class))
        chosen_classes = np.sort(rng.choice(active_classes, size=remain_total, replace=False).astype(np.int64)) if remain_total < active_classes.size else active_classes
        for class_id in chosen_classes:
            class_indices = class_indices_list[int(class_id)]
            candidate_indices = class_indices[selected_mask[class_indices] == 0]
            if candidate_indices.size == 0:
                continue
            current_count = int(class_selected_counts[class_id])
            current_sum = class_selected_sum[class_id]
            mu_full = full_class_mean[class_id]
            old_dist = float(np.linalg.norm(current_sum / float(current_count) - mu_full))
            candidate_features_np = div_features_np[candidate_indices]
            mu_new = (current_sum[None, :] + candidate_features_np) / float(current_count + 1)
            dist_scores = _quantile_minmax((old_dist - np.linalg.norm(mu_new - mu_full[None, :], axis=1)).astype(np.float32))
            combined_scores = (weights["sa"] * sa_scores_np[candidate_indices] + weights["dds"] * dds_static_np[candidate_indices] + dist_weight * dist_scores).astype(np.float32)
            rank = np.argsort(-combined_scores, kind="mergesort")
            pool_n = min(candidate_pool_size, candidate_indices.size)
            pool_indices = candidate_indices[rank[:pool_n]]
            picked_idx = int(pool_indices[0]) if pool_n == 1 else int(rng.choice(pool_indices, size=1, replace=False)[0])
            selected_mask[picked_idx] = 1
            class_selected_counts[class_id] += 1
            class_selected_sum[class_id] += div_features_np[picked_idx]
            total_score_acc += float(np.max(combined_scores))
            selected_count_history.append(int(np.sum(selected_mask)))
            pbar.update(1)
            pbar.set_postfix(active_classes=int(active_classes.size))
    pbar.close()

    final_mask = selected_mask.astype(np.uint8)
    selected_by_class = {int(c): int(final_mask[class_indices_list[c]].sum()) if class_indices_list[c].size > 0 else 0 for c in range(num_classes)}
    selected_bool = final_mask.astype(bool)
    subset_comprehensive_score = float(np.sum((weights["sa"] * sa_scores_np + weights["dds"] * dds_static_np)[selected_bool], dtype=np.float64))
    class_shift_values = []
    for class_id in range(num_classes):
        if class_selected_counts[class_id] <= 0:
            continue
        mu_sub = class_selected_sum[class_id] / float(class_selected_counts[class_id])
        class_shift_values.append(float(np.linalg.norm(mu_sub - full_class_mean[class_id])))
    stats = {
        "solver": "group_classwise_greedy_add_without_div_component",
        "sr": float(sr),
        "dist_weight": float(dist_weight),
        "final_rate": float(final_mask.mean()),
        "selected_by_class": selected_by_class,
        "class_budgets": {int(c): int(v) for c, v in enumerate(class_budgets.tolist())},
        "init_per_class": {int(c): int(v) for c, v in enumerate(init_per_class.tolist())},
        "candidate_pool_size": int(candidate_pool_size),
        "selected_count_history": selected_count_history,
        "accumulated_greedy_score": float(total_score_acc),
        "subset_comprehensive_score": subset_comprehensive_score,
        "distribution_shift": float(np.mean(class_shift_values)) if class_shift_values else 0.0,
        "div_component_removed_from_group_objective": True,
    }
    return final_mask, selected_by_class, stats


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    keep_ratios = parse_ratio_list(args.kr)
    seeds = parse_seed_list(args.seed)
    if not keep_ratios or not seeds:
        raise ValueError("kr 和 seed 均不能为空。")

    dataset_for_names = _build_dataset(dataset_name, transform=None)
    class_names = resolve_class_names_for_prompts(dataset_name=dataset_name, data_root=PROJECT_ROOT / "data", class_names=dataset_for_names.classes)  # type: ignore[attr-defined]
    print(f"[Init] {dataset_name} samples={len(dataset_for_names)} classes={len(class_names)}")

    dds_metric = RatioBandDifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device, eigval_lower_bound=0.02, eigval_upper_bound=0.20)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device, dataset_name=dataset_name, data_root=str(PROJECT_ROOT / "data"), debug_prompts=args.debug_prompts)

    batch_size = 128
    num_workers = 4
    dds_loader = build_score_loader(dds_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    method_name = f"{ABLATION_NAME}_{args.weight_group}_{args.method}"
    total_tasks = len(seeds) * len(keep_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)
        weights = load_ablation_weights(dataset_name, args.weight_group, seed)
        print(f"[Seed {seed}] weights={weights}")
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(dataset_name=dataset_name, clip_model=args.clip_model, input_dim=dds_metric.extractor.embed_dim, seed=seed, map_location=device)
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()

        def _compute_scores() -> dict[str, np.ndarray]:
            dds_scores = dds_metric.score_dataset(tqdm(dds_loader, desc="Scoring DDS(2%-20%)", unit="batch"), adapter=image_adapter).scores
            div_scores = div_metric.score_dataset(tqdm(div_loader, desc="Scoring Div", unit="batch"), adapter=image_adapter).scores
            sa_scores = sa_metric.score_dataset(tqdm(sa_loader, desc="Scoring SA", unit="batch"), adapter_image=image_adapter, adapter_text=text_adapter).scores
            return {"dds": np.asarray(dds_scores), "div": np.asarray(div_scores), "sa": np.asarray(sa_scores), "labels": np.asarray(dataset_for_names.targets)}

        static_start = time.perf_counter()
        static_scores = get_or_compute_static_scores(
            cache_root=PROJECT_ROOT / "static_scores" / "ablation_2_20",
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
            num_samples=len(dataset_for_names),
            compute_fn=_compute_scores,
        )
        static_seconds = time.perf_counter() - static_start
        dds_scores_np = np.asarray(static_scores["dds"], dtype=np.float32)
        div_scores_np = np.asarray(static_scores["div"], dtype=np.float32)
        sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)
        labels = np.asarray(dataset_for_names.targets)
        total_scores_np = weights["dds"] * dds_scores_np + weights["sa"] * sa_scores_np

        for keep_ratio in keep_ratios:
            task_idx += 1
            print(f"[Mask {task_idx}/{total_tasks}] {method_name} | seed={seed} | kr={keep_ratio}")
            mask_path = resolve_mask_path(mode=method_name, dataset=dataset_name, model=args.model_name, seed=seed, keep_ratio=keep_ratio)
            if args.skip_saved and mask_path.exists():
                print(f"[Skip] {mask_path}")
                continue
            group_stats = None
            if args.method == "topk":
                mask, selected_by_class = select_topk_mask(total_scores_np, labels, num_classes=len(class_names), keep_ratio=keep_ratio)
            else:
                mask, selected_by_class, group_stats = select_group_mask_without_div_component(
                    sa_scores_np,
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels,
                    weights=weights,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                    device=device,
                    dataset_name=dataset_name,
                    seed=seed,
                    clip_model=args.clip_model,
                    adapter_image_path=str(adapter_paths["image_path"]),
                    dds_static_scores=dds_scores_np,
                    group_candidate_pool_size=args.group_candidate_pool_size,
                )
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            print(f"saved: {mask_path} | selected={int(mask.sum())} | static_seconds={static_seconds:.2f} | total_seconds={time.perf_counter() - total_start:.2f}")
            if group_stats is not None:
                print(f"group_summary: subset_score={group_stats['subset_comprehensive_score']:.6f} | distribution_shift={group_stats['distribution_shift']:.6f}")


if __name__ == "__main__":
    main()
