"""Learned-group static-metric ablations.

This module contains the shared type-1 solver used by the three entry points.  It
deliberately reads ACT caches only: an ablation run never trains a proxy or
falls back to proxy logs.
"""
from __future__ import annotations

import argparse
import time
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from calculate_my_mask import _build_dataset, build_score_loader, compute_full_class_means, parse_ratio_list
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10
from learn_scoring_weights import resolve_default_proxy_epochs, resolve_dynamic_component_cache_path
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.path_rules import resolve_mask_path
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.seed import parse_seed_list, set_seed
from utils.static_score_cache import get_or_compute_static_scores
from weights.calibration import build_dynamic_target, fit_softplus_ratio_regression

PROJECT_ROOT = Path(__file__).resolve().parent
COMPONENT_NAMES = ("A", "C", "T")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--kr", default="20,30,40,50,60,70,80,90")
    parser.add_argument("--seed", default=",".join(map(str, CONFIG.exp_seeds)))
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=5)
    parser.add_argument("--group-init-count", type=int, default=10)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--ratio-lambda", type=float, default=1e-2)
    parser.add_argument("--regression-learning-rate", type=float, default=2e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser


def load_dynamic_target(dataset: str, proxy_model: str, seed: int, epochs: int, num_samples: int) -> np.ndarray:
    """Load only ``final_normalized`` from the canonical per-seed ACT caches."""
    results = {}
    expected_length = int(num_samples)
    for name in COMPONENT_NAMES:
        path = resolve_dynamic_component_cache_path(dataset, proxy_model, seed, epochs, name)
        if not path.is_file():
            raise FileNotFoundError(f"Required dynamic cache is missing: {path}")
        with np.load(path, allow_pickle=False) as cache:
            if "final_normalized" not in cache:
                raise ValueError(f"Dynamic cache has no final_normalized array: {path}")
            values = np.asarray(cache["final_normalized"], dtype=np.float64)
        if values.shape != (expected_length,):
            raise ValueError(f"{path}: expected final_normalized shape {(expected_length,)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: final_normalized contains NaN/inf")
        results[name] = SimpleNamespace(final_normalized=values)
    dynamic_target, _ = build_dynamic_target(results)
    return dynamic_target


def select_ablation_group_mask(
    *, active: tuple[str, str], scores: dict[str, np.ndarray], div_metric: Div,
    div_loader: DataLoader, image_adapter, labels: np.ndarray, weights: dict[str, float],
    num_classes: int, keep_ratio: int, device: torch.device, seed: int,
    candidate_pool_size: int, group_init_count: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Current type-1 solver with exactly the named static metrics present."""
    if not 0 < keep_ratio <= 100:
        raise ValueError("kr must be in [1, 100]")
    active_set = set(active)
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    class_indices = [np.flatnonzero(labels == c).astype(np.int64) for c in range(num_classes)]
    target = min(n, max(1, int(round(keep_ratio / 100.0 * n))))
    sizes = np.asarray([x.size for x in class_indices], dtype=np.int64)
    raw_budgets = sizes * (keep_ratio / 100.0)
    budgets = np.minimum(np.floor(raw_budgets).astype(np.int64), sizes)
    need = target - int(budgets.sum())
    for class_id in np.lexsort((np.arange(num_classes), -(raw_budgets - budgets))):
        if need <= 0:
            break
        if budgets[class_id] < sizes[class_id]:
            budgets[class_id] += 1
            need -= 1
    if need:
        raise RuntimeError("Could not allocate exact class budgets")

    rng = np.random.default_rng(seed)
    features_t, _ = div_metric._encode_images(div_loader, image_adapter)
    features = (features_t.detach().cpu().numpy() if isinstance(features_t, torch.Tensor) else np.asarray(features_t)).astype(np.float32)
    full_means = compute_full_class_means(features, labels, num_classes)
    selected = np.zeros(n, dtype=np.uint8)
    counts = np.zeros(num_classes, dtype=np.int64)
    sums = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    init_per_class = np.zeros(num_classes, dtype=np.int64)

    for c, indices in enumerate(class_indices):
        init_count = min(max(0, int(group_init_count)), int(budgets[c]), indices.size)
        if init_count == 0:
            continue
        if "sa" not in active_set:  # SA ablation: uniform over the complete class.
            init_pool = indices
        else:  # Main type-1 initialization: uniform over the raw-SA top 50% pool.
            pool_size = min(indices.size, max(init_count, int(np.ceil(0.5 * indices.size))))
            init_pool = indices[np.argsort(-scores["sa"][indices], kind="mergesort")[:pool_size]]
        chosen = init_pool if init_pool.size <= init_count else rng.choice(init_pool, init_count, replace=False)
        selected[chosen] = 1
        counts[c] = init_count
        init_per_class[c] = init_count
        sums[c] = features[chosen].sum(axis=0, dtype=np.float32)

    history = [int(selected.sum())]
    total_to_add = int(budgets.sum() - init_per_class.sum())
    dist_max = max(0.0, 0.7 - 0.004 * keep_ratio)
    dist_min = 0.5 * dist_max
    pool_size = max(1, int(candidate_pool_size))
    pbar = tqdm(total=total_to_add, desc="[ablation group] classwise greedy add", unit="sample")
    while True:
        remaining = budgets - counts
        active_classes = np.flatnonzero(remaining > 0)
        if not active_classes.size:
            break
        remain_total = int(remaining.sum())
        chosen_classes = (np.sort(rng.choice(active_classes, remain_total, replace=False))
                          if remain_total < active_classes.size else active_classes)
        for c in chosen_classes:
            candidates = class_indices[c][selected[class_indices[c]] == 0]
            current_count = int(counts[c])
            if candidates.size == 0 or current_count <= 0:
                continue
            progress = float(np.clip(current_count / float(budgets[c]), 0.0, 1.0))
            dist_weight = dist_min + (dist_max - dist_min) * progress
            old_dist = float(np.linalg.norm(sums[c] / current_count - full_means[c]))
            new_means = (sums[c][None, :] + features[candidates]) / float(current_count + 1)
            dist_local = standard_zscore(old_dist - np.linalg.norm(new_means - full_means[c][None, :], axis=1))
            combined = dist_weight * dist_local
            if "sa" in active_set:
                combined = combined + weights["sa"] * standard_zscore(scores["sa"][candidates])
            if "dds" in active_set:
                combined = combined + weights["dds"] * standard_zscore(scores["dds"][candidates])
            if "div" in active_set:
                refs = class_indices[c][selected[class_indices[c]] > 0]
                div_raw = div_metric._knn_mean_distance_to_reference(
                    query_features=torch.as_tensor(features[candidates], dtype=torch.float32, device=device),
                    reference_features=torch.as_tensor(features[refs], dtype=torch.float32, device=device),
                    k=float(max(3, int(ceil(0.05 * current_count)))),
                    query_indices=torch.as_tensor(candidates, dtype=torch.long, device=device),
                    reference_indices=torch.as_tensor(refs, dtype=torch.long, device=device),
                ).detach().cpu().numpy().astype(np.float32)
                combined = combined + weights["div"] * standard_zscore(div_raw)
            rank = np.argsort(-combined, kind="mergesort")[:min(pool_size, candidates.size)]
            candidate_pool = candidates[rank]
            picked = int(candidate_pool[0] if candidate_pool.size == 1 else rng.choice(candidate_pool))
            selected[picked] = 1
            counts[c] += 1
            sums[c] += features[picked]
            history.append(int(selected.sum()))
            pbar.update(1)
    pbar.close()

    selected_bool = selected.astype(bool)
    comprehensive = np.zeros(n, dtype=np.float64)
    for name in active:
        if name == "div":
            labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
            final_div = np.asarray(div_metric.score_dataset_dynamic(div_loader, adapter=image_adapter,
                selected_mask=selected, image_features=features_t, labels=labels_t).scores)
            comprehensive += weights[name] * standard_zscore_by_class(final_div, labels)
        else:
            comprehensive += weights[name] * standard_zscore_by_class(scores[name], labels)
    shifts = [np.linalg.norm(sums[c] / counts[c] - full_means[c]) for c in range(num_classes) if counts[c] > 0]
    stats = {
        "solver": "group_classwise_greedy_add", "type": 1,
        "dist_weight": float(dist_max), "dist_weight_max": float(dist_max), "dist_weight_min": float(dist_min),
        "dist_weight_schedule": "linear_increase_by_class_progress",
        "selected_by_class": {c: int(counts[c]) for c in range(num_classes)},
        "class_budgets": {c: int(budgets[c]) for c in range(num_classes)},
        "init_per_class": {c: int(init_per_class[c]) for c in range(num_classes)},
        "candidate_pool_size": pool_size, "selected_count_history": history,
        "subset_comprehensive_score": float(comprehensive[selected_bool].sum()),
        "distribution_shift": float(np.mean(shifts)) if shifts else 0.0,
    }
    return selected, stats


def run_ablation(args: argparse.Namespace, *, active: tuple[str, str], mode: str) -> None:
    device = torch.device(args.device) if args.device else CONFIG.global_device
    dataset_name = args.dataset.strip().lower()
    epochs = args.proxy_epochs if args.proxy_epochs is not None else resolve_default_proxy_epochs(dataset_name)
    dataset = _build_dataset(dataset_name, transform=None)
    class_names = list(resolve_class_names_for_prompts(dataset_name=dataset_name, data_root=PROJECT_ROOT / "data", class_names=dataset.classes))
    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device,
                                  dataset_name=dataset_name, data_root=str(PROJECT_ROOT / "data"), debug_prompts=args.debug_prompts)
    dds_loader = build_score_loader(dds_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, dataset_name, device, args.batch_size, args.num_workers)
    labels = np.asarray(dataset.targets, dtype=np.int64)

    for seed in parse_seed_list(args.seed):
        set_seed(seed)
        image_adapter, text_adapter, paths = load_trained_adapters(dataset_name=dataset_name, clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim, seed=seed, map_location=device)
        image_adapter.to(device).eval(); text_adapter.to(device).eval()
        def compute_scores() -> dict[str, np.ndarray]:
            return {
                "dds": np.asarray(dds_metric.score_dataset(tqdm(dds_loader, desc="Scoring DDS"), adapter=image_adapter).scores),
                "div": np.asarray(div_metric.score_dataset(tqdm(div_loader, desc="Scoring Div"), adapter=image_adapter).scores),
                "sa": np.asarray(sa_metric.score_dataset(tqdm(sa_loader, desc="Scoring SA"), adapter_image=image_adapter, adapter_text=text_adapter).scores),
                "labels": labels,
            }
        scores = get_or_compute_static_scores(cache_root=PROJECT_ROOT / "static_scores", dataset=dataset_name, seed=seed,
            clip_model=args.clip_model, adapter_image_path=str(paths["image_path"]), adapter_text_path=str(paths["text_path"]),
            div_k=div_metric.k, dds_k=dds_metric.k, dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
            dds_eigval_upper_bound=dds_metric.eigval_upper_bound, prompt_template=sa_metric.prompt_template,
            num_samples=len(dataset), compute_fn=compute_scores)
        if not np.array_equal(labels, scores["labels"]):
            raise ValueError("Static cache labels do not match the current dataset")
        # Feature order is exactly ``active`` (SA ablation: Div, DDS).
        features = np.stack([standard_zscore_by_class(scores[name], labels) for name in active], axis=1).astype(np.float64)
        target = load_dynamic_target(dataset_name, args.proxy_model, seed, int(epochs), len(dataset))
        fit = fit_softplus_ratio_regression(features, target, args.ratio_lambda, args.regression_learning_rate,
                                            args.regression_max_iter, args.regression_tol, device)
        fitted = np.asarray(fit["normalized_weights"], dtype=np.float64)
        if not np.isclose(fitted.sum(), 1.0):
            raise RuntimeError("Fitted active weights do not sum to one")
        weights = dict(zip(active, fitted.tolist()))
        print(f"[seed={seed}] active={active}, weights={weights}")
        for keep_ratio in parse_ratio_list(args.kr):
            path = resolve_mask_path(mode=mode, dataset=dataset_name, model=args.model_name, seed=seed, keep_ratio=keep_ratio)
            if args.skip_saved and path.exists():
                print(f"[skip] {path}"); continue
            start = time.perf_counter()
            mask, stats = select_ablation_group_mask(active=active, scores=scores, div_metric=div_metric,
                div_loader=div_loader, image_adapter=image_adapter, labels=labels, weights=weights,
                num_classes=len(class_names), keep_ratio=keep_ratio, device=device, seed=seed,
                candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, mask=mask.astype(np.uint8))
            print(f"[saved] {path} selected={int(mask.sum())} elapsed={time.perf_counter()-start:.2f}s stats={stats}")


def main() -> None:
    args = build_parser("Learned-group ablation without SA").parse_args()
    run_ablation(args, active=("div", "dds"), mode="ablation_sa")


if __name__ == "__main__":
    main()
