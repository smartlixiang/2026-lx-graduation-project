from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from learn_scoring_weights import (  # noqa: E402
    fit_ridge_regression_nonnegative,
    resolve_default_proxy_epochs,
    resolve_dynamic_component_cache_path,
)
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402


EXPERIMENT_KEY = "ablation_dds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation script: remove DDS from learned_group / learned_topk pipeline."
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS, help="Target dataset name.")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"), help="Dataset root path.")
    parser.add_argument("--kr", type=str, default="20,30,40,50,60,70,80,90", help="keep_ratio list (percent), comma separated or single value.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model name.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu")
    parser.add_argument("--seed", type=str, default=",".join(str(s) for s in CONFIG.exp_seeds), help="Experiment seeds, comma separated.")
    parser.add_argument("--method", type=str, default="group", choices=["topk", "group"], help="Selection method.")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Model name used in bookkeeping / compatibility.")
    parser.add_argument("--skip-saved", action="store_true", help="Skip mask generation when the target mask already exists.")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1, help="Candidate pool size in group mode.")
    parser.add_argument("--debug-prompts", action="store_true", help="Print a few Tiny-ImageNet prompts for debugging.")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--proxy-training-seed", type=int, default=CONFIG.global_seed, help="Only used for locating already-computed dynamic caches.")
    parser.add_argument("--weights-json", type=str, default="weights/ablation_weights.json")
    parser.add_argument("--static-cache-root", type=str, default="static_scores")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument("--dds-k", type=int, default=5)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.8)
    parser.add_argument("--dds-eigval-lower-bound", type=float, default=0.02)
    parser.add_argument("--dds-eigval-upper-bound", type=float, default=0.20)
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def parse_ratio_list(ratio_text: str) -> list[int]:
    cleaned = ratio_text.strip()
    if not cleaned:
        return []
    if "," in cleaned:
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [cleaned]
    return [int(item) for item in items]


def _format_percent(value: float) -> str:
    return f"{int(round(value * 100))}%"


def _format_div_k(div_k: float) -> str:
    if float(div_k).is_integer() and div_k >= 1:
        return str(int(div_k))
    if 0 < div_k < 1:
        return _format_percent(div_k)
    raise ValueError("div_k must be a positive integer or a ratio in (0,1).")


def build_static_cache_dir(cache_root: Path, dataset: str, seed: int, div_k: float, dds_lower: float, dds_upper: float) -> Path:
    param_dir = f"Div_{_format_div_k(div_k)}_DDS_[{_format_percent(dds_lower)}-{_format_percent(dds_upper)}]"
    return cache_root / dataset / str(int(seed)) / param_dir


def load_metric_cache(cache_dir: Path, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"{metric_name}_cache.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required static cache file not found: {cache_path}\n"
            "This script loads static indicators from local cache only and will not recompute them."
        )
    data = np.load(cache_path, allow_pickle=False)
    required = {"scores", "labels", "indices", "meta"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"Invalid static cache file: {cache_path}")
    scores = np.asarray(data["scores"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    indices = np.asarray(data["indices"], dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or indices.ndim != 1:
        raise ValueError(f"Invalid array ranks in static cache: {cache_path}")
    if not np.array_equal(indices, np.arange(indices.shape[0], dtype=np.int64)):
        raise ValueError(f"Static cache indices are not in global order: {cache_path}")
    return scores, labels


def load_dynamic_component_cache(cache_path: Path, component_name: str) -> tuple[np.ndarray, np.ndarray]:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required dynamic cache file not found: {cache_path}\n"
            "This script loads dynamic components from local cache only and will not recompute them."
        )
    data = np.load(cache_path, allow_pickle=False)
    required = {"component_name", "labels", "final_normalized"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"Invalid dynamic cache file: {cache_path}")
    cache_component = str(np.asarray(data["component_name"]).item())
    if cache_component.strip().upper() != component_name.strip().upper():
        raise ValueError(f"Dynamic cache component mismatch: expected {component_name}, got {cache_component} @ {cache_path}")
    labels = np.asarray(data["labels"], dtype=np.int64)
    final_normalized = np.asarray(data["final_normalized"], dtype=np.float32)
    if labels.ndim != 1 or final_normalized.ndim != 1 or labels.shape[0] != final_normalized.shape[0]:
        raise ValueError(f"Invalid dynamic cache shapes: {cache_path}")
    if not np.all(np.isfinite(final_normalized)):
        raise ValueError(f"Dynamic cache contains NaN/inf: {cache_path}")
    return final_normalized, labels


def ensure_ablation_weights_json(path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                return loaded
    return {}


def save_ablation_weight_record(path: Path, *, dataset_name: str, seed: int, payload: dict[str, object]) -> None:
    data = ensure_ablation_weights_json(path)
    dataset_entry = data.get(dataset_name)
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
    experiment_entry = dataset_entry.get(EXPERIMENT_KEY)
    if not isinstance(experiment_entry, dict):
        experiment_entry = {}
    experiment_entry[str(seed)] = payload
    dataset_entry[EXPERIMENT_KEY] = experiment_entry
    data[dataset_name] = dataset_entry
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_dataset(dataset_name: str, data_root: str, transform) -> datasets.VisionDataset:
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_score_loader(preprocess, dataset_name: str, data_root: str, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    dataset = _build_dataset(dataset_name, data_root, preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")


def load_class_names(dataset_name: str, data_root: str) -> list[str]:
    dataset = _build_dataset(dataset_name, data_root, transform=None)
    # type: ignore[attr-defined]
    return list(resolve_class_names_for_prompts(dataset_name=dataset_name, data_root=data_root, class_names=dataset.classes))


def _hash_file(path: Path) -> str:
    import hashlib
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mean_stats_cache_path(dataset_name: str, clip_model: str, adapter_image_path: str) -> Path:
    adapter_sha1 = _hash_file(Path(adapter_image_path))
    clip_tag = clip_model.replace("/", "-").replace(" ", "_")
    return PROJECT_ROOT / "static_scores" / "group_mean_stats" / dataset_name / clip_tag / f"img_adapter_{adapter_sha1}.npz"


def _get_or_compute_group_mean_stats(*, cache_path: Path, image_features: np.ndarray, labels: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(image_features.shape[0])
    feat_dim = int(image_features.shape[1]) if image_features.ndim == 2 else 0
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_n = int(np.asarray(cached["n_samples"]).item())
            cached_dim = int(np.asarray(cached["feat_dim"]).item())
            cached_cls = int(np.asarray(cached["num_classes"]).item())
            means = np.asarray(cached["full_class_mean"], dtype=np.float32)
            vars_ = np.asarray(cached["full_class_var"], dtype=np.float32)
            if cached_n == n_samples and cached_dim == feat_dim and cached_cls == int(num_classes) and means.shape == (num_classes, feat_dim) and vars_.shape == (num_classes,):
                return means, vars_
        except Exception:
            pass

    full_class_mean = np.zeros((num_classes, feat_dim), dtype=np.float32)
    full_class_var = np.zeros((num_classes,), dtype=np.float32)
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_feats = image_features[class_mask]
        if class_feats.shape[0] == 0:
            continue
        class_mean = np.mean(class_feats, axis=0, dtype=np.float32)
        diff = class_feats - class_mean
        sigma2 = float(np.mean(np.sum(diff * diff, axis=1)))
        full_class_mean[class_id] = class_mean
        full_class_var[class_id] = np.float32(max(sigma2, 0.0))

    np.savez_compressed(
        cache_path,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        n_samples=np.asarray(n_samples, dtype=np.int64),
        feat_dim=np.asarray(feat_dim, dtype=np.int64),
        num_classes=np.asarray(num_classes, dtype=np.int64),
    )
    return full_class_mean, full_class_var


def select_topk_mask_two_metrics(sa_scores: np.ndarray, div_scores: np.ndarray, labels: np.ndarray, num_classes: int, keep_ratio: int, weights: dict[str, float]) -> tuple[np.ndarray, dict[int, int]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr must be in [1,100].")
    total_scores = weights["sa"] * sa_scores + weights["div"] * div_scores
    mask = np.zeros(total_scores.shape[0], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    ratio = keep_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        num_select = class_indices.size if keep_ratio == 100 else max(1, int(class_indices.size * ratio))
        class_scores = total_scores[class_indices]
        topk_indices = class_indices[np.argpartition(-class_scores, num_select - 1)[:num_select]]
        mask[topk_indices] = 1
        selected_by_class[class_id] = int(num_select)
    return mask, selected_by_class


def select_group_mask_ablation_dds(
    sa_scores: np.ndarray,
    div_metric: Div,
    div_loader: DataLoader,
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
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr must be in [1,100].")
    num_samples = sa_scores.shape[0]
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    sr = float(keep_ratio) / 100.0
    target_size = int(round(sr * num_samples))
    target_size = min(num_samples, max(1, target_size)) if num_samples > 0 else 0
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    class_indices_list = [np.flatnonzero(labels_np == c).astype(np.int64) for c in range(num_classes)]
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = (div_features.detach().cpu().numpy() if isinstance(div_features, torch.Tensor) else np.asarray(div_features)).astype(np.float32)

    mean_stats_cache_path = _mean_stats_cache_path(dataset_name=dataset_name, clip_model=clip_model, adapter_image_path=adapter_image_path)
    full_class_mean, _ = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path, image_features=div_features_np, labels=labels_np, num_classes=num_classes)
    full_class_mean_f32 = full_class_mean.astype(np.float32, copy=False)

    def _quantile_minmax(values: np.ndarray, low_q: float = 0.002, high_q: float = 0.998) -> np.ndarray:
        if values.size == 0:
            return np.zeros(0, dtype=np.float32)
        low = float(np.quantile(values, low_q))
        high = float(np.quantile(values, high_q))
        if abs(high - low) <= 1e-12:
            return np.full(values.shape, 0.5, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _allocate_class_budgets() -> np.ndarray:
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
            raise RuntimeError("Class-budget allocation failed.")
        return budgets

    class_budgets = _allocate_class_budgets()
    candidate_pool_size = max(1, int(group_candidate_pool_size))
    dist_weight = max(0.0, 1.0 - 0.01 * float(keep_ratio))

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    class_selected_counts = np.zeros(num_classes, dtype=np.int64)
    class_selected_sum = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    init_per_class = np.zeros(num_classes, dtype=np.int64)
    static_init_score = (weights["sa"] * sa_scores_np).astype(np.float32)

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

    total_to_add = int(np.sum(class_budgets) - np.sum(init_per_class))
    selected_count_history: list[int] = [int(np.sum(selected_mask))]
    total_score_acc = 0.0
    pbar = tqdm(total=total_to_add, desc="[ablation_dds group] classwise greedy add", unit="sample")
    while True:
        remaining_by_class = class_budgets - class_selected_counts
        active_classes = np.flatnonzero(remaining_by_class > 0).astype(np.int64)
        if active_classes.size == 0:
            break
        remain_total = int(np.sum(remaining_by_class))
        chosen_classes = np.sort(rng.choice(active_classes, size=remain_total, replace=False).astype(np.int64)
                                 ) if remain_total < active_classes.size else active_classes
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

            combined_scores = (weights["sa"] * sa_scores_np[candidate_indices] + weights["div"]
                               * div_scores + dist_weight * dist_scores).astype(np.float32)
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
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    final_div_scores = np.asarray(
        div_metric.score_dataset_dynamic(div_loader, adapter=image_adapter, selected_mask=final_mask,
                                         image_features=div_features, labels=labels_t).scores,
        dtype=np.float32,
    )
    selected_bool = final_mask.astype(bool)
    subset_comprehensive_score = float(np.sum((weights["sa"] * sa_scores_np + weights["div"] * final_div_scores)[selected_bool], dtype=np.float64))

    class_shift_values: list[float] = []
    for class_id in range(num_classes):
        if class_selected_counts[class_id] <= 0:
            continue
        mu_sub = class_selected_sum[class_id] / float(class_selected_counts[class_id])
        mu_full = full_class_mean_f32[class_id]
        class_shift_values.append(float(np.linalg.norm(mu_sub - mu_full)))
    distribution_shift = float(np.mean(class_shift_values)) if class_shift_values else 0.0

    stats: dict[str, object] = {
        "solver": "group_classwise_greedy_add_ablation_dds",
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
        "distribution_shift": distribution_shift,
    }
    return final_mask, selected_by_class, stats


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    if dataset_name != CIFAR100:
        print(f"WARNING: ablation_dds.py is primarily designed for cifar100; current dataset={dataset_name}")

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    method = args.method.strip().lower()
    if method not in {"topk", "group"}:
        raise ValueError("method must be one of {'topk','group'}")

    keep_ratios = parse_ratio_list(args.kr)
    if not keep_ratios:
        raise ValueError("kr must not be empty.")
    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("seed must not be empty.")

    resolved_proxy_epochs = int(args.proxy_epochs) if args.proxy_epochs is not None else resolve_default_proxy_epochs(dataset_name)
    static_cache_root = PROJECT_ROOT / args.static_cache_root
    weights_json_path = PROJECT_ROOT / args.weights_json

    class_names = load_class_names(dataset_name, args.data_root)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device, k=args.div_k)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=args.data_root,
        debug_prompts=args.debug_prompts,
    )
    div_loader = build_score_loader(div_metric.extractor.preprocess, dataset_name, args.data_root, device, args.batch_size, args.num_workers)
    _ = sa_metric  # keep construction aligned with current project style

    component_arrays: dict[str, np.ndarray] = {}
    component_cache_paths: dict[str, str] = {}
    labels_dynamic_ref: np.ndarray | None = None
    for component_name in ("A", "C", "T", "P"):
        cache_path = PROJECT_ROOT / resolve_dynamic_component_cache_path(dataset_name, args.proxy_model, resolved_proxy_epochs, component_name)
        arr, labels_dynamic = load_dynamic_component_cache(cache_path, component_name)
        if labels_dynamic_ref is None:
            labels_dynamic_ref = labels_dynamic
        elif not np.array_equal(labels_dynamic_ref, labels_dynamic):
            raise RuntimeError(f"Dynamic cache label mismatch for component {component_name}.")
        component_arrays[component_name] = arr.astype(np.float32)
        component_cache_paths[component_name] = str(cache_path)

    assert labels_dynamic_ref is not None
    dynamic_labels = labels_dynamic_ref
    dynamic_targets = (component_arrays["A"] + component_arrays["C"] + 0.5 * component_arrays["T"] + 0.5 * component_arrays["P"]) / 3.0
    dynamic_targets = np.asarray(dynamic_targets, dtype=np.float64)

    total_tasks = len(seeds) * len(keep_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)
        print(f"\n===== Seed {seed} =====")
        static_cache_dir = build_static_cache_dir(static_cache_root, dataset_name, seed, args.div_k,
                                                  args.dds_eigval_lower_bound, args.dds_eigval_upper_bound)
        if not static_cache_dir.is_dir():
            raise FileNotFoundError(
                f"Static cache directory not found: {static_cache_dir}\n"
                "This script loads SA/Div/DDS from local cache only and will not recompute them."
            )

        sa_scores, labels_sa = load_metric_cache(static_cache_dir, "SA")
        div_scores, labels_div = load_metric_cache(static_cache_dir, "Div")
        _dds_scores, labels_dds = load_metric_cache(static_cache_dir, "DDS")
        if not np.array_equal(labels_sa, labels_div) or not np.array_equal(labels_sa, labels_dds):
            raise RuntimeError("Static cache label mismatch among SA/Div/DDS.")
        if not np.array_equal(labels_sa, dynamic_labels):
            raise RuntimeError("Dynamic cache labels and static cache labels do not match.")

        static_features = np.stack([sa_scores, div_scores], axis=1).astype(np.float64)
        weights_vec, bias = fit_ridge_regression_nonnegative(
            static_features, dynamic_targets, args.ridge_lambda, args.learning_rate, args.max_iter, args.tol)
        weights = {"sa": float(weights_vec[0]), "div": float(weights_vec[1]), "bias": float(bias)}

        image_adapter, _text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=div_metric.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()

        save_ablation_weight_record(
            weights_json_path,
            dataset_name=dataset_name,
            seed=seed,
            payload={
                "experiment": EXPERIMENT_KEY,
                "static_components": ["SA", "Div"],
                "dynamic_components": ["A", "C", "T", "P"],
                "sa": float(weights_vec[0]),
                "div": float(weights_vec[1]),
                "bias": float(bias),
                "ridge_lambda": float(args.ridge_lambda),
                "proxy_model": args.proxy_model,
                "proxy_epochs": int(resolved_proxy_epochs),
                "proxy_training_seed": int(args.proxy_training_seed),
                "dynamic_cache_paths": component_cache_paths,
                "static_cache_dir": str(static_cache_dir),
                "adapter_image_path": str(adapter_paths["image_path"]),
                "adapter_text_path": str(adapter_paths["text_path"]),
            },
        )
        print(f"[weights] saved {EXPERIMENT_KEY} weights for seed={seed}: SA={weights['sa']:.6f}, Div={weights['div']:.6f}, bias={weights['bias']:.6f}")

        for keep_ratio in keep_ratios:
            task_idx += 1
            mask_path = resolve_mask_path(mode=EXPERIMENT_KEY, dataset=dataset_name, model=args.model_name, seed=seed, keep_ratio=keep_ratio)
            if args.skip_saved and mask_path.exists():
                print(f"[{task_idx}/{total_tasks}] skip saved mask: {mask_path}")
                continue

            print(f"[{task_idx}/{total_tasks}] dataset={dataset_name} seed={seed} kr={keep_ratio} method={method} -> {mask_path}")

            if method == "topk":
                mask, selected_by_class = select_topk_mask_two_metrics(
                    sa_scores=sa_scores, div_scores=div_scores, labels=labels_sa, num_classes=len(class_names), keep_ratio=keep_ratio, weights=weights)
                group_stats = None
            else:
                mask, selected_by_class, group_stats = select_group_mask_ablation_dds(
                    sa_scores=sa_scores,
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels_sa,
                    weights=weights,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                    device=device,
                    dataset_name=dataset_name,
                    seed=seed,
                    clip_model=args.clip_model,
                    adapter_image_path=str(adapter_paths["image_path"]),
                    group_candidate_pool_size=args.group_candidate_pool_size,
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            elapsed = time.perf_counter() - total_start
            print(f"[mask] saved {mask_path} | selected={int(mask.sum())} | final_rate={float(mask.mean()):.6f} | elapsed={elapsed:.2f}s")
            if group_stats is not None:
                print(
                    f"[group] dist_weight={group_stats['dist_weight']:.6f} | "
                    f"subset_score={group_stats['subset_comprehensive_score']:.6f} | "
                    f"distribution_shift={group_stats['distribution_shift']:.6f}"
                )
            print(f"[class count preview] {dict(list(selected_by_class.items())[:5])}")

    print(f"\nDone. Ablation weights saved to {weights_json_path}")
    print(f"Masks saved under {PROJECT_ROOT / 'mask' / EXPERIMENT_KEY}")


if __name__ == "__main__":
    main()
