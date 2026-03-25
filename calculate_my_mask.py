from __future__ import annotations

import argparse
import hashlib
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

from model.adapter import load_trained_adapters  # noqa: E402
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate selection masks")
    parser.add_argument(
        "--dataset",
        type=str,
        default=CIFAR10,
        choices=AVAILABLE_DATASETS,
        help="目标数据集名称",
    )
    parser.add_argument(
        "--kr",
        type=str,
        default="20,30,40,50,60,70,80,90",
        help="keep_ratio 列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--seed",
        type=str,
        default=str(CONFIG.global_seed),
        help="随机种子列表，逗号分隔",
    )
    parser.add_argument(
        "--weight-group",
        type=str,
        default="naive",
        help="权重组，仅支持 {naive, learned}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="topk",
        help="数据选择方式，可选 {topk, group}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="mask 保存路径中的模型名称",
    )
    parser.add_argument(
        "--skip-saved",
        action="store_true",
        help="仅在显式传入该参数时，若目标 mask 已存在则跳过重新生成",
    )
    parser.add_argument(
        "--group-candidate-pool-size",
        type=int,
        default=1,
        help="group 模式下类内贪心候选池大小，1 表示纯贪心。",
    )
    return parser.parse_args()


def _build_dataset(dataset_name: str, transform) -> datasets.VisionDataset:
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


def parse_ratio_list(ratio_text: str) -> list[int]:
    cleaned = ratio_text.strip()
    if not cleaned:
        return []
    if "," in cleaned:
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [cleaned]
    return [int(item) for item in items]


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
    default_weight = 1.0 / 3.0
    for key in ("dds", "div", "sa"):
        if key not in naive:
            naive[key] = default_weight
            updated = True

    naive_total = 0.0
    for key in ("dds", "div", "sa"):
        try:
            naive[key] = float(naive[key])
        except (TypeError, ValueError):
            naive[key] = default_weight
            updated = True
        naive_total += naive[key]

    if naive_total <= 0:
        for key in ("dds", "div", "sa"):
            naive[key] = default_weight
        updated = True
    elif abs(naive_total - 1.0) > 1e-12:
        for key in ("dds", "div", "sa"):
            naive[key] /= naive_total
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


def _to_weight_triplet(selected: dict[str, object], group_name: str) -> dict[str, float]:
    required = {"dds", "div", "sa"}
    missing = required - selected.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"权重组 {group_name} 缺少必要键: {missing_str}")
    weights: dict[str, float] = {}
    for key in sorted(required):
        value = selected[key]
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group_name} 的 {key} 无法转换为 float。") from exc
    return weights


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    weight_group: str,
    seed: int,
) -> dict[str, float]:
    mode = weight_group.strip().lower()
    if mode not in {"naive", "learned"}:
        raise ValueError("weight-group 仅支持 {'naive', 'learned'}")

    if mode == "naive":
        selected = all_weights.get("naive")
        if selected is None or not isinstance(selected, dict):
            raise KeyError("未找到 naive 权重组。")
        return _to_weight_triplet(selected, "naive")

    selected = all_weights.get(str(seed))
    if selected is None or not isinstance(selected, dict):
        raise KeyError(f"未找到 learned 权重组（seed={seed}）。")
    return _to_weight_triplet(selected, str(seed))


def build_score_loader(
    preprocess,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = _build_dataset(dataset_name, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mean_stats_cache_path(
    dataset_name: str,
    clip_model: str,
    adapter_image_path: str,
) -> Path:
    adapter_sha1 = _hash_file(Path(adapter_image_path))
    clip_tag = clip_model.replace("/", "-").replace(" ", "_")
    return PROJECT_ROOT / "static_scores" / "group_mean_stats" / dataset_name / clip_tag / f"img_adapter_{adapter_sha1}.npz"


def _get_or_compute_group_mean_stats(
    *,
    cache_path: Path,
    image_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
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
            if (
                cached_n == n_samples
                and cached_dim == feat_dim
                and cached_cls == int(num_classes)
                and means.shape == (num_classes, feat_dim)
                and vars_.shape == (num_classes,)
            ):
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


def select_topk_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    keep_ratio: int,
) -> tuple[np.ndarray, dict[int, int]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")
    mask = np.zeros(scores.shape[0], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    ratio = keep_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        if keep_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        class_scores = scores[class_indices]
        topk_indices = class_indices[
            np.argpartition(-class_scores, num_select - 1)[:num_select]
        ]
        mask[topk_indices] = 1
        selected_by_class[class_id] = int(num_select)
    return mask, selected_by_class


def select_group_mask(
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
    weight_group: str,
    clip_model: str,
    adapter_image_path: str,
    div_static_scores: np.ndarray | None = None,
    dds_static_scores: np.ndarray | None = None,
    group_candidate_pool_size: int = 1,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")

    num_samples = sa_scores.shape[0]
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    div_static_np = np.asarray(div_static_scores, dtype=np.float32) if div_static_scores is not None else np.zeros(num_samples, dtype=np.float32)
    dds_static_np = np.asarray(dds_static_scores, dtype=np.float32) if dds_static_scores is not None else np.zeros(num_samples, dtype=np.float32)
    if labels_np.shape[0] != num_samples or div_static_np.shape[0] != num_samples or dds_static_np.shape[0] != num_samples:
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

    mean_stats_cache_path = _mean_stats_cache_path(
        dataset_name=dataset_name,
        clip_model=clip_model,
        adapter_image_path=adapter_image_path,
    )
    full_class_mean, _ = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
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
            raise RuntimeError("类别预算分配失败，无法满足目标总样本数。")
        return budgets

    class_budgets = _allocate_class_budgets()
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
    total_to_add = int(np.sum(class_budgets) - np.sum(init_per_class))
    pbar = tqdm(total=total_to_add, desc="[group] classwise greedy add", unit="sample")
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
            pbar.update(1)
            pbar.set_postfix(active_classes=int(active_classes.size))
    pbar.close()

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
        "solver": "group_classwise_greedy_add",
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

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    method = args.method.strip().lower()
    if method not in {"topk", "group"}:
        raise ValueError(f"未知 method={method}，应为 {{'topk','group'}}")

    weight_group = args.weight_group.strip().lower()
    if weight_group not in {"naive", "learned"}:
        raise ValueError("weight-group 仅支持 {'naive', 'learned'}")

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)

    data_load_start = time.perf_counter()
    dataset_for_names = _build_dataset(dataset_name, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    print(
        f"[Init] {dataset_name} data ready | samples={len(dataset_for_names)} | elapsed={time.perf_counter() - data_load_start:.2f}s"
    )

    metric_init_start = time.perf_counter()
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
    print(
        f"[Init] Metrics ready (DDS/Div/SA) | elapsed={time.perf_counter() - metric_init_start:.2f}s"
    )

    batch_size = 128
    num_workers = 4

    loader_build_start = time.perf_counter()
    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    print(
        f"[Init] DataLoaders ready (DDS/Div/SA) | elapsed={time.perf_counter() - loader_build_start:.2f}s"
    )

    method_name = f"{weight_group}_{method}"
    keep_ratios = parse_ratio_list(args.kr)
    if not keep_ratios:
        raise ValueError("kr 参数不能为空。")
    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("seed 参数不能为空。")

    total_tasks = len(seeds) * len(keep_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)
        weights = load_scoring_weights(all_weights, weight_group, seed)
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

        static_compute_start = time.perf_counter()
        static_scores = get_or_compute_static_scores(
            cache_root=PROJECT_ROOT / "static_scores",
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
        static_score_seconds = time.perf_counter() - static_compute_start
        print(
            f"[Seed {seed}] Static scores ready (cache/compute) | elapsed={static_score_seconds:.2f}s"
        )

        dds_scores_np = np.asarray(static_scores["dds"])
        div_scores_np = np.asarray(static_scores["div"])
        sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)

        if not (len(dds_scores_np) == len(div_scores_np) == len(sa_scores_np)):
            raise RuntimeError("三个指标的样本数不一致，无法合并。")

        total_scores_np = (
            weights["dds"] * dds_scores_np
            + weights["div"] * div_scores_np
            + weights["sa"] * sa_scores_np
        )
        labels = np.asarray(dataset_for_names.targets)
        for keep_ratio in keep_ratios:
            task_idx += 1
            print(
                f"[Mask {task_idx}/{total_tasks}] seed={seed} | kr={keep_ratio} | method={method} | weight_group={weight_group}"
            )
            mask_path = resolve_mask_path(
                mode=method_name,
                dataset=dataset_name,
                model=args.model_name,
                seed=seed,
                keep_ratio=keep_ratio,
            )
            if args.skip_saved and mask_path.exists():
                print(f"[Skip] mask already exists and --skip-saved is enabled: {mask_path}")
                continue

            group_stats: dict[str, object] | None = None
            if method == "topk":
                mask, selected_by_class = select_topk_mask(
                    total_scores_np,
                    labels,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                )
            else:
                mask, selected_by_class, group_stats = select_group_mask(
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
                    weight_group=weight_group,
                    clip_model=args.clip_model,
                    adapter_image_path=str(adapter_paths["image_path"]),
                    div_static_scores=div_scores_np,
                    dds_static_scores=dds_scores_np,
                    group_candidate_pool_size=args.group_candidate_pool_size,
                )

            total_time = time.perf_counter() - total_start
            mask_dir = mask_path.parent
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            print(
                f"seed={seed} | kr={keep_ratio} | selected={int(mask.sum())} "
                f"| static_score_seconds={static_score_seconds:.2f} | total_seconds={total_time:.2f}"
            )
            if group_stats is not None:
                print(
                    f"group_summary: dist_weight={group_stats['dist_weight']:.6f} | "
                    f"subset_score={group_stats['subset_comprehensive_score']:.6f} | "
                    f"distribution_shift={group_stats['distribution_shift']:.6f}"
                )
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
