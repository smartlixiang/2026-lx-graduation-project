from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import load_trained_adapters  # noqa: E402
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.group_lambda import (  # noqa: E402
    DEFAULT_EPS,
    DEFAULT_M,
    DEFAULT_R,
    compute_balance_penalty,
    get_or_estimate_lambda,
)


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
        "--seeds",
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
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="group 模式下是否额外对比 topk",
    )
    parser.add_argument("--repair-pool-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repair-pool-m-base", type=float, default=2.0)
    parser.add_argument("--repair-pool-m-slope", type=float, default=3.0)
    parser.add_argument("--repair-softmax-temp", type=float, default=1.0)
    return parser.parse_args()


def _build_dataset(dataset_name: str, transform) -> datasets.VisionDataset:
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
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


def _get_group_penalty_scales(dataset_name: str, keep_ratio: int) -> tuple[float, float]:
    """Return fixed penalty scales used by group mode."""
    dataset_key = dataset_name.strip().lower()
    if dataset_key == CIFAR10:
        scale_mean = max(0.0, 4.6 - 0.04 * float(keep_ratio))
    elif dataset_key == CIFAR100:
        scale_mean = max(0.0, 7.5 - 0.05 * float(keep_ratio))
    else:
        raise ValueError(f"Unsupported dataset for group penalty scaling: {dataset_name}")
    scale_cls = 5.0
    return float(scale_cls), float(scale_mean)


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
    dds_metric: DifficultyDirection,
    div_metric: Div,
    dds_loader: DataLoader,
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
    repair_pool_random: bool = True,
    repair_pool_m_base: float = 2.0,
    repair_pool_m_slope: float = 3.0,
    repair_softmax_temp: float = 1.0,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    # Legacy CLI knobs are retained only for backward compatibility.
    del repair_pool_random, repair_pool_m_base, repair_pool_m_slope, repair_softmax_temp
    del dds_loader
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")

    num_samples = sa_scores.shape[0]
    if labels.shape[0] != num_samples:
        raise ValueError("sa_scores 与 labels 的样本数不一致。")

    sr = float(keep_ratio) / 100.0
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    if div_static_scores is not None:
        div_static_np = np.asarray(div_static_scores, dtype=np.float32)
    else:
        div_static_np = np.zeros(num_samples, dtype=np.float32)
    if dds_static_scores is not None:
        dds_static_np = np.asarray(dds_static_scores, dtype=np.float32)
    else:
        dds_static_np = np.zeros(num_samples, dtype=np.float32)

    if div_static_np.shape[0] != num_samples or dds_static_np.shape[0] != num_samples:
        raise ValueError("静态分数与样本数不一致。")

    target_size = int(round(sr * num_samples))
    target_size = min(num_samples, max(1, target_size)) if num_samples > 0 else 0
    all_indices = np.arange(num_samples, dtype=np.int64)
    class_indices_list = [np.flatnonzero(labels_np == c).astype(np.int64) for c in range(num_classes)]

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

    lambda_sample_M = DEFAULT_M
    lambda_ratio_r = DEFAULT_R
    lambda_eps = DEFAULT_EPS

    cached_real_stats: dict[bytes, tuple[float, np.ndarray, np.ndarray]] = {}
    membership_buffer = np.zeros(num_samples, dtype=bool)
    sum_features = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float64)
    cnt_features = np.zeros(num_classes, dtype=np.int64)

    static_part = (weights["sa"] * sa_scores_np + weights["dds"] * dds_static_np).astype(np.float32)
    guide_score_np = static_part

    def _indices_to_mask(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(num_samples, dtype=np.uint8)
        if indices.size > 0:
            mask[np.asarray(indices, dtype=np.int64)] = 1
        return mask

    def _build_membership(indices: np.ndarray) -> np.ndarray:
        membership_buffer.fill(False)
        if indices.size > 0:
            membership_buffer[np.asarray(indices, dtype=np.int64)] = True
        return membership_buffer

    def _complement_indices(indices: np.ndarray) -> np.ndarray:
        return all_indices[~_build_membership(indices)]

    def _mask_cache_key(cur_mask: np.ndarray) -> bytes:
        return np.asarray(cur_mask, dtype=np.uint8).tobytes()

    def _compute_mean_penalty(cur_mask: np.ndarray) -> float:
        selected = np.flatnonzero(cur_mask > 0).astype(np.int64)
        if selected.size == 0:
            return 0.0
        sum_features.fill(0.0)
        cnt_features.fill(0)
        sel_labels = labels_np[selected]
        np.add.at(sum_features, sel_labels, div_features_np[selected].astype(np.float64))
        np.add.at(cnt_features, sel_labels, 1)
        active = cnt_features > 0
        if not np.any(active):
            return 0.0
        selected_mean = np.zeros_like(full_class_mean, dtype=np.float64)
        selected_mean[active] = sum_features[active] / cnt_features[active, None]
        diff = selected_mean[active] - full_class_mean[active].astype(np.float64)
        return float(np.sum(np.linalg.norm(diff, axis=1)))

    def _real_stats_cached(cur_mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        key = _mask_cache_key(cur_mask)
        cached = cached_real_stats.get(key)
        if cached is not None:
            return cached
        selected = cur_mask.astype(bool)
        div_scores = np.asarray(
            div_metric.score_dataset_dynamic(
                div_loader,
                adapter=image_adapter,
                selected_mask=cur_mask,
                image_features=div_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        s_ref = static_part + weights["div"] * div_scores
        counts = np.bincount(labels_np[selected], minlength=num_classes).astype(np.int64)
        s_val = float(np.sum(s_ref[selected]))
        cached_real_stats[key] = (s_val, s_ref, counts)
        return cached_real_stats[key]

    lambda_record = get_or_estimate_lambda(
        cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
        dataset=dataset_name,
        seed=seed,
        kr=keep_ratio,
        weight_group=weight_group,
        n_samples=num_samples,
        target_size=target_size,
        eval_score_fn=lambda cur_mask: _real_stats_cached(cur_mask)[0],
        penalty_fn=lambda cur_mask: compute_balance_penalty(cur_mask, labels_np, num_classes, target_size),
        mean_penalty_fn=lambda cur_mask: _compute_mean_penalty(cur_mask),
        M=lambda_sample_M,
        r=lambda_ratio_r,
        eps=lambda_eps,
        tqdm_desc=f"Estimating lambdas (seed={seed}, kr={keep_ratio}, wg={weight_group})",
    )

    required_lambda_keys = {"lambda_std_cls", "lambda_std_mean", "lambda_cls", "lambda_mean"}
    if not required_lambda_keys.issubset(lambda_record.keys()):
        raise RuntimeError(
            "group_lambda cache missing required fields: lambda_std_cls/lambda_std_mean; "
            "please regenerate utils/group_lambda.json with current format."
        )

    scale_cls, scale_mean = _get_group_penalty_scales(dataset_name, keep_ratio)
    lambda_cls = float(lambda_record["lambda_cls"]) * float(scale_cls)
    lambda_mean = float(lambda_record["lambda_mean"]) * float(scale_mean)
    lambda_std_cls = float(lambda_record["lambda_std_cls"])
    lambda_std_mean = float(lambda_record["lambda_std_mean"])

    def _evaluate(indices: np.ndarray) -> dict[str, object]:
        unique = np.unique(np.asarray(indices, dtype=np.int64))
        if unique.size == 0:
            raise ValueError("_evaluate expects non-empty indices")
        if unique.size != target_size:
            raise ValueError(f"_evaluate expects fixed-cardinality subset, got {unique.size} != {target_size}")
        mask = _indices_to_mask(unique)
        raw_score, _, counts = _real_stats_cached(mask)
        class_penalty = compute_balance_penalty(mask, labels_np, num_classes, target_size)
        mean_penalty = _compute_mean_penalty(mask)
        class_corr = float(lambda_cls * class_penalty)
        mean_corr = float(lambda_mean * mean_penalty)
        fitness = float(raw_score - class_corr - mean_corr)
        return {
            "indices": unique,
            "mask": mask,
            "raw_score": float(raw_score),
            "penalty": float(class_penalty),
            "mean_penalty": float(mean_penalty),
            "class_corr": class_corr,
            "mean_corr": mean_corr,
            "fitness": fitness,
            "counts": counts,
        }

    full_class_counts = np.bincount(labels_np, minlength=num_classes).astype(np.int64)
    class_priors = full_class_counts.astype(np.float64) / max(1, int(num_samples))
    target_class_counts = np.rint(class_priors * float(target_size)).astype(np.int64)

    def _build_subset_class_stats(indices: np.ndarray) -> dict[str, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        class_counts = np.bincount(labels_np[indices], minlength=num_classes).astype(np.int64)
        class_feature_sums = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float64)
        if indices.size > 0:
            np.add.at(class_feature_sums, labels_np[indices], div_features_np[indices].astype(np.float64))
        class_means = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float64)
        active = class_counts > 0
        if np.any(active):
            class_means[active] = class_feature_sums[active] / class_counts[active, None]
        return {
            "class_counts": class_counts,
            "class_feature_sums": class_feature_sums,
            "class_means": class_means,
        }

    def _rank_normalize(scores_np: np.ndarray, descending: bool = True) -> np.ndarray:
        scores = np.asarray(scores_np, dtype=np.float64)
        n = int(scores.size)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if n == 1:
            return np.ones(1, dtype=np.float32)
        if descending:
            order = np.argsort(-scores, kind="mergesort")
        else:
            order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, 0, -1, dtype=np.float64)
        return ((ranks - 1.0) / float(n - 1)).astype(np.float32)

    def _score_add_candidates_by_ranks(
        current_indices: np.ndarray,
        candidate_indices: np.ndarray,
        *,
        weak_class_bias: bool = False,
    ) -> dict[str, np.ndarray]:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_indices.size == 0:
            return {
                "candidate_indices": candidate_indices,
                "total_add_rank": np.zeros(0, dtype=np.float32),
                "rank_static": np.zeros(0, dtype=np.float32),
                "rank_spread": np.zeros(0, dtype=np.float32),
                "rank_herd": np.zeros(0, dtype=np.float32),
            }
        stats = _build_subset_class_stats(current_indices)
        class_counts = stats["class_counts"]
        class_sums = stats["class_feature_sums"]
        class_means = stats["class_means"]

        static_raw = guide_score_np[candidate_indices].astype(np.float64)
        spread_raw = np.zeros(candidate_indices.size, dtype=np.float64)
        herd_raw = np.zeros(candidate_indices.size, dtype=np.float64)
        for i, cand_idx in enumerate(candidate_indices.tolist()):
            c = int(labels_np[cand_idx])
            feat = div_features_np[cand_idx].astype(np.float64)
            base_mean = class_means[c] if class_counts[c] > 0 else full_class_mean[c].astype(np.float64)
            spread_raw[i] = float(np.linalg.norm(feat - base_mean))
            old_dist = float(np.linalg.norm(base_mean - full_class_mean[c].astype(np.float64)))
            new_mean = (class_sums[c] + feat) / float(class_counts[c] + 1)
            new_dist = float(np.linalg.norm(new_mean - full_class_mean[c].astype(np.float64)))
            herd_raw[i] = old_dist - new_dist

        rank_static = _rank_normalize(static_raw, descending=True)
        rank_spread = _rank_normalize(spread_raw, descending=True)
        rank_herd = _rank_normalize(herd_raw, descending=True)
        total_add_rank = rank_static + rank_spread + rank_herd
        if weak_class_bias:
            deficits = target_class_counts - class_counts
            weak_bonus = np.clip(deficits[labels_np[candidate_indices]], 0, None).astype(np.float32)
            if weak_bonus.size > 0 and float(np.max(weak_bonus)) > 0:
                total_add_rank = total_add_rank + 0.25 * (weak_bonus / (float(np.max(weak_bonus)) + 1e-8))
        return {
            "candidate_indices": candidate_indices,
            "total_add_rank": total_add_rank.astype(np.float32),
            "rank_static": rank_static,
            "rank_spread": rank_spread,
            "rank_herd": rank_herd,
        }

    def _score_drop_candidates_by_ranks(
        current_indices: np.ndarray,
        candidate_indices: np.ndarray,
        *,
        weak_class_bias: bool = False,
    ) -> dict[str, np.ndarray]:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        if candidate_indices.size == 0:
            return {
                "candidate_indices": candidate_indices,
                "total_drop_rank": np.zeros(0, dtype=np.float32),
                "rank_static_drop": np.zeros(0, dtype=np.float32),
                "rank_spread_drop": np.zeros(0, dtype=np.float32),
                "rank_herd_drop": np.zeros(0, dtype=np.float32),
            }
        stats = _build_subset_class_stats(current_indices)
        class_counts = stats["class_counts"]
        class_sums = stats["class_feature_sums"]
        class_means = stats["class_means"]

        static_drop_raw = -guide_score_np[candidate_indices].astype(np.float64)
        spread_drop_raw = np.zeros(candidate_indices.size, dtype=np.float64)
        herd_drop_raw = np.zeros(candidate_indices.size, dtype=np.float64)
        for i, cand_idx in enumerate(candidate_indices.tolist()):
            c = int(labels_np[cand_idx])
            feat = div_features_np[cand_idx].astype(np.float64)
            base_mean = class_means[c] if class_counts[c] > 0 else full_class_mean[c].astype(np.float64)
            spread_drop_raw[i] = -float(np.linalg.norm(feat - base_mean))
            old_dist = float(np.linalg.norm(base_mean - full_class_mean[c].astype(np.float64)))
            if class_counts[c] <= 1:
                new_dist = 0.0
            else:
                new_mean = (class_sums[c] - feat) / float(class_counts[c] - 1)
                new_dist = float(np.linalg.norm(new_mean - full_class_mean[c].astype(np.float64)))
            herd_drop_raw[i] = old_dist - new_dist

        rank_static_drop = _rank_normalize(static_drop_raw, descending=True)
        rank_spread_drop = _rank_normalize(spread_drop_raw, descending=True)
        rank_herd_drop = _rank_normalize(herd_drop_raw, descending=True)
        total_drop_rank = rank_static_drop + rank_spread_drop + rank_herd_drop
        if weak_class_bias:
            surplus = _build_subset_class_stats(current_indices)["class_counts"] - target_class_counts
            weak_bonus = np.clip(surplus[labels_np[candidate_indices]], 0, None).astype(np.float32)
            if weak_bonus.size > 0 and float(np.max(weak_bonus)) > 0:
                total_drop_rank = total_drop_rank + 0.25 * (weak_bonus / (float(np.max(weak_bonus)) + 1e-8))
        return {
            "candidate_indices": candidate_indices,
            "total_drop_rank": total_drop_rank.astype(np.float32),
            "rank_static_drop": rank_static_drop,
            "rank_spread_drop": rank_spread_drop,
            "rank_herd_drop": rank_herd_drop,
        }

    def _repair_size(indices: np.ndarray) -> np.ndarray:
        current = np.unique(np.asarray(indices, dtype=np.int64))
        if current.size < target_size:
            need = target_size - current.size
            while need > 0:
                available = _complement_indices(current)
                if available.size == 0:
                    break
                add_scores = _score_add_candidates_by_ranks(current, available, weak_class_bias=True)
                order = np.argsort(-add_scores["total_add_rank"], kind="mergesort")
                top_k = min(max(8, need * 3), order.size)
                chosen_pos = int(np.random.choice(order[:top_k]))
                chosen = int(add_scores["candidate_indices"][chosen_pos])
                current = np.unique(np.concatenate([current, np.asarray([chosen], dtype=np.int64)]))
                need = target_size - current.size

        if current.size > target_size:
            over = current.size - target_size
            while over > 0 and current.size > 0:
                drop_scores = _score_drop_candidates_by_ranks(current, current, weak_class_bias=True)
                order = np.argsort(-drop_scores["total_drop_rank"], kind="mergesort")
                top_k = min(max(8, over * 3), order.size)
                chosen_pos = int(np.random.choice(order[:top_k]))
                drop = int(drop_scores["candidate_indices"][chosen_pos])
                current = current[current != drop]
                over = current.size - target_size

        if current.size < target_size:
            available = _complement_indices(current)
            if available.size > 0:
                rng_fill = np.random.default_rng(seed + int(current.size) + int(target_size))
                add = rng_fill.choice(available, size=min(target_size - current.size, available.size), replace=False)
                current = np.unique(np.concatenate([current, add.astype(np.int64)]))

        return np.sort(current[:target_size].astype(np.int64))

    def _random_init_population(pop_size: int, target_size_local: int, num_samples_local: int, base_seed: int) -> list[np.ndarray]:
        population: list[np.ndarray] = []
        for idx in range(pop_size):
            rng = np.random.default_rng(base_seed * (idx + 1))
            indiv = np.sort(rng.choice(num_samples_local, size=target_size_local, replace=False).astype(np.int64))
            population.append(indiv)
        return population

    def _tournament_select(pop_items: list[dict[str, object]], rng: np.random.Generator) -> np.ndarray:
        i1 = int(rng.integers(0, len(pop_items)))
        i2 = int(rng.integers(0, len(pop_items)))
        a = pop_items[i1]
        b = pop_items[i2]
        return np.asarray(a["indices"] if float(a["fitness"]) >= float(b["fitness"]) else b["indices"], dtype=np.int64)

    def _crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        common = np.intersect1d(parent_a, parent_b, assume_unique=False)
        child = np.unique(common)
        needed = max(0, target_size - child.size)
        if needed > 0:
            union_only = np.setdiff1d(np.union1d(parent_a, parent_b), child, assume_unique=False)
            if union_only.size > 0:
                add_scores = _score_add_candidates_by_ranks(child, union_only, weak_class_bias=True)
                order = np.argsort(-add_scores["total_add_rank"], kind="mergesort")
                take = add_scores["candidate_indices"][order[: min(needed, union_only.size)]]
                child = np.unique(np.concatenate([child, take]))
        return _repair_size(child)

    def _mutate_swap(indices: np.ndarray, replace_count: int, rng: np.random.Generator) -> np.ndarray:
        current = np.asarray(indices, dtype=np.int64)
        if current.size == 0:
            return current
        for _ in range(max(1, replace_count)):
            available = _complement_indices(current)
            if available.size == 0 or current.size == 0:
                break
            drop_scores = _score_drop_candidates_by_ranks(current, current, weak_class_bias=True)
            add_scores = _score_add_candidates_by_ranks(current, available, weak_class_bias=True)
            drop_order = np.argsort(-drop_scores["total_drop_rank"], kind="mergesort")
            add_order = np.argsort(-add_scores["total_add_rank"], kind="mergesort")
            drop_pool_k = min(max(12, replace_count * 3), drop_order.size)
            add_pool_k = min(max(24, replace_count * 4), add_order.size)
            drop = int(drop_scores["candidate_indices"][int(rng.choice(drop_order[:drop_pool_k]))])
            add = int(add_scores["candidate_indices"][int(rng.choice(add_order[:add_pool_k]))])
            current = current[current != drop]
            current = np.unique(np.concatenate([current, np.asarray([add], dtype=np.int64)]))
        return _repair_size(current)

    def _local_search_swap(indices: np.ndarray, num_steps: int, rng: np.random.Generator) -> np.ndarray:
        current = np.asarray(indices, dtype=np.int64)
        if num_steps <= 0 or current.size == 0:
            return current
        for _ in range(num_steps):
            available = _complement_indices(current)
            if available.size == 0 or current.size == 0:
                break
            drop_scores = _score_drop_candidates_by_ranks(current, current, weak_class_bias=True)
            add_scores = _score_add_candidates_by_ranks(current, available, weak_class_bias=True)
            drop_order = np.argsort(-drop_scores["total_drop_rank"], kind="mergesort")
            add_order = np.argsort(-add_scores["total_add_rank"], kind="mergesort")
            drop_pool_k = min(max(6, 2 * num_steps), drop_order.size)
            add_pool_k = min(max(12, 3 * num_steps), add_order.size)
            drop = int(drop_scores["candidate_indices"][int(rng.choice(drop_order[:drop_pool_k]))])
            add = int(add_scores["candidate_indices"][int(rng.choice(add_order[:add_pool_k]))])
            current = current[current != drop]
            current = np.unique(np.concatenate([current, np.asarray([add], dtype=np.int64)]))
        return _repair_size(current)

    def _jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.intersect1d(a, b).size
        union = np.union1d(a, b).size
        return float(inter / max(1, union))

    def _population_diversity(items: list[dict[str, object]]) -> float:
        if len(items) <= 1:
            return 0.0
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sims.append(_jaccard_similarity(np.asarray(items[i]["indices"]), np.asarray(items[j]["indices"])))
        return float(1.0 - np.mean(np.asarray(sims, dtype=np.float64))) if sims else 0.0

    def _survivor_selection(candidates: list[dict[str, object]], pop_size: int, sim_thr: float) -> list[dict[str, object]]:
        sorted_items = sorted(candidates, key=lambda x: float(x["fitness"]), reverse=True)
        survivors: list[dict[str, object]] = []
        for item in sorted_items:
            if len(survivors) >= pop_size:
                break
            idx = np.asarray(item["indices"], dtype=np.int64)
            if any(_jaccard_similarity(idx, np.asarray(s["indices"], dtype=np.int64)) > sim_thr for s in survivors):
                continue
            survivors.append(item)
        if len(survivors) < pop_size:
            for item in sorted_items:
                if len(survivors) >= pop_size:
                    break
                if all(id(item) != id(s) for s in survivors):
                    survivors.append(item)
        return survivors

    print(
        f"[GroupLambda] dataset={dataset_name} kr={keep_ratio} "
        f"scale_cls={scale_cls:.6f} scale_mean={scale_mean:.6f} "
        f"lambda_std_cls={lambda_std_cls:.8f} lambda_std_mean={lambda_std_mean:.8f} "
        f"lambda_cls={lambda_cls:.8f} lambda_mean={lambda_mean:.8f}"
    )

    ga_population_size = int(np.clip(16 + keep_ratio // 5, 16, 36))
    ga_generations = int(np.clip(8 + keep_ratio // 10, 8, 18))
    mutation_ratio_eff = float(np.clip(0.18 - 0.0010 * keep_ratio, 0.03, 0.10))
    mutation_strength = int(np.clip(max(2, round(target_size * mutation_ratio_eff)), 2, max(2, target_size // 2)))
    local_search_steps = int(np.clip(5 - keep_ratio // 25, 2, 5))
    crossover_children = max(2, int(ga_population_size * 0.75))
    elite_count = max(1, ga_population_size // 8)
    similarity_thr = 0.85 if keep_ratio <= 50 else 0.90

    population = _random_init_population(ga_population_size, target_size, num_samples, seed)
    population_items = [_evaluate(ind) for ind in population]
    best_item = max(population_items, key=lambda x: float(x["fitness"]))

    score_history = [float(best_item["raw_score"])]
    class_correction_history = [float(best_item["class_corr"])]
    mean_correction_history = [float(best_item["mean_corr"])]
    fitness_history = [float(best_item["fitness"])]
    unique_population_count_per_gen = [len({tuple(np.asarray(x["indices"], dtype=np.int64).tolist()) for x in population_items})]
    best_overlap_with_prev_gen = [1.0]
    best_gen = 0

    rng = np.random.default_rng(seed * 9973 + keep_ratio)
    prev_best_indices = np.asarray(best_item["indices"], dtype=np.int64)

    gen_bar = tqdm(
        range(ga_generations),
        desc=f"[group-ea] seed={seed} kr={keep_ratio}",
        unit="gen",
        leave=False,
    )
    for gen in gen_bar:
        sorted_pop = sorted(population_items, key=lambda x: float(x["fitness"]), reverse=True)
        elites = sorted_pop[:elite_count]
        children: list[dict[str, object]] = []

        while len(children) < crossover_children:
            p1 = _tournament_select(population_items, rng)
            p2 = _tournament_select(population_items, rng)
            child = _crossover(p1, p2)
            if rng.random() < 0.95:
                child = _mutate_swap(child, mutation_strength, rng)
            child = _local_search_swap(child, local_search_steps, rng)
            children.append(_evaluate(child))

        merged = population_items + children
        population_items = _survivor_selection(merged, ga_population_size, similarity_thr)
        if len(elites) > 0:
            for elite in elites:
                if all(id(elite) != id(p) for p in population_items):
                    population_items[-1] = elite

        cur_best = max(population_items, key=lambda x: float(x["fitness"]))
        improved = float(cur_best["fitness"]) > float(best_item["fitness"])
        if improved:
            best_item = cur_best
            best_gen = gen + 1

        cur_best_indices = np.asarray(cur_best["indices"], dtype=np.int64)
        overlap = _jaccard_similarity(prev_best_indices, cur_best_indices)
        prev_best_indices = cur_best_indices

        score_history.append(float(cur_best["raw_score"]))
        class_correction_history.append(float(cur_best["class_corr"]))
        mean_correction_history.append(float(cur_best["mean_corr"]))
        fitness_history.append(float(cur_best["fitness"]))
        unique_population_count_per_gen.append(len({tuple(np.asarray(x["indices"], dtype=np.int64).tolist()) for x in population_items}))
        best_overlap_with_prev_gen.append(float(overlap))

        pop_diversity = _population_diversity(population_items)
        gen_bar.set_postfix(
            best_fitness=f"{float(cur_best['fitness']):.4f}",
            raw=f"{float(cur_best['raw_score']):.4f}",
            cls=f"{float(cur_best['class_corr']):.4f}",
            mean=f"{float(cur_best['mean_corr']):.4f}",
            diversity=f"{pop_diversity:.4f}",
            improved=improved,
        )
        print(
            f"[group-ea] gen={gen + 1}/{ga_generations} best_fitness={float(cur_best['fitness']):.6f} "
            f"best_raw_score={float(cur_best['raw_score']):.6f} best_class_corr={float(cur_best['class_corr']):.6f} "
            f"best_mean_corr={float(cur_best['mean_corr']):.6f} population_diversity={pop_diversity:.6f} "
            f"improved_this_gen={improved}"
        )

    final_mask = np.asarray(best_item["mask"], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    final_rate = float(final_mask.mean())
    stats: dict[str, object] = {
        "solver": "fixed_cardinality_memetic_ea",
        "sr": float(sr),
        "final_rate": final_rate,
        "init_best_fitness": float(fitness_history[0]),
        "final_best_fitness": float(best_item["fitness"]),
        "best_gen": int(best_gen),
        "unique_population_count_per_gen": unique_population_count_per_gen,
        "best_overlap_with_prev_gen": best_overlap_with_prev_gen,
        "ga_population_size": int(ga_population_size),
        "ga_generations": int(ga_generations),
        "elite_count": int(elite_count),
        "init_fitness": float(fitness_history[0]),
        "final_fitness": float(best_item["fitness"]),
        "raw_score": float(best_item["raw_score"]),
        "penalty": float(best_item["penalty"]),
        "final_mean_penalty": float(best_item["mean_penalty"]),
        "S": float(best_item["raw_score"]),
        "S_prime": float(best_item["fitness"]),
        "selected_by_class": selected_by_class,
        "best_iter": int(np.argmax(np.asarray(fitness_history, dtype=np.float32))) if fitness_history else 0,
        "score_history": score_history,
        "fitness_history": fitness_history,
        "correction_history": [float(c) + float(m) for c, m in zip(class_correction_history, mean_correction_history)],
        "class_correction_history": class_correction_history,
        "mean_correction_history": mean_correction_history,
        "scale_cls": float(scale_cls),
        "scale_mean": float(scale_mean),
        "lambda_std_cls": float(lambda_std_cls),
        "lambda_std_mean": float(lambda_std_mean),
        "lambda_cls": float(lambda_cls),
        "lambda_mean": float(lambda_mean),
    }

    return final_mask, selected_by_class, stats


def _sanitize_for_filename(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def save_score_curve_plot(
    score_history: Sequence[float],
    class_correction_history: Sequence[float],
    mean_correction_history: Sequence[float],
    fitness_history: Sequence[float] | None = None,
    *,
    dataset: str,
    method: str,
    weight_group: str,
    model_name: str,
    seed: int,
    keep_ratio: int,
    clip_model: str,
) -> Path:
    out_dir = PROJECT_ROOT / "mask_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_sanitize_for_filename(dataset)}_{keep_ratio}_{seed}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    if fitness_history is not None and len(fitness_history) > 0:
        fit_arr = np.asarray(fitness_history, dtype=np.float64)
    else:
        score_arr = np.asarray(score_history, dtype=np.float64)
        class_corr_arr = np.asarray(class_correction_history, dtype=np.float64)
        mean_corr_arr = np.asarray(mean_correction_history, dtype=np.float64)
        fit_arr = score_arr - class_corr_arr - mean_corr_arr

    x = np.arange(fit_arr.shape[0], dtype=np.int32)
    ax.plot(x, fit_arr, linewidth=1.8, color="#d62728", label="S'(D)")
    ax.set_title("Group optimization trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Comprehensive score S'(D)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        raise ValueError("seeds 参数不能为空。")

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
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
        div_features_for_compare = None

        def compute_subset_dynamic_sum(selected_mask: np.ndarray) -> float:
            nonlocal div_features_for_compare
            if div_features_for_compare is None:
                div_features_for_compare, _ = div_metric._encode_images(div_loader, image_adapter)
            div_scores_dyn = np.asarray(
                div_metric.score_dataset_dynamic(
                    div_loader,
                    adapter=image_adapter,
                    selected_mask=selected_mask,
                    image_features=div_features_for_compare,
                    labels=labels_t,
                ).scores,
                dtype=np.float32,
            )
            subset_scores = (
                weights["sa"] * sa_scores_np
                + weights["div"] * div_scores_dyn
                + weights["dds"] * dds_scores_np
            )
            return float(subset_scores[selected_mask.astype(bool)].sum())

        for keep_ratio in keep_ratios:
            task_idx += 1
            print(
                f"[Mask {task_idx}/{total_tasks}] seed={seed} | kr={keep_ratio} | method={method} | weight_group={weight_group}"
            )
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
                    dds_metric=dds_metric,
                    div_metric=div_metric,
                    dds_loader=dds_loader,
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
                    repair_pool_random=args.repair_pool_random,
                    repair_pool_m_base=args.repair_pool_m_base,
                    repair_pool_m_slope=args.repair_pool_m_slope,
                    repair_softmax_temp=args.repair_softmax_temp,
                )
                debug_curve = save_score_curve_plot(
                    group_stats.get("score_history", []),
                    group_stats.get("class_correction_history", []),
                    group_stats.get("mean_correction_history", []),
                    fitness_history=group_stats.get("fitness_history", []),
                    dataset=dataset_name,
                    method=method,
                    weight_group=weight_group,
                    model_name=args.model_name,
                    seed=seed,
                    keep_ratio=keep_ratio,
                    clip_model=args.clip_model,
                )
                print(f"[Debug] score curve saved to: {debug_curve}")

                if args.compare:
                    topk_mask, _ = select_topk_mask(
                        total_scores_np,
                        labels,
                        num_classes=len(class_names),
                        keep_ratio=keep_ratio,
                    )
                    inter = int(np.logical_and(mask == 1, topk_mask == 1).sum())
                    sel = int(mask.sum())
                    overlap = inter / max(1, sel)
                    sum_group = compute_subset_dynamic_sum(mask)
                    sum_topk = compute_subset_dynamic_sum(topk_mask)
                    better = "group" if sum_group >= sum_topk else "topk"
                    diff = abs(sum_group - sum_topk)
                    print(
                        "[Compare] "
                        f"overlap={overlap:.4f} ({inter}/{sel}) | "
                        f"sum_group={sum_group:.4f} | sum_topk={sum_topk:.4f} | "
                        f"better={better} (Δ={diff:.4f})"
                    )

            total_time = time.perf_counter() - total_start
            mask_path = resolve_mask_path(
                mode=method_name,
                dataset=dataset_name,
                model=args.model_name,
                seed=seed,
                keep_ratio=keep_ratio,
            )
            mask_dir = mask_path.parent
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            print(
                f"seed={seed} | kr={keep_ratio} | selected={int(mask.sum())} "
                f"| static_score_seconds={static_score_seconds:.2f} | total_seconds={total_time:.2f}"
            )
            if group_stats is not None:
                print(
                    "group_stats: "
                    f"sr={group_stats['sr']:.6f} | rate={group_stats['final_rate']:.6f} | "
                    f"m_c={group_stats['selected_by_class']} | "
                    f"best_iter={group_stats['best_iter']} | "
                    f"S(D)={group_stats['S']:.6f} | pen(D)={group_stats['penalty']:.6f} | "
                    f"mean_pen(D)={group_stats['final_mean_penalty']:.6f} | "
                    f"lambda_mean={group_stats['lambda_mean']:.6f} | S'(D)={group_stats['S_prime']:.6f}"
                )
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
