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
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    if div_static_scores is not None:
        div_static_np = np.asarray(div_static_scores, dtype=np.float32)
        if div_static_np.shape[0] != num_samples:
            raise ValueError("div_static_scores 与 sa_scores 的样本数不一致。")
    else:
        div_static_np = np.zeros(num_samples, dtype=np.float32)

    if dds_static_scores is not None:
        dds_static_np = np.asarray(dds_static_scores, dtype=np.float32)
        if dds_static_np.shape[0] != num_samples:
            raise ValueError("dds_static_scores 与 sa_scores 的样本数不一致。")
    else:
        dds_static_np = np.zeros(num_samples, dtype=np.float32)

    labels_np = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(num_samples, dtype=np.int64)
    class_indices_list = [np.flatnonzero(labels_np == class_id).astype(np.int64) for class_id in range(num_classes)]
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
    full_class_mean, full_class_var = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    class_features_list = [div_features_np[class_indices] for class_indices in class_indices_list]

    target_size = int(round(sr * num_samples))
    if num_samples > 0:
        target_size = min(num_samples, max(1, target_size))
    else:
        target_size = 0

    lambda_sample_M = DEFAULT_M
    lambda_ratio_r = DEFAULT_R
    lambda_eps = DEFAULT_EPS

    cached_real_stats: dict[bytes, tuple[float, np.ndarray, np.ndarray]] = {}
    membership_buffer = np.zeros(num_samples, dtype=bool)

    def _mask_cache_key(cur_mask: np.ndarray) -> bytes:
        return np.asarray(cur_mask, dtype=np.uint8).tobytes()

    def _indices_to_mask(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(num_samples, dtype=np.uint8)
        if indices.size > 0:
            mask[np.asarray(indices, dtype=np.int64)] = 1
        return mask

    def _pick_top_by_score(candidate_idx: np.ndarray, k: int, score_np: np.ndarray) -> np.ndarray:
        if k <= 0 or candidate_idx.size == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(-score_np[candidate_idx], kind="mergesort")
        return candidate_idx[order[: min(k, candidate_idx.size)]].astype(np.int64)

    def _build_membership(indices: np.ndarray) -> np.ndarray:
        membership_buffer.fill(False)
        if indices.size > 0:
            membership_buffer[np.asarray(indices, dtype=np.int64)] = True
        return membership_buffer

    def _complement_indices(indices: np.ndarray) -> np.ndarray:
        member = _build_membership(indices)
        return all_indices[~member]

    def _real_stats_cached(cur_mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        cur_key = _mask_cache_key(cur_mask)
        cached = cached_real_stats.get(cur_key)
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
        s_ref = weights["sa"] * sa_scores_np + weights["div"] * div_scores + weights["dds"] * dds_static_np
        counts = np.bincount(labels_np[selected], minlength=num_classes).astype(np.int64)
        s_val = float(np.sum(s_ref[selected]))
        cached_real_stats[cur_key] = (s_val, s_ref, counts)
        return cached_real_stats[cur_key]

    guide_score_np = (sa_scores_np + dds_static_np).astype(np.float32)

    def _compute_mean_penalty(cur_mask: np.ndarray) -> float:
        selected = cur_mask.astype(bool)
        penalty_sum = 0.0
        for class_id in range(num_classes):
            class_indices = class_indices_list[class_id]
            if class_indices.size == 0:
                continue
            class_selected = selected[class_indices]
            count = int(np.sum(class_selected))
            if count <= 0:
                continue
            subset_sum = np.sum(class_features_list[class_id][class_selected], axis=0, dtype=np.float32)
            subset_mean = subset_sum / float(count)
            diff = subset_mean - full_class_mean[class_id]
            penalty_sum += float(np.dot(diff, diff)) / (float(full_class_var[class_id]) + lambda_eps)
        return float(penalty_sum)

    lambda_info = get_or_estimate_lambda(
        cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
        dataset=dataset_name,
        seed=seed,
        kr=keep_ratio,
        weight_group=weight_group,
        n_samples=num_samples,
        target_size=target_size,
        eval_score_fn=lambda mask: _real_stats_cached(mask)[0],
        penalty_fn=lambda mask: compute_balance_penalty(mask, labels_np, num_classes, target_size),
        mean_penalty_fn=_compute_mean_penalty,
        M=max(10, min(lambda_sample_M, num_samples)),
        r=lambda_ratio_r,
        eps=lambda_eps,
        tqdm_desc=f"Estimating lambdas (seed={seed}, kr={keep_ratio}, wg={weight_group})",
    )
    if "lambda_std_cls" not in lambda_info or "lambda_std_mean" not in lambda_info:
        raise KeyError(
            "group_lambda cache missing required fields: lambda_std_cls/lambda_std_mean; "
            "please regenerate utils/group_lambda.json with current format."
        )
    scale_cls, scale_mean = _get_group_penalty_scales(dataset_name, keep_ratio)
    lambda_std_cls = float(lambda_info["lambda_std_cls"])
    lambda_std_mean = float(lambda_info["lambda_std_mean"])
    lambda_cls = float(scale_cls * lambda_std_cls)
    lambda_mean = float(scale_mean * lambda_std_mean)

    def _evaluate_indices(indices: np.ndarray) -> dict[str, object]:
        unique_indices = np.unique(np.asarray(indices, dtype=np.int64))
        if unique_indices.size == 0:
            raise ValueError("_evaluate_indices expects non-empty indices")
        mask = _indices_to_mask(unique_indices)
        raw_score, s_ref, counts = _real_stats_cached(mask)
        class_penalty = compute_balance_penalty(mask, labels_np, num_classes, target_size)
        mean_penalty = _compute_mean_penalty(mask)
        fitness = raw_score - lambda_cls * class_penalty - lambda_mean * mean_penalty
        return {
            "indices": unique_indices,
            "mask": mask,
            "fitness": float(fitness),
            "raw_score": float(raw_score),
            "penalty": float(class_penalty),
            "mean_penalty": float(mean_penalty),
            "s_ref": s_ref,
            "counts": counts,
        }

    def _build_candidate_pool(
        guide_score: np.ndarray,
        div_static: np.ndarray,
        labels_arr: np.ndarray,
        target_k: int,
        n_classes: int,
    ) -> np.ndarray:
        pool_by_static = min(num_samples, max(4 * target_k, 512))
        pool_by_div = min(num_samples, max(2 * target_k, 256))
        static_head = _pick_top_by_score(all_indices, pool_by_static, guide_score)
        div_head = _pick_top_by_score(all_indices, pool_by_div, div_static)

        per_class_take = max(1, int(np.ceil(target_k / max(1, n_classes))))
        class_cover = []
        for class_id in range(n_classes):
            cls_idx = np.flatnonzero(labels_arr == class_id).astype(np.int64)
            if cls_idx.size == 0:
                continue
            class_cover.append(_pick_top_by_score(cls_idx, per_class_take, guide_score))
        class_cover_arr = np.concatenate(class_cover).astype(np.int64) if class_cover else np.empty(0, dtype=np.int64)

        candidate_pool = np.unique(np.concatenate([static_head, div_head, class_cover_arr]).astype(np.int64))
        min_pool = min(num_samples, max(4 * target_k, 512))
        max_pool = min(num_samples, max(6 * target_k, 1024))
        if candidate_pool.size < min_pool:
            fill = _pick_top_by_score(all_indices, min_pool, guide_score)
            candidate_pool = np.unique(np.concatenate([candidate_pool, fill]).astype(np.int64))
        if candidate_pool.size > max_pool:
            candidate_pool = _pick_top_by_score(candidate_pool, max_pool, guide_score)
        return np.sort(candidate_pool.astype(np.int64))

    def _greedy_construct(candidate_pool: np.ndarray, cheap_score: np.ndarray, greedy_eval_k: int) -> tuple[dict[str, object], list[float], list[float], list[float], list[float]]:
        selected = np.empty(0, dtype=np.int64)
        score_hist: list[float] = []
        class_corr_hist: list[float] = []
        mean_corr_hist: list[float] = []
        fitness_hist: list[float] = []
        greedy_bar = tqdm(
            range(target_size),
            desc=f"[group-greedy] seed={seed} kr={keep_ratio}",
            unit="iter",
            leave=False,
        )
        for _ in greedy_bar:
            available = _complement_indices(selected)
            avail_in_pool = available[np.isin(available, candidate_pool, assume_unique=False)]
            if avail_in_pool.size == 0:
                avail_in_pool = available
            ranked = _pick_top_by_score(avail_in_pool, min(greedy_eval_k, avail_in_pool.size), cheap_score)
            best_item = None
            best_fit = -np.inf
            for cand in ranked:
                trial = np.concatenate([selected, np.asarray([cand], dtype=np.int64)])
                item = _evaluate_indices(trial)
                if float(item["fitness"]) > best_fit:
                    best_fit = float(item["fitness"])
                    best_item = item
            if best_item is None:
                # construct by cheap score for underfilled stage
                selected = np.concatenate([selected, ranked[:1]])
                continue
            selected = np.asarray(best_item["indices"], dtype=np.int64)
            score_hist.append(float(best_item["raw_score"]))
            class_corr_hist.append(float(lambda_cls * float(best_item["penalty"])))
            mean_corr_hist.append(float(lambda_mean * float(best_item["mean_penalty"])))
            fitness_hist.append(float(best_item["fitness"]))
            greedy_bar.set_postfix(best_fitness=f"{float(best_item['fitness']):.4f}")

        if selected.size < target_size:
            filler = _pick_top_by_score(_complement_indices(selected), target_size - selected.size, cheap_score)
            selected = np.unique(np.concatenate([selected, filler]).astype(np.int64))
        final_item = _evaluate_indices(selected)
        if not fitness_hist or abs(fitness_hist[-1] - float(final_item["fitness"])) > 1e-12:
            score_hist.append(float(final_item["raw_score"]))
            class_corr_hist.append(float(lambda_cls * float(final_item["penalty"])))
            mean_corr_hist.append(float(lambda_mean * float(final_item["mean_penalty"])))
            fitness_hist.append(float(final_item["fitness"]))
        return final_item, score_hist, class_corr_hist, mean_corr_hist, fitness_hist

    def _swap_refine(
        current_item: dict[str, object],
        candidate_pool: np.ndarray,
        cheap_score: np.ndarray,
        drop_k: int,
        add_k: int,
        max_rounds: int,
    ) -> tuple[dict[str, object], int, int, list[float], list[float], list[float], list[float]]:
        swap_eps = 1e-8
        rounds_used = 0
        accepted = 0
        score_hist: list[float] = []
        class_corr_hist: list[float] = []
        mean_corr_hist: list[float] = []
        fitness_hist: list[float] = []
        cur_item = current_item
        swap_bar = tqdm(
            range(max_rounds),
            desc=f"[group-swap] seed={seed} kr={keep_ratio}",
            unit="round",
            leave=False,
        )
        for _ in swap_bar:
            rounds_used += 1
            current = np.asarray(cur_item["indices"], dtype=np.int64)
            pool_minus = candidate_pool[np.isin(candidate_pool, current, invert=True, assume_unique=False)]
            if pool_minus.size == 0:
                break
            drop_order = np.argsort(guide_score_np[current], kind="mergesort")
            drop_cands = current[drop_order[: min(drop_k, current.size)]]
            add_cands = _pick_top_by_score(pool_minus, min(add_k, pool_minus.size), cheap_score)
            best_item = cur_item
            best_fit = float(cur_item["fitness"])
            for d in drop_cands:
                remain = current[current != d]
                for a in add_cands:
                    trial = np.concatenate([remain, np.asarray([a], dtype=np.int64)])
                    item = _evaluate_indices(trial)
                    fit = float(item["fitness"])
                    if fit > best_fit:
                        best_fit = fit
                        best_item = item
            if best_fit > float(cur_item["fitness"]) + swap_eps:
                cur_item = best_item
                accepted += 1
                score_hist.append(float(cur_item["raw_score"]))
                class_corr_hist.append(float(lambda_cls * float(cur_item["penalty"])))
                mean_corr_hist.append(float(lambda_mean * float(cur_item["mean_penalty"])))
                fitness_hist.append(float(cur_item["fitness"]))
                swap_bar.set_postfix(best_fitness=f"{float(cur_item['fitness']):.4f}", accepted=accepted)
            else:
                break
        return cur_item, rounds_used, accepted, score_hist, class_corr_hist, mean_corr_hist, fitness_hist

    print(
        f"[GroupLambda] dataset={dataset_name} kr={keep_ratio} "
        f"scale_cls={scale_cls:.6f} scale_mean={scale_mean:.6f} "
        f"lambda_std_cls={lambda_std_cls:.8f} lambda_std_mean={lambda_std_mean:.8f} "
        f"lambda_cls={lambda_cls:.8f} lambda_mean={lambda_mean:.8f}"
    )

    candidate_pool = _build_candidate_pool(guide_score_np, div_static_np, labels_np, target_size, num_classes)
    greedy_eval_k = min(64, int(candidate_pool.size))
    drop_k = min(max(8, int(0.05 * target_size)), target_size)
    add_k = min(max(32, int(0.10 * target_size)), max(1, int(candidate_pool.size - target_size)))
    max_swap_rounds = 20

    cheap_a = guide_score_np + 0.5 * div_static_np
    cheap_b = 0.7 * guide_score_np + 1.3 * div_static_np

    item_a, s1, c1, m1, f1 = _greedy_construct(candidate_pool, cheap_a, greedy_eval_k)
    init_fitness_a = float(item_a["fitness"])
    item_a, rounds_a, accepted_a, s1r, c1r, m1r, f1r = _swap_refine(item_a, candidate_pool, cheap_a, drop_k, add_k, max_swap_rounds)
    hist_a = (s1 + s1r, c1 + c1r, m1 + m1r, f1 + f1r)

    item_b, s2, c2, m2, f2 = _greedy_construct(candidate_pool, cheap_b, greedy_eval_k)
    init_fitness_b = float(item_b["fitness"])
    item_b, rounds_b, accepted_b, s2r, c2r, m2r, f2r = _swap_refine(item_b, candidate_pool, cheap_b, drop_k, add_k, max_swap_rounds)
    hist_b = (s2 + s2r, c2 + c2r, m2 + m2r, f2 + f2r)

    if float(item_b["fitness"]) > float(item_a["fitness"]):
        best_item = item_b
        score_history, class_correction_history, mean_correction_history, fitness_history = hist_b
        rounds_used = rounds_b
        accepted_swaps = accepted_b
        init_fitness = init_fitness_b
    else:
        best_item = item_a
        score_history, class_correction_history, mean_correction_history, fitness_history = hist_a
        rounds_used = rounds_a
        accepted_swaps = accepted_a
        init_fitness = init_fitness_a

    final_mask = np.asarray(best_item["mask"], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    final_rate = float(final_mask.mean())
    stats: dict[str, object] = {
        "solver": "greedy_swap",
        "sr": float(sr),
        "final_rate": final_rate,
        "candidate_pool_size": int(candidate_pool.size),
        "greedy_eval_k": int(greedy_eval_k),
        "drop_k": int(drop_k),
        "add_k": int(add_k),
        "max_swap_rounds": int(max_swap_rounds),
        "num_swap_rounds_used": int(rounds_used),
        "num_accepted_swaps": int(accepted_swaps),
        "init_fitness": float(init_fitness),
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
