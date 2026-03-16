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
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.group_lambda import (  # noqa: E402
    DEFAULT_CLS_LAMBDA_BASE,
    DEFAULT_EPS,
    DEFAULT_M,
    DEFAULT_MEAN_LAMBDA_BASE,
    compute_balance_penalty,
    get_or_estimate_lambda,
    get_default_mean_lambda_base,
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
    parser.add_argument("--group-iterations", type=int, default=800)
    parser.add_argument(
        "--group-batch-size",
        type=int,
        default=24,
        help="kr<=50 时的初始 batch_size；kr>50 时自动减半",
    )
    parser.add_argument(
        "--group-min-batch-size",
        type=int,
        default=2,
        help="kr<=50 时的最小 batch_size；kr>50 时自动减半",
    )
    parser.add_argument("--group-eval-interval", type=int, default=4)
    parser.add_argument(
        "--group-candidate-pool-multiplier",
        type=float,
        default=10,
        help="候选池大小倍率，按当前 batch_size 的倍率分别构造子集内外候选池",
    )
    parser.add_argument(
        "--score-patience",
        type=int,
        default=5,
        help="连续多少次 eval 综合评分(S') 不上涨后才调整 batch_size",
    )
    parser.add_argument(
        "--herding-patience",
        type=int,
        default=3,
        help="herding 修正项连续上升多少次后才调整 batch_size",
    )
    parser.add_argument(
        "--score-decay",
        type=float,
        default=0.8,
        help="因综合评分不上涨触发 batch_size 衰减时的倍率",
    )
    parser.add_argument(
        "--herding-decay",
        type=float,
        default=0.9,
        help="因 herding 修正项连续上升触发 batch_size 衰减时的倍率",
    )
    parser.add_argument("--mean-lambda-base", type=float, default=None, help="kr=20 时 mean 修正项目标占比（默认随数据集变化）")
    parser.add_argument("--cls-lambda-base", type=float, default=DEFAULT_CLS_LAMBDA_BASE, help="kr=20 时 class 修正项目标占比")
    parser.add_argument(
        "--group-tolerance-ratio",
        type=float,
        default=0.85,
        help="最终解容忍比例：选择比值>=最优比值*tolerance_ratio 的最晚轮次",
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
            raise FileNotFoundError(f"tiny_imagenet train split not found: {train_root}")
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
    group_iterations: int = 200,
    group_batch_size: int = 64,
    group_min_batch_size: int = 2,
    group_eval_interval: int = 10,
    group_candidate_pool_multiplier: float = 3.0,
    score_patience: int = 5,
    herding_patience: int = 3,
    score_decay: float = 0.8,
    herding_decay: float = 0.9,
    cls_lambda_base: float = DEFAULT_CLS_LAMBDA_BASE,
    mean_lambda_base: float = DEFAULT_MEAN_LAMBDA_BASE,
    tolerance_ratio: float = 0.95,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")
    if tolerance_ratio <= 0:
        raise ValueError("tolerance_ratio 必须大于 0。")

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
    full_class_mean, full_class_var = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    full_class_mean_f32 = full_class_mean.astype(np.float32, copy=False)
    full_class_var_f32 = full_class_var.astype(np.float32, copy=False)

    static_part = (weights["sa"] * sa_scores_np + weights["dds"] * dds_static_np).astype(np.float32)
    static_quality = (sa_scores_np + dds_static_np).astype(np.float32)

    sum_features = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    cnt_features = np.zeros(num_classes, dtype=np.int64)

    def _indices_to_mask(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(num_samples, dtype=np.uint8)
        mask[np.asarray(indices, dtype=np.int64)] = 1
        return mask

    def _compute_mean_penalty(cur_mask: np.ndarray) -> float:
        # mean penalty uses class-averaged variance-normalized centroid shift
        # so lambda calibration from random baseline is directly comparable across datasets.
        selected = np.flatnonzero(cur_mask > 0).astype(np.int64)
        if selected.size == 0:
            return 0.0
        sum_features.fill(0.0)
        cnt_features.fill(0)
        sel_labels = labels_np[selected]
        np.add.at(sum_features, sel_labels, div_features_np[selected])
        np.add.at(cnt_features, sel_labels, 1)
        active = cnt_features > 0
        if not np.any(active):
            return 0.0
        selected_mean = np.zeros_like(full_class_mean_f32, dtype=np.float32)
        selected_mean[active] = sum_features[active] / cnt_features[active, None].astype(np.float32)
        diff = selected_mean[active] - full_class_mean_f32[active]
        dist2 = np.sum(diff * diff, axis=1, dtype=np.float32)
        normalized = dist2 / (full_class_var_f32[active] + float(DEFAULT_EPS))
        if num_classes <= 0:
            return 0.0
        return float(np.sum(normalized, dtype=np.float32) / float(num_classes))

    def _evaluate(indices: np.ndarray) -> dict[str, object]:
        unique = np.unique(np.asarray(indices, dtype=np.int64))
        if unique.size != target_size:
            raise ValueError(f"_evaluate expects fixed-cardinality subset, got {unique.size} != {target_size}")
        mask = _indices_to_mask(unique)
        div_scores = np.asarray(
            div_metric.score_dataset_dynamic(
                div_loader,
                adapter=image_adapter,
                selected_mask=mask,
                image_features=div_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        s_ref = static_part + weights["div"] * div_scores
        selected = mask.astype(bool)
        raw_score = float(np.sum(s_ref[selected]))
        class_penalty = compute_balance_penalty(mask, labels_np, num_classes, target_size)
        mean_penalty = _compute_mean_penalty(mask)
        class_corr = float(lambda_cls * class_penalty)
        mean_corr = float(lambda_mean * mean_penalty)
        return {
            "indices": unique,
            "mask": mask,
            "raw_score": raw_score,
            "penalty": float(class_penalty),
            "mean_penalty": float(mean_penalty),
            "class_corr": class_corr,
            "mean_corr": mean_corr,
            "fitness": float(raw_score - class_corr - mean_corr),
            "counts": np.bincount(labels_np[selected], minlength=num_classes).astype(np.int64),
        }

    def _eval_raw_score_mask(cur_mask: np.ndarray) -> float:
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
        return float(np.sum(s_ref[cur_mask.astype(bool)]))

    lambda_record = get_or_estimate_lambda(
        cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
        dataset=dataset_name,
        seed=seed,
        kr=keep_ratio,
        weight_group=weight_group,
        n_samples=num_samples,
        target_size=target_size,
        eval_score_fn=_eval_raw_score_mask,
        penalty_fn=lambda cur_mask: compute_balance_penalty(cur_mask, labels_np, num_classes, target_size),
        mean_penalty_fn=lambda cur_mask: _compute_mean_penalty(cur_mask),
        M=DEFAULT_M,
        eps=DEFAULT_EPS,
        cls_lambda_base=cls_lambda_base,
        mean_lambda_base=mean_lambda_base,
        tqdm_desc=f"Estimating lambdas (seed={seed}, kr={keep_ratio}, wg={weight_group})",
    )
    lambda_cls = float(lambda_record["lambda_cls"])
    lambda_mean = float(lambda_record["lambda_mean"])
    print(
        f"[Lambda] dataset={dataset_name} | seed={seed} | kr={keep_ratio} "
        f"| raw_mean={float(lambda_record['raw_mean']):.8f} "
        f"| class_penalty_mean={float(lambda_record['class_penalty_mean']):.8f} "
        f"| mean_penalty_mean={float(lambda_record['mean_penalty_mean']):.8f} "
        f"| lambda_cls={lambda_cls:.8f} | lambda_mean={lambda_mean:.8f} "
        f"| cls_lambda_base={float(cls_lambda_base):.6f} | mean_lambda_base={float(mean_lambda_base):.6f}"
    )

    def _rank_points(scores_np: np.ndarray, descending: bool) -> np.ndarray:
        n = int(scores_np.size)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        order = np.argsort(-scores_np if descending else scores_np, kind="mergesort")
        points = np.zeros(n, dtype=np.int64)
        points[order] = np.arange(n, 0, -1, dtype=np.int64)
        return points

    def _candidate_sampling_probs(
        candidate_indices: np.ndarray,
        class_counts_cur: np.ndarray,
        target_size_cur: int,
        num_classes_cur: int,
        labels_arr: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        n = int(candidate_indices.size)
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        if target_size_cur <= 0 or num_classes_cur <= 0:
            return np.full(n, 1.0 / float(n), dtype=np.float64)

        target_per_class = float(target_size_cur) / float(num_classes_cur)
        if target_per_class <= 0:
            return np.full(n, 1.0 / float(n), dtype=np.float64)

        cand_labels = labels_arr[candidate_indices]
        a = (target_per_class - class_counts_cur[cand_labels].astype(np.float64)) / target_per_class
        a = np.clip(a, -0.1, 0.1)

        if mode == "in":
            logits = 1.0 + a
        elif mode == "out":
            logits = 1.0 - a
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")

        logits = logits - np.max(logits)
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        if not np.isfinite(denom) or denom <= 0.0:
            return np.full(n, 1.0 / float(n), dtype=np.float64)
        return probs / denom

    rng = np.random.default_rng(seed)
    current_indices = np.sort(rng.choice(num_samples, size=target_size, replace=False).astype(np.int64))
    class_counts = np.bincount(labels_np[current_indices], minlength=num_classes).astype(np.int64)
    class_sums = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    np.add.at(class_sums, labels_np[current_indices], div_features_np[current_indices])

    # 初始真实评估
    current_item = _evaluate(current_indices)
    init_item = dict(current_item)

    def _selection_metric(item: dict[str, object]) -> float:
        delta = float(item["fitness"]) - float(init_item["fitness"])
        herding_corr = float(item["mean_corr"])
        return float(delta / (herding_corr + float(DEFAULT_EPS)))

    best_item = dict(current_item)
    best_item["indices"] = np.asarray(current_item["indices"], dtype=np.int64).copy()
    best_item["mask"] = np.asarray(current_item["mask"], dtype=np.uint8).copy()
    best_metric_value = _selection_metric(best_item)
    last_eval_fit = float(current_item["fitness"])

    score_history = [float(current_item["raw_score"])]
    class_correction_history = [float(current_item["class_corr"])]
    mean_correction_history = [float(current_item["mean_corr"])]
    fitness_history = [float(current_item["fitness"])]
    eval_steps = [0]
    eval_items: list[dict[str, object]] = [
        {
            "indices": np.asarray(current_item["indices"], dtype=np.int64).copy(),
            "mask": np.asarray(current_item["mask"], dtype=np.uint8).copy(),
            "raw_score": float(current_item["raw_score"]),
            "penalty": float(current_item["penalty"]),
            "mean_penalty": float(current_item["mean_penalty"]),
            "class_corr": float(current_item["class_corr"]),
            "mean_corr": float(current_item["mean_corr"]),
            "fitness": float(current_item["fitness"]),
        }
    ]

    eval_every = max(1, int(group_eval_interval))
    cur_batch_size = max(1, int(group_batch_size))
    min_batch_size = max(1, int(group_min_batch_size))
    candidate_pool_multiplier = max(1.0, float(group_candidate_pool_multiplier))
    score_patience = max(1, int(score_patience))
    herding_patience = max(1, int(herding_patience))
    score_decay = float(np.clip(score_decay, 0.01, 0.999))
    herding_decay = float(np.clip(herding_decay, 0.01, 0.999))
    if cur_batch_size < min_batch_size:
        cur_batch_size = min_batch_size
    stale_score_count = 0
    herding_rise_count = 0
    best_fitness_for_decay = float(current_item["fitness"])
    prev_mean_corr = float(current_item["mean_corr"])

    iter_bar = tqdm(range(max(0, int(group_iterations))), desc="[group] iterations", unit="iter")
    for step in iter_bar:
        membership = np.zeros(num_samples, dtype=bool)
        membership[current_indices] = True
        in_indices = current_indices
        out_indices = all_indices[~membership]
        k = min(int(cur_batch_size), in_indices.size, out_indices.size)
        if k <= 0:
            break
        candidate_pool_size = max(1, int(np.ceil(float(cur_batch_size) * candidate_pool_multiplier)))

        mu_sub = np.zeros_like(class_sums, dtype=np.float32)
        active = class_counts > 0
        mu_sub[active] = class_sums[active] / class_counts[active, None].astype(np.float32)

        in_labels = labels_np[in_indices]
        out_labels = labels_np[out_indices]
        in_feats = div_features_np[in_indices]
        out_feats = div_features_np[out_indices]

        in_static_drop = -static_quality[in_indices].astype(np.float32)
        out_static_add = static_quality[out_indices].astype(np.float32)

        in_div_dist = np.linalg.norm(in_feats - mu_sub[in_labels], axis=1)
        in_div_drop = -in_div_dist
        out_div_add = np.linalg.norm(out_feats - mu_sub[out_labels], axis=1)
        empty_cls_mask = class_counts[out_labels] == 0
        out_div_add[empty_cls_mask] = 1e6

        in_old_dist = np.linalg.norm(mu_sub[in_labels] - full_class_mean_f32[in_labels], axis=1)
        cnt_in = class_counts[in_labels].astype(np.float32)
        safe_denom_drop = np.maximum(cnt_in - 1.0, 1.0)
        in_new_mean = (class_sums[in_labels] - in_feats) / safe_denom_drop[:, None]
        in_new_dist = np.linalg.norm(in_new_mean - full_class_mean_f32[in_labels], axis=1)
        in_new_dist[cnt_in <= 1.0] = 0.0
        in_mean_drop = in_old_dist - in_new_dist

        out_old_dist = np.linalg.norm(mu_sub[out_labels] - full_class_mean_f32[out_labels], axis=1)
        cnt_out = class_counts[out_labels].astype(np.float32)
        out_new_mean = (class_sums[out_labels] + out_feats) / (cnt_out[:, None] + 1.0)
        out_new_dist = np.linalg.norm(out_new_mean - full_class_mean_f32[out_labels], axis=1)
        out_mean_add = out_old_dist - out_new_dist

        score_in = (
            _rank_points(in_static_drop, descending=True)
            + _rank_points(in_div_drop, descending=True)
            + _rank_points(in_mean_drop, descending=True)
        )
        score_out = (
            _rank_points(out_static_add, descending=True)
            + _rank_points(out_div_add, descending=True)
            + _rank_points(out_mean_add, descending=True)
        )

        in_candidate_size = min(candidate_pool_size, in_indices.size)
        out_candidate_size = min(candidate_pool_size, out_indices.size)
        in_candidate_pool = in_indices[np.argsort(-score_in, kind="mergesort")[:in_candidate_size]]
        out_candidate_pool = out_indices[np.argsort(-score_out, kind="mergesort")[:out_candidate_size]]

        k = min(int(cur_batch_size), in_candidate_pool.size, out_candidate_pool.size)
        if k <= 0:
            break
        drop_probs = _candidate_sampling_probs(
            in_candidate_pool,
            class_counts,
            target_size,
            num_classes,
            labels_np,
            mode="out",
        )
        add_probs = _candidate_sampling_probs(
            out_candidate_pool,
            class_counts,
            target_size,
            num_classes,
            labels_np,
            mode="in",
        )
        drop_idx = rng.choice(in_candidate_pool, size=k, replace=False, p=drop_probs).astype(np.int64)
        add_idx = rng.choice(out_candidate_pool, size=k, replace=False, p=add_probs).astype(np.int64)

        membership[drop_idx] = False
        membership[add_idx] = True
        current_indices = np.flatnonzero(membership).astype(np.int64)
        if current_indices.size != target_size:
            raise RuntimeError("group swap 后子集大小异常。")

        # 增量更新 class_sums/class_counts
        drop_labels = labels_np[drop_idx]
        add_labels = labels_np[add_idx]
        np.add.at(class_sums, drop_labels, -div_features_np[drop_idx])
        np.add.at(class_sums, add_labels, div_features_np[add_idx])
        np.add.at(class_counts, drop_labels, -1)
        np.add.at(class_counts, add_labels, 1)

        approx_total_rank = float(np.mean(score_out[np.argsort(-score_out, kind="mergesort")[:k]]) -
                                  np.mean(score_in[np.argsort(-score_in, kind="mergesort")[:k]]))

        is_last_iter = step == max(0, int(group_iterations)) - 1
        should_eval = ((step + 1) % eval_every == 0) or is_last_iter
        if should_eval:
            current_item = _evaluate(current_indices)
            last_eval_fit = float(current_item["fitness"])
            eval_steps.append(int(step + 1))
            score_history.append(float(current_item["raw_score"]))
            class_correction_history.append(float(current_item["class_corr"]))
            mean_correction_history.append(float(current_item["mean_corr"]))
            fitness_history.append(float(current_item["fitness"]))
            eval_items.append(
                {
                    "indices": np.asarray(current_item["indices"], dtype=np.int64).copy(),
                    "mask": np.asarray(current_item["mask"], dtype=np.uint8).copy(),
                    "raw_score": float(current_item["raw_score"]),
                    "penalty": float(current_item["penalty"]),
                    "mean_penalty": float(current_item["mean_penalty"]),
                    "class_corr": float(current_item["class_corr"]),
                    "mean_corr": float(current_item["mean_corr"]),
                    "fitness": float(current_item["fitness"]),
                }
            )
            cur_metric_value = _selection_metric(current_item)
            improved = cur_metric_value > best_metric_value
            if improved:
                best_item = dict(current_item)
                best_item["indices"] = np.asarray(current_item["indices"], dtype=np.int64).copy()
                best_item["mask"] = np.asarray(current_item["mask"], dtype=np.uint8).copy()
                best_metric_value = cur_metric_value

            cur_fitness = float(current_item["fitness"])
            if cur_fitness > best_fitness_for_decay:
                best_fitness_for_decay = cur_fitness
                stale_score_count = 0
            else:
                stale_score_count += 1

            cur_mean_corr = float(current_item["mean_corr"])
            if cur_mean_corr > prev_mean_corr:
                herding_rise_count += 1
            else:
                herding_rise_count = 0
            prev_mean_corr = cur_mean_corr

            if stale_score_count >= score_patience and cur_batch_size > min_batch_size:
                new_batch_size = max(min_batch_size, int(np.floor(cur_batch_size * score_decay)))
                cur_batch_size = max(min_batch_size, new_batch_size)
                stale_score_count = 0

            if herding_rise_count >= herding_patience and cur_batch_size > min_batch_size:
                new_batch_size = max(min_batch_size, int(np.floor(cur_batch_size * herding_decay)))
                cur_batch_size = max(min_batch_size, new_batch_size)
                herding_rise_count = 0

        iter_bar.set_postfix(
            best_metric=f"{best_metric_value:.4f}",
            cur_sprime=f"{last_eval_fit:.4f}",
            cur_herd=f"{float(current_item['mean_corr']):.4f}",
            batch_size=int(cur_batch_size),
        )

    metric_history = np.asarray([
        (f - float(init_item["fitness"])) / (m + float(DEFAULT_EPS))
        for f, m in zip(fitness_history, mean_correction_history)
    ], dtype=np.float64)
    best_metric_history_value = float(np.max(metric_history)) if metric_history.size > 0 else float(best_metric_value)
    metric_threshold = float(best_metric_history_value * float(tolerance_ratio))
    qualified_indices = np.flatnonzero(metric_history >= metric_threshold)
    if qualified_indices.size > 0:
        final_eval_pos = int(qualified_indices[-1])
    else:
        final_eval_pos = int(np.argmax(metric_history)) if metric_history.size > 0 else 0
    final_item = dict(eval_items[final_eval_pos]) if eval_items else dict(best_item)
    final_eval_step = int(eval_steps[final_eval_pos]) if eval_steps else 0

    final_mask = np.asarray(final_item["mask"], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    stats: dict[str, object] = {
        "solver": "group_batch_swap",
        "sr": float(sr),
        "final_rate": float(final_mask.mean()),
        "init_fitness": float(fitness_history[0]),
        "final_fitness": float(final_item["fitness"]),
        "raw_score": float(final_item["raw_score"]),
        "penalty": float(final_item["penalty"]),
        "final_mean_penalty": float(final_item["mean_penalty"]),
        "S": float(final_item["raw_score"]),
        "S_prime": float(final_item["fitness"]),
        "selected_by_class": selected_by_class,
        "best_iter": int(eval_steps[int(np.argmax(np.asarray(fitness_history, dtype=np.float32)))]) if fitness_history else 0,
        "selected_best_iter": int(eval_steps[int(np.argmax(np.asarray([
            (f - float(init_item['fitness'])) / (m + float(DEFAULT_EPS))
            for f, _c, m in zip(fitness_history, class_correction_history, mean_correction_history)
        ], dtype=np.float32)))]) if fitness_history else 0,
        "final_selected_iter": int(final_eval_step),
        "selection_metric": float(best_metric_history_value),
        "selection_metric_threshold": float(metric_threshold),
        "selection_tolerance_ratio": float(tolerance_ratio),
        "init_S_prime": float(init_item["fitness"]),
        "init_herding_corr": float(init_item["mean_corr"]),
        "init_class_corr": float(init_item["class_corr"]),
        "best_S_prime": float(best_item["fitness"]),
        "best_herding_corr": float(best_item["mean_corr"]),
        "best_class_corr": float(best_item["class_corr"]),
        "final_herding_corr": float(final_item["mean_corr"]),
        "final_class_corr": float(final_item["class_corr"]),
        "best_sprime_over_herding": float(best_item["fitness"] / (float(best_item["mean_corr"]) + float(DEFAULT_EPS))),
        "final_sprime_over_herding": float(final_item["fitness"] / (float(final_item["mean_corr"]) + float(DEFAULT_EPS))),
        "score_history": score_history,
        "fitness_history": fitness_history,
        "correction_history": [float(c) + float(m) for c, m in zip(class_correction_history, mean_correction_history)],
        "class_correction_history": class_correction_history,
        "mean_correction_history": mean_correction_history,
        "eval_steps": eval_steps,
        "lambda_cls": float(lambda_cls),
        "lambda_mean": float(lambda_mean),
        "group_iterations": int(group_iterations),
        "group_batch_size": int(cur_batch_size),
        "group_batch_size_init": int(max(1, int(group_batch_size))),
        "group_min_batch_size": int(max(1, int(group_min_batch_size))),
        "group_eval_interval": int(eval_every),
        "group_candidate_pool_multiplier": float(candidate_pool_multiplier),
        "score_patience": int(score_patience),
        "herding_patience": int(herding_patience),
        "score_decay": float(score_decay),
        "herding_decay": float(herding_decay),
    }
    return final_mask, selected_by_class, stats


def _sanitize_for_filename(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def save_score_curve_plot(
    score_history: Sequence[float],
    class_correction_history: Sequence[float],
    mean_correction_history: Sequence[float],
    fitness_history: Sequence[float] | None = None,
    eval_steps: Sequence[int] | None = None,
    *,
    dataset: str,
    method: str,
    weight_group: str,
    model_name: str,
    seed: int,
    keep_ratio: int,
    clip_model: str,
    selected_eval_step: int | None = None,
    final_eval_step: int | None = None,
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

    if eval_steps is not None and len(eval_steps) == fit_arr.shape[0]:
        x = np.asarray(eval_steps, dtype=np.int32)
    else:
        x = np.arange(fit_arr.shape[0], dtype=np.int32)
    herding_corr_arr = np.asarray(mean_correction_history, dtype=np.float64)

    ax.plot(x, fit_arr, linewidth=1.8, color="#d62728", label="S'(D)")
    ax.set_title("Group optimization trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Comprehensive score S'(D)")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, herding_corr_arr, linewidth=1.5, color="#1f77b4", linestyle="--", label="Herding correction")
    ax2.set_ylabel("Herding correction")

    if selected_eval_step is not None and fit_arr.size > 0:
        selected_x = int(selected_eval_step)
        fit_y = float(np.interp(selected_x, x, fit_arr))
        herd_y = float(np.interp(selected_x, x, herding_corr_arr))
        ax.axvline(selected_x, color="green", linestyle="--", linewidth=1.6, alpha=0.9, label="Best ratio subset")
        ax.plot(selected_x, fit_y, marker=(5, 1), color="green", markersize=10, linestyle="None")
        ax2.plot(selected_x, herd_y, marker=(5, 1), color="green", markersize=10, linestyle="None")

    if final_eval_step is not None and fit_arr.size > 0:
        final_x = int(final_eval_step)
        ax.axvline(final_x, color="orange", linestyle="--", linewidth=1.6, alpha=0.9, label="Final selected subset")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    if args.mean_lambda_base is None:
        args.mean_lambda_base = get_default_mean_lambda_base(dataset_name)

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
                    group_iterations=args.group_iterations,
                    group_batch_size=args.group_batch_size,
                    group_min_batch_size=args.group_min_batch_size,
                    group_eval_interval=args.group_eval_interval,
                    group_candidate_pool_multiplier=args.group_candidate_pool_multiplier,
                    score_patience=args.score_patience,
                    herding_patience=args.herding_patience,
                    score_decay=args.score_decay,
                    herding_decay=args.herding_decay,
                    cls_lambda_base=args.cls_lambda_base,
                    mean_lambda_base=args.mean_lambda_base,
                    tolerance_ratio=args.group_tolerance_ratio,
                )
                debug_curve = save_score_curve_plot(
                    group_stats.get("score_history", []),
                    group_stats.get("class_correction_history", []),
                    group_stats.get("mean_correction_history", []),
                    fitness_history=group_stats.get("fitness_history", []),
                    eval_steps=group_stats.get("eval_steps"),
                    selected_eval_step=int(group_stats.get("selected_best_iter", 0)),
                    final_eval_step=int(group_stats.get("final_selected_iter", 0)),
                    dataset=dataset_name,
                    method=method,
                    weight_group=weight_group,
                    model_name=args.model_name,
                    seed=seed,
                    keep_ratio=keep_ratio,
                    clip_model=args.clip_model,
                )
                print(f"[Debug] score curve saved to: {debug_curve}")

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
                    "group_stats: "
                    f"sr={group_stats['sr']:.6f} | rate={group_stats['final_rate']:.6f} | "
                    f"m_c={group_stats['selected_by_class']} | "
                    f"best_iter={group_stats['best_iter']} | "
                    f"selected_best_iter={group_stats['selected_best_iter']} | "
                    f"final_selected_iter={group_stats['final_selected_iter']} | "
                    f"S(D)={group_stats['S']:.6f} | pen(D)={group_stats['penalty']:.6f} | "
                    f"mean_pen(D)={group_stats['final_mean_penalty']:.6f} | "
                    f"lambda_mean={group_stats['lambda_mean']:.6f} | S'(D)={group_stats['S_prime']:.6f} | "
                    f"metric={group_stats['selection_metric']:.6f} | "
                    f"metric_th={group_stats['selection_metric_threshold']:.6f} | "
                    f"tol={group_stats['selection_tolerance_ratio']:.3f}"
                )
                print(
                    "best_ratio: "
                    f"{group_stats['best_S_prime']:.2f}/{group_stats['best_herding_corr']:.2f}="
                    f"{group_stats['best_sprime_over_herding']:.2f} | "
                    f"class_corr={group_stats['best_class_corr']:.2f}"
                )
                print(
                    "final_ratio: "
                    f"{group_stats['S_prime']:.2f}/{group_stats['final_herding_corr']:.2f}="
                    f"{group_stats['final_sprime_over_herding']:.2f} | "
                    f"class_corr={group_stats['final_class_corr']:.2f}"
                )
                print(
                    "random_init_stats: "
                    f"score={group_stats['init_S_prime']:.2f} | "
                    f"herding_corr={group_stats['init_herding_corr']:.2f} | "
                    f"class_corr={group_stats['init_class_corr']:.2f}"
                )
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
