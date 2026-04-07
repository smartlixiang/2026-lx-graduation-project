from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from tqdm import tqdm

from utils.proxy_log_utils import load_dataset_labels
from utils.score_utils import quantile_minmax

Q_LOW_DEFAULT = 0.002
Q_HIGH_DEFAULT = 0.998
K_RATIO_DEFAULT = 0.05
EPS = 1e-8


@dataclass
class FoldLogData:
    """One CV fold log with train/val indices and logits."""

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_logits: np.ndarray
    val_logits: np.ndarray


@dataclass
class DynamicComponentResult:
    """Unified output schema for each dynamic component."""

    raw_foldwise: np.ndarray
    fold_normalized: np.ndarray
    aggregated: np.ndarray
    final_normalized: np.ndarray


def assert_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/inf values.")


def safe_standardize(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(values)) if values.size else 0.0
    std = float(np.nanstd(values)) if values.size else 0.0
    if not np.isfinite(mean):
        mean = 0.0
    if (not np.isfinite(std)) or std < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / (std + eps)).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, samples, classes).")
    safe_logits = np.nan_to_num(logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    shifted = safe_logits - np.max(safe_logits, axis=2, keepdims=True)
    exp_shifted = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.sum(exp_shifted, axis=2, keepdims=True)
    denom = np.where(denom > EPS, denom, 1.0)
    probs = exp_shifted / denom
    assert_finite("softmax(probs)", probs)
    return probs.astype(np.float32)


def true_class_probabilities(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if probs.ndim != 3:
        raise ValueError("probs must have shape (epochs, samples, classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[1]:
        raise ValueError("labels shape mismatch for probabilities.")
    label_idx = labels.reshape(1, -1, 1)
    r = np.take_along_axis(probs, label_idx, axis=2).squeeze(2)
    r = np.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
    assert_finite("true_class_probabilities", r)
    return r.astype(np.float32)


def confusion_distribution_wo_true(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build q by zeroing true-class prob and renormalizing over non-true classes."""
    if probs.ndim != 3:
        raise ValueError("probs must have shape (epochs, samples, classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[1]:
        raise ValueError("labels shape mismatch for confusion distribution.")

    q = np.nan_to_num(probs.astype(np.float64, copy=True), nan=0.0, posinf=1.0, neginf=0.0)
    sample_idx = np.arange(probs.shape[1])
    q[:, sample_idx, labels] = 0.0
    denom = np.sum(q, axis=2, keepdims=True)
    denom = np.where(denom > EPS, denom, 1.0)
    q = q / denom
    q[:, sample_idx, labels] = 0.0
    assert_finite("confusion_distribution_wo_true", q)
    return q.astype(np.float32)


def compute_class_knn_mean_distance(
    features: np.ndarray,
    labels: np.ndarray,
    k_ratio: float = K_RATIO_DEFAULT,
    *,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Per-sample class-wise KNN mean Euclidean distance."""
    if features.ndim != 2:
        raise ValueError("features must be 2D: (num_samples, dim).")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels shape mismatch for knn distance.")
    if k_ratio <= 0:
        raise ValueError("k_ratio must be positive.")

    num_samples = features.shape[0]
    out = np.zeros(num_samples, dtype=np.float32)
    labels = labels.astype(np.int64)

    classes = np.unique(labels)
    class_iter = tqdm(classes, desc=progress_desc, unit="class", leave=False) if progress_desc else classes
    for cls in class_iter:
        idx = np.where(labels == int(cls))[0]
        n_c = idx.size
        if n_c <= 1:
            out[idx] = 0.0
            continue

        k = max(1, min(n_c - 1, int(math.ceil(k_ratio * n_c))))
        feats = np.nan_to_num(features[idx].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.sum(feats * feats, axis=1)
        dist2 = norms[:, None] + norms[None, :] - 2.0 * (feats @ feats.T)
        dist2 = np.maximum(dist2, 0.0)
        np.fill_diagonal(dist2, np.inf)
        nearest = np.partition(dist2, k - 1, axis=1)[:, :k]
        out[idx] = np.mean(np.sqrt(nearest), axis=1).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert_finite("compute_class_knn_mean_distance", out)
    return out


def quantile_minmax_dynamic(values: np.ndarray, q_low: float = Q_LOW_DEFAULT, q_high: float = Q_HIGH_DEFAULT) -> np.ndarray:
    normalized = quantile_minmax(values.astype(np.float32), q_low=q_low, q_high=q_high, fallback_value=0.5)
    normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
    assert_finite("quantile_minmax_dynamic", normalized)
    return normalized.astype(np.float32)


def resolve_epoch_windows(num_epochs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return early/mid/late epoch index arrays with robust handling for short runs."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    if num_epochs == 1:
        idx = np.array([0], dtype=np.int64)
        return idx, idx, idx
    if num_epochs == 2:
        return np.array([0], dtype=np.int64), np.array([0, 1], dtype=np.int64), np.array([1], dtype=np.int64)

    early_n = max(1, int(round(0.3 * num_epochs)))
    mid_n = max(1, int(round(0.4 * num_epochs)))
    late_n = max(1, num_epochs - early_n - mid_n)

    while early_n + mid_n + late_n > num_epochs:
        if mid_n >= max(early_n, late_n) and mid_n > 1:
            mid_n -= 1
        elif late_n > 1:
            late_n -= 1
        elif early_n > 1:
            early_n -= 1
        else:
            break
    while early_n + mid_n + late_n < num_epochs:
        mid_n += 1

    boundaries = np.cumsum([early_n, mid_n, late_n])
    early_idx = np.arange(0, boundaries[0], dtype=np.int64)
    mid_idx = np.arange(boundaries[0], boundaries[1], dtype=np.int64)
    late_idx = np.arange(boundaries[1], num_epochs, dtype=np.int64)

    if early_idx.size == 0:
        early_idx = np.array([0], dtype=np.int64)
    if mid_idx.size == 0:
        mid_idx = np.array([num_epochs // 2], dtype=np.int64)
    if late_idx.size == 0:
        late_idx = np.array([num_epochs - 1], dtype=np.int64)

    return early_idx, mid_idx, late_idx


def aggregate_train_fold_component(fold_values: np.ndarray, folds: list[FoldLogData]) -> np.ndarray:
    num_folds, num_samples = fold_values.shape
    if num_folds != len(folds):
        raise ValueError("fold_values rows must match folds length.")
    agg_sum = np.zeros(num_samples, dtype=np.float64)
    agg_count = np.zeros(num_samples, dtype=np.int64)
    for f_idx, fold in enumerate(folds):
        idx = fold.train_indices
        vals = fold_values[f_idx, idx]
        finite = np.isfinite(vals)
        agg_sum[idx[finite]] += vals[finite].astype(np.float64)
        agg_count[idx[finite]] += 1
    if np.any(agg_count <= 0):
        missing = np.where(agg_count <= 0)[0]
        raise ValueError(f"Some samples never appeared in train folds: {missing[:10]}.")
    return (agg_sum / np.maximum(agg_count, 1)).astype(np.float32)


def aggregate_val_fold_component(fold_values: np.ndarray, folds: list[FoldLogData]) -> np.ndarray:
    num_folds, num_samples = fold_values.shape
    if num_folds != len(folds):
        raise ValueError("fold_values rows must match folds length.")
    aggregated = np.full(num_samples, np.nan, dtype=np.float32)
    for f_idx, fold in enumerate(folds):
        aggregated[fold.val_indices] = fold_values[f_idx, fold.val_indices]
    if np.any(~np.isfinite(aggregated)):
        missing = np.where(~np.isfinite(aggregated))[0]
        raise ValueError(f"Some samples missing validation-fold assignment: {missing[:10]}.")
    return aggregated


def load_cv_fold_logs(proxy_log_dir: str | Path, dataset_name: str, data_root: str) -> tuple[list[FoldLogData], np.ndarray]:
    """Load fold_*.npz proxy logs and dataset labels aligned to global sample indices."""
    log_dir = Path(proxy_log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Proxy log directory not found: {log_dir}")

    labels_all = load_dataset_labels(dataset_name, data_root)
    num_samples = labels_all.shape[0]

    fold_paths = sorted(log_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {log_dir}")

    folds: list[FoldLogData] = []
    seen_val = np.zeros(num_samples, dtype=np.int64)

    for fold_id, fold_path in enumerate(fold_paths):
        data = np.load(fold_path)
        required = ("train_indices", "val_indices", "train_logits", "val_logits")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing keys {missing} in fold file: {fold_path}")

        train_indices = data["train_indices"].astype(np.int64)
        val_indices = data["val_indices"].astype(np.int64)
        train_logits = data["train_logits"].astype(np.float32)
        val_logits = data["val_logits"].astype(np.float32)

        if np.any(train_indices < 0) or np.any(train_indices >= num_samples):
            raise ValueError(f"train_indices out of range in {fold_path}")
        if np.any(val_indices < 0) or np.any(val_indices >= num_samples):
            raise ValueError(f"val_indices out of range in {fold_path}")
        if train_logits.ndim != 3 or val_logits.ndim != 3:
            raise ValueError(f"train_logits/val_logits must be 3D in {fold_path}")
        if train_logits.shape[1] != train_indices.shape[0]:
            raise ValueError(f"train_logits sample dimension mismatch in {fold_path}")
        if val_logits.shape[1] != val_indices.shape[0]:
            raise ValueError(f"val_logits sample dimension mismatch in {fold_path}")
        if train_logits.shape[0] != val_logits.shape[0] or train_logits.shape[2] != val_logits.shape[2]:
            raise ValueError(f"train/val logits shape mismatch in {fold_path}")

        seen_val[val_indices] += 1
        folds.append(
            FoldLogData(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_logits=train_logits,
                val_logits=val_logits,
            )
        )

    if not np.all(seen_val == 1):
        raise ValueError("Validation indices across folds must cover each sample exactly once.")

    return folds, labels_all.astype(np.int64)


def default_dynamic_cache_path(
    dataset: str,
    *,
    proxy_model: str = "resnet18",
    epochs: int | None = None,
) -> Path:
    epoch_tag = str(int(epochs)) if epochs is not None else "latest"
    return Path("weights") / "dynamic_cache" / dataset / proxy_model / epoch_tag / "dynamic_components.npz"
