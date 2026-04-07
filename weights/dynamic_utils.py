from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

from utils.proxy_log_utils import load_dataset_labels
from utils.score_utils import quantile_minmax

Q_LOW_DEFAULT = 0.002
Q_HIGH_DEFAULT = 0.998
K_RATIO_DEFAULT = 0.05


@dataclass
class FoldLogData:
    """One CV fold log with train/val indices and logits."""

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_logits: np.ndarray
    val_logits: np.ndarray


def assert_finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/inf values.")


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, samples, classes).")
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=2, keepdims=True)
    assert_finite("softmax(probs)", probs)
    return probs.astype(np.float32)


def true_class_probabilities(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if probs.ndim != 3:
        raise ValueError("probs must have shape (epochs, samples, classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[1]:
        raise ValueError("labels shape mismatch for probabilities.")
    label_idx = labels.reshape(1, -1, 1)
    r = np.take_along_axis(probs, label_idx, axis=2).squeeze(2)
    assert_finite("true_class_probabilities", r)
    return r.astype(np.float32)


def confusion_distribution_wo_true(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Build q by zeroing true-class prob and renormalizing over non-true classes."""
    if probs.ndim != 3:
        raise ValueError("probs must have shape (epochs, samples, classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[1]:
        raise ValueError("labels shape mismatch for confusion distribution.")

    q = probs.astype(np.float64, copy=True)
    sample_idx = np.arange(probs.shape[1])
    q[:, sample_idx, labels] = 0.0
    denom = np.sum(q, axis=2, keepdims=True)
    tiny_mask = denom <= 1e-12
    denom[tiny_mask] = 1.0
    q = q / denom
    q[:, sample_idx, labels] = 0.0
    assert_finite("confusion_distribution_wo_true", q)
    return q.astype(np.float32)


def compute_class_knn_mean_distance(features: np.ndarray, labels: np.ndarray, k_ratio: float = K_RATIO_DEFAULT) -> np.ndarray:
    """Per-sample class-wise KNN mean Euclidean distance.

    For class size n_c, k = max(1, min(n_c - 1, ceil(k_ratio * n_c))).
    If n_c <= 1, distance is set to 0 for those samples.
    """
    if features.ndim != 2:
        raise ValueError("features must be 2D: (num_samples, dim).")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels shape mismatch for knn distance.")
    if k_ratio <= 0:
        raise ValueError("k_ratio must be positive.")

    num_samples = features.shape[0]
    out = np.zeros(num_samples, dtype=np.float32)
    labels = labels.astype(np.int64)

    for cls in np.unique(labels):
        mask = labels == int(cls)
        idx = np.where(mask)[0]
        n_c = idx.size
        if n_c <= 1:
            out[idx] = 0.0
            continue
        k = max(1, min(n_c - 1, int(math.ceil(k_ratio * n_c))))

        feats = features[idx].astype(np.float64)
        norms = np.sum(feats * feats, axis=1)
        dist2 = norms[:, None] + norms[None, :] - 2.0 * (feats @ feats.T)
        dist2 = np.maximum(dist2, 0.0)
        np.fill_diagonal(dist2, np.inf)
        nearest = np.partition(dist2, k, axis=1)[:, :k]
        out[idx] = np.mean(np.sqrt(nearest), axis=1).astype(np.float32)

    assert_finite("compute_class_knn_mean_distance", out)
    return out


def quantile_minmax_dynamic(values: np.ndarray, q_low: float = Q_LOW_DEFAULT, q_high: float = Q_HIGH_DEFAULT) -> np.ndarray:
    normalized = quantile_minmax(values.astype(np.float32), q_low=q_low, q_high=q_high, fallback_value=0.5)
    assert_finite("quantile_minmax_dynamic", normalized)
    return normalized.astype(np.float32)


def resolve_epoch_windows(num_epochs: int) -> tuple[slice, slice, slice]:
    """Return early/mid/late slices using 30% and 70% split on 1-based epoch definition."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    early_end = max(1, int(math.floor(0.3 * num_epochs)))
    mid_start = int(math.floor(0.3 * num_epochs)) + 1
    mid_end = max(mid_start, int(math.floor(0.7 * num_epochs)))
    late_start = int(math.floor(0.7 * num_epochs)) + 1

    early_slice = slice(0, early_end)
    mid_slice = slice(mid_start - 1, mid_end)
    late_slice = slice(late_start - 1, num_epochs)
    return early_slice, mid_slice, late_slice


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
