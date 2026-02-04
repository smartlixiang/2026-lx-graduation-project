"""Shared utilities for scoring normalization and robust statistics."""

# Key interfaces:
# - stable_sigmoid(values): numerically stable sigmoid for numpy arrays.
# - robust_z_by_class(values, labels): class-wise median/MAD standardization.
# - quantile_minmax_by_class(values, labels): class-wise quantile min-max to [0,1].
# - quantile_minmax(values): global quantile min-max to [0,1].
from __future__ import annotations

from typing import Iterable

import numpy as np


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    negative = ~positive
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_vals = np.exp(values[negative])
    result[negative] = exp_vals / (1.0 + exp_vals)
    return result.astype(np.float32)


def resolve_window_length(num_epochs: int, ratio: float = 0.2, min_epochs: int = 5) -> int:
    """Resolve a robust window length for early/late segments."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    if min_epochs <= 0:
        raise ValueError("min_epochs must be positive.")
    window = max(min_epochs, int(ratio * num_epochs))
    return min(num_epochs, window)


def resolve_early_late_slices(
    num_epochs: int,
    ratio: float = 0.5,
    min_epochs: int = 5,
    skip_first: bool = True,
) -> tuple[slice, slice, int]:
    """Resolve aligned early/late slices using a shared window length."""
    window = resolve_window_length(num_epochs, ratio=ratio, min_epochs=min_epochs)
    if skip_first and num_epochs > 1:
        window = min(window, num_epochs - 1)
        early_start = 1
    else:
        early_start = 0
    early_end = min(num_epochs, early_start + window)
    late_start = max(0, num_epochs - window)
    early_slice = slice(early_start, early_end)
    late_slice = slice(late_start, num_epochs)
    return early_slice, late_slice, window


def _iter_classes(labels: np.ndarray) -> Iterable[int]:
    for cls in np.unique(labels.astype(np.int64)):
        yield int(cls)


def robust_z_by_class(values: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute class-wise robust z-score using median/MAD."""
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if values.shape[0] != labels.shape[0]:
        raise ValueError("labels length must match values length.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    output = np.zeros_like(values, dtype=np.float32)
    labels = labels.astype(np.int64)
    for cls in _iter_classes(labels):
        mask = labels == cls
        if not np.any(mask):
            continue
        class_vals = values[mask]
        median = np.median(class_vals)
        mad = np.median(np.abs(class_vals - median))
        output[mask] = ((class_vals - median) / (mad + eps)).astype(np.float32)
    return output


def quantile_minmax_by_class(
    values: np.ndarray,
    labels: np.ndarray,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-8,
    fallback_value: float = 0.0,
) -> np.ndarray:
    """Apply class-wise quantile min-max normalization to [0,1]."""
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array.")
    if values.shape[0] != labels.shape[0]:
        raise ValueError("labels length must match values length.")
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("q_low/q_high must satisfy 0 <= q_low < q_high <= 1.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    output = np.zeros_like(values, dtype=np.float32)
    labels = labels.astype(np.int64)
    for cls in _iter_classes(labels):
        mask = labels == cls
        if not np.any(mask):
            continue
        class_vals = values[mask]
        lo = float(np.quantile(class_vals, q_low))
        hi = float(np.quantile(class_vals, q_high))
        if hi <= lo:
            output[mask] = float(fallback_value)
            continue
        clipped = np.clip(class_vals, lo, hi)
        output[mask] = ((clipped - lo) / (hi - lo + eps)).astype(np.float32)
    return output


def quantile_minmax(
    values: np.ndarray,
    q_low: float = 0.01,
    q_high: float = 0.99,
    eps: float = 1e-8,
    fallback_value: float = 0.0,
) -> np.ndarray:
    """Apply global quantile min-max normalization to [0,1]."""
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("q_low/q_high must satisfy 0 <= q_low < q_high <= 1.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    lo = float(np.quantile(values, q_low))
    hi = float(np.quantile(values, q_high))
    if hi <= lo:
        return np.full_like(values, float(fallback_value), dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    return ((clipped - lo) / (hi - lo + eps)).astype(np.float32)
