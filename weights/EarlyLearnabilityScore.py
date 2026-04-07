from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamic_v2_utils import (
    FoldLogData,
    quantile_minmax_v2,
    resolve_epoch_windows,
    softmax,
    true_class_probabilities,
)


@dataclass
class EarlyLearnabilityResult:
    """Result for A (Absorbable Stability in training view).

    Inputs:
      - folds: CV fold logs with train_indices/train_logits.
      - labels_all: labels aligned to global sample indices [0, num_samples).

    Returns (all aligned to global indices):
      - agg: mean of fold-internal norm1 values over folds where sample is in train set.
      - norm2: global second quantile-minmax normalization of agg.
      - foldwise_norm1: matrix (num_folds, num_samples), NaN where sample not in train for a fold.
    """

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class EarlyLearnabilityScore:
    """A: training-view absorbable stability = z(early true prob mean) - z(late true prob variance)."""

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> EarlyLearnabilityResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg_sum = np.zeros(num_samples, dtype=np.float64)
        agg_count = np.zeros(num_samples, dtype=np.int64)
        eps = 1e-8

        for f_idx, fold in enumerate(folds):
            train_idx = fold.train_indices
            y_train = labels_all[train_idx]

            probs = softmax(fold.train_logits)
            r = true_class_probabilities(probs, y_train)

            num_epochs = fold.train_logits.shape[0]
            early_slice, _, late_slice = resolve_epoch_windows(num_epochs)

            a1 = np.mean(r[early_slice], axis=0).astype(np.float64)
            a2 = np.var(r[late_slice], axis=0).astype(np.float64)

            z1 = (a1 - a1.mean()) / (a1.std() + eps)
            z2 = (a2 - a2.mean()) / (a2.std() + eps)
            raw = (z1 - z2).astype(np.float32)

            norm1 = quantile_minmax_v2(raw)
            foldwise_norm1[f_idx, train_idx] = norm1
            agg_sum[train_idx] += norm1.astype(np.float64)
            agg_count[train_idx] += 1

        if np.any(agg_count <= 0):
            raise ValueError("Some samples never appeared in train folds when computing A.")

        agg = (agg_sum / agg_count).astype(np.float32)
        norm2 = quantile_minmax_v2(agg)
        return EarlyLearnabilityResult(agg=agg, norm2=norm2, foldwise_norm1=foldwise_norm1)
