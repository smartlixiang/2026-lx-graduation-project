from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    FoldLogData,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
    true_class_probabilities,
)


@dataclass
class EarlyLearnabilityResult:
    """Result for A (training-view absorbable stability).

    A_raw_i(f) = z(early true-class probability mean) - z(late true-class probability variance)
    Then fold-internal quantile min-max -> A_norm1, cross-fold train mean -> A_agg,
    and global quantile min-max -> A_norm2.
    """

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class EarlyLearnabilityScore:
    """A: absorbable stability from training dynamics."""

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> EarlyLearnabilityResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg_sum = np.zeros(num_samples, dtype=np.float64)
        agg_count = np.zeros(num_samples, dtype=np.int64)
        eps = 1e-8

        for f_idx, fold in enumerate(tqdm(folds, desc="Extracting A", unit="fold")):
            train_idx = fold.train_indices
            y_train = labels_all[train_idx]

            probs = softmax(fold.train_logits)
            r = true_class_probabilities(probs, y_train)

            num_epochs = fold.train_logits.shape[0]
            early_slice, _, late_slice = resolve_epoch_windows(num_epochs)

            early_mean = np.mean(r[early_slice], axis=0).astype(np.float64)
            late_var = np.var(r[late_slice], axis=0).astype(np.float64)

            z_early_mean = (early_mean - early_mean.mean()) / (early_mean.std() + eps)
            z_late_var = (late_var - late_var.mean()) / (late_var.std() + eps)
            a_raw = (z_early_mean - z_late_var).astype(np.float32)

            norm1 = quantile_minmax_dynamic(a_raw)
            foldwise_norm1[f_idx, train_idx] = norm1
            agg_sum[train_idx] += norm1.astype(np.float64)
            agg_count[train_idx] += 1

        if np.any(agg_count <= 0):
            raise ValueError("Some samples never appeared in train folds when computing A.")

        agg = (agg_sum / agg_count).astype(np.float32)
        norm2 = quantile_minmax_dynamic(agg)
        return EarlyLearnabilityResult(agg=agg, norm2=norm2, foldwise_norm1=foldwise_norm1)
