from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamic_v2_utils import (
    FoldLogData,
    compute_class_knn_mean_distance,
    confusion_distribution_wo_true,
    quantile_minmax_v2,
    resolve_epoch_windows,
    softmax,
    true_class_probabilities,
)


@dataclass
class OOFPatternGapResult:
    """Result for D (OOFPatternGapScore), aligned to global sample indices."""

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class OOFPatternGapScore:
    """OOF pattern gap score D: late unsupported + class-specific pattern uniqueness.

    For each validation sample in its OOF fold:
      - late_prob: mean true-class probability over late_slice.
      - pattern_dist: class-wise KNN mean Euclidean distance of late confusion prototype.
      - raw: (1 - late_prob) * pattern_dist.
    """

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> OOFPatternGapResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg = np.full(num_samples, np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(folds):
            val_idx = fold.val_indices
            y_val = labels_all[val_idx]

            probs = softmax(fold.val_logits)
            r = true_class_probabilities(probs, y_val)

            # late_slice corresponds to [floor(0.7*E)+1, E] in 1-based epoch indexing.
            _, _, late_slice = resolve_epoch_windows(fold.val_logits.shape[0])
            late_prob = np.mean(r[late_slice], axis=0).astype(np.float32)

            q = confusion_distribution_wo_true(probs, y_val)
            q_val = np.mean(q[late_slice], axis=0).astype(np.float32)

            pattern_dist = compute_class_knn_mean_distance(q_val, y_val, k_ratio=0.05)
            raw = (1.0 - late_prob) * pattern_dist
            norm1 = quantile_minmax_v2(raw.astype(np.float32))

            foldwise_norm1[f_idx, val_idx] = norm1
            agg[val_idx] = norm1

        if np.any(~np.isfinite(agg)):
            raise ValueError("Some samples were not assigned OOF pattern gap score in validation folds.")

        norm2 = quantile_minmax_v2(agg)
        return OOFPatternGapResult(agg=agg.astype(np.float32), norm2=norm2, foldwise_norm1=foldwise_norm1)
