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
class OOFSupportResult:
    """Result for G (OOFSupportScore), aligned to global sample indices."""

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class OOFSupportScore:
    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> OOFSupportResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg = np.full(num_samples, np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(folds):
            val_idx = fold.val_indices
            y_val = labels_all[val_idx]
            probs = softmax(fold.val_logits)
            r = true_class_probabilities(probs, y_val)

            _, _, late_slice = resolve_epoch_windows(fold.val_logits.shape[0])
            late_probs = r[late_slice]
            pred = np.argmax(probs[late_slice], axis=2)
            late_acc = np.mean((pred == y_val[None, :]).astype(np.float32), axis=0)
            late_prob = np.mean(late_probs, axis=0)
            raw = 0.5 * (late_acc + late_prob)

            norm1 = quantile_minmax_v2(raw.astype(np.float32))
            foldwise_norm1[f_idx, val_idx] = norm1
            agg[val_idx] = norm1

        if np.any(~np.isfinite(agg)):
            raise ValueError("Some samples were not assigned OOF support score in validation folds.")

        norm2 = quantile_minmax_v2(agg)
        return OOFSupportResult(agg=agg.astype(np.float32), norm2=norm2, foldwise_norm1=foldwise_norm1)
