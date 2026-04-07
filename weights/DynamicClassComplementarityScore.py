from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    FoldLogData,
    compute_class_knn_mean_distance,
    confusion_distribution_wo_true,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
)


@dataclass
class DynamicClassComplementarityResult:
    """Result for C (training-view class-wise dynamic complementarity)."""

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class DynamicClassComplementarityScore:
    """C: mid-training confusion prototype + same-class 5% KNN Euclidean distance."""

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicClassComplementarityResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg_sum = np.zeros(num_samples, dtype=np.float64)
        agg_count = np.zeros(num_samples, dtype=np.int64)

        for f_idx, fold in enumerate(tqdm(folds, desc="Extracting C", unit="fold")):
            train_idx = fold.train_indices
            y_train = labels_all[train_idx]

            probs = softmax(fold.train_logits)
            q = confusion_distribution_wo_true(probs, y_train)

            num_epochs = fold.train_logits.shape[0]
            _, mid_slice, _ = resolve_epoch_windows(num_epochs)
            q_proto = np.mean(q[mid_slice], axis=0).astype(np.float32)

            c_raw = compute_class_knn_mean_distance(q_proto, y_train, k_ratio=0.05)
            norm1 = quantile_minmax_dynamic(c_raw)

            foldwise_norm1[f_idx, train_idx] = norm1
            agg_sum[train_idx] += norm1.astype(np.float64)
            agg_count[train_idx] += 1

        if np.any(agg_count <= 0):
            raise ValueError("Some samples never appeared in train folds when computing C.")

        agg = (agg_sum / agg_count).astype(np.float32)
        norm2 = quantile_minmax_dynamic(agg)
        return DynamicClassComplementarityResult(agg=agg, norm2=norm2, foldwise_norm1=foldwise_norm1)
