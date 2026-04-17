from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    DynamicComponentResult,
    FoldLogData,
    aggregate_train_fold_component,
    compute_class_knn_mean_distance,
    confusion_distribution_wo_true,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
)


class ConfusionComplementarityScore:
    """C: Confusion Complementarity over training-view same-class confusion-change patterns."""

    def __init__(self, k_ratio: float = 0.05) -> None:
        self.k_ratio = float(k_ratio)

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        fold_iter = tqdm(folds, desc="Computing C by fold", unit="fold")
        for f_idx, fold in enumerate(fold_iter):
            train_idx = fold.train_indices
            y_train = labels_all[train_idx]

            probs = softmax(fold.train_logits)
            q = confusion_distribution_wo_true(probs, y_train)
            early_idx, mid_idx, _ = resolve_epoch_windows(fold.train_logits.shape[0])
            q_early = np.mean(q[early_idx], axis=0)
            q_mid = np.mean(q[mid_idx], axis=0)
            delta_q = (q_early - q_mid).astype(np.float32)

            raw = compute_class_knn_mean_distance(
                delta_q,
                y_train,
                k_ratio=self.k_ratio,
                progress_desc=f"Computing C fold {f_idx} class-KNN",
            )

            raw_foldwise[f_idx, train_idx] = raw
            fold_normalized[f_idx, train_idx] = quantile_minmax_dynamic(raw)

        aggregated = aggregate_train_fold_component(fold_normalized, folds)
        final_normalized = quantile_minmax_dynamic(aggregated)
        return DynamicComponentResult(
            raw_foldwise=raw_foldwise,
            fold_normalized=fold_normalized,
            aggregated=aggregated,
            final_normalized=final_normalized,
        )
