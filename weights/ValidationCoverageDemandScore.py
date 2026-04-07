from __future__ import annotations

import math

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    DynamicComponentResult,
    EPS,
    FoldLogData,
    K_RATIO_DEFAULT,
    aggregate_val_fold_component,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
    true_class_probabilities,
)


class ValidationCoverageDemandScore:
    """E: validation-view same-class training-fold coverage demand."""

    def __init__(self, k_ratio: float = K_RATIO_DEFAULT) -> None:
        self.k_ratio = float(k_ratio)

    def compute(
        self,
        folds: list[FoldLogData],
        labels_all: np.ndarray,
        static_features: np.ndarray,
    ) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        if static_features.ndim != 2 or static_features.shape[0] != num_samples:
            raise ValueError(f"static_features shape mismatch: {static_features.shape}, expected ({num_samples}, dim).")

        g = np.nan_to_num(static_features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing E by fold", unit="fold")):
            train_idx = fold.train_indices
            val_idx = fold.val_indices
            y_train = labels_all[train_idx]
            y_val = labels_all[val_idx]

            probs_val = softmax(fold.val_logits)
            r_val = true_class_probabilities(probs_val, y_val)
            early_idx, _, late_idx = resolve_epoch_windows(fold.val_logits.shape[0])
            p_early = np.mean(r_val[early_idx], axis=0)
            p_late = np.mean(r_val[late_idx], axis=0)
            gain_val = np.maximum(p_late - p_early, 0.0)

            raw = np.zeros(val_idx.shape[0], dtype=np.float32)
            sample_iter = tqdm(
                range(val_idx.shape[0]),
                desc=f"Computing E fold {f_idx} same-class coverage",
                unit="sample",
                leave=False,
            )
            for local_i in sample_iter:
                yi = int(y_val[local_i])
                gi = g[val_idx[local_i]]
                same_pool = train_idx[y_train == yi]
                if same_pool.size == 0:
                    same_pool = train_idx
                if same_pool.size == 0:
                    raw[local_i] = 0.0
                    continue

                same_feats = g[same_pool]
                dist_all = np.linalg.norm(same_feats - gi[None, :], axis=1)
                k_same = max(1, min(same_pool.size, int(math.ceil(self.k_ratio * same_pool.size))))
                s_i = float(np.mean(np.partition(dist_all, k_same - 1)[:k_same]))
                raw[local_i] = float(np.nan_to_num(s_i * gain_val[local_i], nan=0.0, posinf=0.0, neginf=0.0))

            raw_foldwise[f_idx, val_idx] = raw
            fold_normalized[f_idx, val_idx] = quantile_minmax_dynamic(raw)

        aggregated = aggregate_val_fold_component(fold_normalized, folds)
        final_normalized = quantile_minmax_dynamic(aggregated)
        return DynamicComponentResult(
            raw_foldwise=raw_foldwise,
            fold_normalized=fold_normalized,
            aggregated=aggregated,
            final_normalized=final_normalized,
        )
