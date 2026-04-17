from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    DynamicComponentResult,
    FoldLogData,
    aggregate_train_fold_component,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    safe_standardize,
    softmax,
    true_class_probabilities,
)


class AbsorptionGainScore:
    """A: Absorption Gain with late-stage fluctuation penalty."""

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing A by fold", unit="fold")):
            train_idx = fold.train_indices
            y_train = labels_all[train_idx]

            probs = softmax(fold.train_logits)
            r = true_class_probabilities(probs, y_train)
            early_idx, mid_idx, late_idx = resolve_epoch_windows(fold.train_logits.shape[0])

            early_mean = np.mean(r[early_idx], axis=0)
            mid_mean = np.mean(r[mid_idx], axis=0)
            late_var = np.var(r[late_idx], axis=0)

            gain_z = safe_standardize(mid_mean - early_mean)
            var_z = safe_standardize(late_var)
            raw = gain_z - 0.5 * var_z

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
