from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    DynamicComponentResult,
    FoldLogData,
    aggregate_train_fold_component,
    standard_zscore_dynamic,
    resolve_epoch_windows,
    safe_standardize,
    softmax,
    true_class_probabilities,
)


class AbsorptionGainScore:
    """A: Absorption Gain with early boundary information and late-stage stability.

    This component keeps the original role of A as the training-view absorbability
    signal, but it no longer only rewards the increase from early to mid training.
    For ordinary clean samples, useful training value may appear in three related
    forms:
      1. early boundary information: the sample is informative before it is saturated;
      2. absorption gain: the sample is progressively absorbed by the proxy model;
      3. late stability: the learned state does not fluctuate heavily near the end.

    Each sub-signal is standardized inside the fold and then combined with equal
    weights. No extra manually tuned coefficient is introduced.
    """

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

            # Early boundary information. This avoids simply favoring either
            # already-saturated easy samples or persistently unlearnable samples.
            boundary_info = np.mean(r[early_idx] * (1.0 - r[early_idx]), axis=0)

            # Original absorption-gain signal: whether the proxy model increases
            # its true-class belief from the early stage to the middle stage.
            gain = mid_mean - early_mean

            # Late-stage stability. Lower late variance indicates a more stable
            # learned state, so the sign is reversed before standardization.
            stability = -late_var

            raw = (
                safe_standardize(boundary_info)
                + safe_standardize(gain)
                + safe_standardize(stability)
            ).astype(np.float32)

            raw_foldwise[f_idx, train_idx] = raw
            fold_normalized[f_idx, train_idx] = standard_zscore_dynamic(raw)

        aggregated = aggregate_train_fold_component(fold_normalized, folds)
        final_normalized = standard_zscore_dynamic(aggregated)
        return DynamicComponentResult(
            raw_foldwise=raw_foldwise,
            fold_normalized=fold_normalized,
            aggregated=aggregated,
            final_normalized=final_normalized,
        )
