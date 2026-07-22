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
    """A: Absorption Gain with early boundary information and loss-variance stability.

    This component keeps the original role of A as the training-view absorbability
    signal, but it no longer only rewards the increase from early to mid training.
    For ordinary clean samples, useful training value may appear in three related
    forms:
      1. early boundary information: the sample is informative before it is saturated;
      2. absorption gain: the sample is progressively absorbed by the proxy model;
      3. loss-variance stability: compare late and middle true-label CE volatility.

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

            # Early boundary information. This avoids simply favoring either
            # already-saturated easy samples or persistently unlearnable samples.
            boundary_info = np.mean(r[early_idx] * (1.0 - r[early_idx]), axis=0)

            # Double-ended support weighted gain: a mid-stage increase is trusted
            # most when both early and middle true-label probabilities are supported.
            support = np.sqrt(np.clip(early_mean * mid_mean, 0.0, 1.0))
            gain = support * (mid_mean - early_mean)

            logits = fold.train_logits.astype(np.float64, copy=False)
            max_logits = np.max(logits, axis=2, keepdims=True)
            logsumexp = np.squeeze(max_logits, axis=2) + np.log(np.sum(np.exp(logits - max_logits), axis=2))
            true_logits = np.take_along_axis(logits, y_train[None, :, None], axis=2).squeeze(axis=2)
            loss = logsumexp - true_logits
            middle_loss_var = np.var(loss[mid_idx], axis=0)
            late_loss_var = np.var(loss[late_idx], axis=0)
            stability = late_loss_var - middle_loss_var

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
