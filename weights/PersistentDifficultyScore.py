from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    DynamicComponentResult,
    FoldLogData,
    aggregate_val_fold_component,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
)


class PersistentDifficultyScore:
    """E: persistent validation difficulty over mid+late after discounting early baseline."""

    def __init__(self, tau_e: float = 1.0) -> None:
        if tau_e <= 0:
            raise ValueError("tau_e must be positive.")
        self.tau_e = float(tau_e)

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing E by fold", unit="fold")):
            val_idx = fold.val_indices
            y_val = labels_all[val_idx]

            logits_val = np.nan_to_num(fold.val_logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            probs_val = softmax(logits_val.astype(np.float32)).astype(np.float64)

            epoch_ids = np.arange(probs_val.shape[0])[:, None]
            sample_ids = np.arange(probs_val.shape[1])[None, :]
            true_class_probs = probs_val[epoch_ids, sample_ids, y_val[None, :]]

            early_idx, mid_idx, late_idx = resolve_epoch_windows(logits_val.shape[0])
            hard_idx = np.concatenate([mid_idx, late_idx])

            early_true_prob_mean = np.mean(true_class_probs[early_idx], axis=0)
            hard_true_prob_mean = np.mean(true_class_probs[hard_idx], axis=0)

            early_difficulty_baseline = 1.0 - early_true_prob_mean
            hard_difficulty_residual = 1.0 - hard_true_prob_mean

            raw = (hard_difficulty_residual - 0.5 * early_difficulty_baseline).astype(np.float32)
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
