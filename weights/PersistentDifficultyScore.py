from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    EPS,
    DynamicComponentResult,
    FoldLogData,
    aggregate_val_fold_component,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
)


class PersistentDifficultyScore:
    """E: late-stage persistent difficulty on OOF validation trajectories."""

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

            epoch_ids = np.arange(logits_val.shape[0])[:, None]
            sample_ids = np.arange(logits_val.shape[1])[None, :]
            true_logits = logits_val[epoch_ids, sample_ids, y_val[None, :]]
            masked = logits_val.copy()
            masked[epoch_ids, sample_ids, y_val[None, :]] = -np.inf
            rival_logits = np.max(masked, axis=2)
            margin = true_logits - rival_logits

            _, _, late_idx = resolve_epoch_windows(logits_val.shape[0])

            margin_term = 1.0 / (1.0 + np.exp(margin[late_idx] / self.tau_e))
            e_margin = np.mean(margin_term, axis=0)

            entropy = -np.sum(probs_val * np.log(np.clip(probs_val, EPS, 1.0)), axis=2)
            num_classes = probs_val.shape[2]
            entropy_norm = entropy / np.log(max(num_classes, 2))
            e_entropy = np.mean(entropy_norm[late_idx], axis=0)

            raw = (e_margin + e_entropy).astype(np.float32)
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
