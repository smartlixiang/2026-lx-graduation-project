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
    """
    E: validation-view persistent difficulty over mid+late.

    Persistent Difficulty = validation-view persistent difficulty over mid+late,
    computed by epoch-wise standardized margin difficulty and entropy difficulty.
    """

    def __init__(self, tau_e: float = 1.0) -> None:
        if tau_e <= 0:
            raise ValueError("tau_e must be positive.")
        self.tau_e = float(tau_e)

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        eps = 1e-8

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing E by fold", unit="fold")):
            val_idx = fold.val_indices
            y_val = labels_all[val_idx]

            logits_val = np.nan_to_num(fold.val_logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            probs_val = softmax(logits_val.astype(np.float32)).astype(np.float64)

            _, mid_idx, late_idx = resolve_epoch_windows(logits_val.shape[0])
            hard_idx = np.concatenate([mid_idx, late_idx])

            # margin difficulty: r_margin(i,t) = -m_val(i,t)
            num_epochs, num_val, num_classes = logits_val.shape
            epoch_ids = np.arange(num_epochs)[:, None]
            sample_ids = np.arange(num_val)[None, :]
            true_logits = logits_val[epoch_ids, sample_ids, y_val[None, :]]

            masked_logits = logits_val.copy()
            masked_logits[epoch_ids, sample_ids, y_val[None, :]] = -np.inf
            max_non_true_logits = np.max(masked_logits, axis=2)
            margin_val = true_logits - max_non_true_logits
            r_margin = -margin_val

            # entropy difficulty: r_entropy(i,t) = H_val(i,t) / log(C)
            entropy = -np.sum(probs_val * np.log(np.clip(probs_val, eps, 1.0)), axis=2)
            entropy_norm = np.log(max(num_classes, 2))
            r_entropy = entropy / entropy_norm

            # epoch-wise z-score across validation samples
            mean_t_margin = np.mean(r_margin, axis=1, keepdims=True)
            std_t_margin = np.std(r_margin, axis=1, keepdims=True)
            safe_std_margin = np.where(std_t_margin > eps, std_t_margin, 1.0)
            z_margin = (r_margin - mean_t_margin) / safe_std_margin

            mean_t_entropy = np.mean(r_entropy, axis=1, keepdims=True)
            std_t_entropy = np.std(r_entropy, axis=1, keepdims=True)
            safe_std_entropy = np.where(std_t_entropy > eps, std_t_entropy, 1.0)
            z_entropy = (r_entropy - mean_t_entropy) / safe_std_entropy

            # mid+late persistent averages
            e_margin = np.mean(z_margin[hard_idx], axis=0)
            e_entropy = np.mean(z_entropy[hard_idx], axis=0)
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
