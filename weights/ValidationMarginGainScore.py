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
)


class ValidationMarginGainScore:
    """D: validation-view margin gain times local boundary coefficient."""

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

        norms = np.linalg.norm(static_features, axis=1, keepdims=True)
        g = static_features / np.where(norms > EPS, norms, 1.0)

        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing D by fold", unit="fold")):
            train_idx = fold.train_indices
            val_idx = fold.val_indices
            y_train = labels_all[train_idx]
            y_val = labels_all[val_idx]

            logits_val = np.nan_to_num(fold.val_logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            epoch_ids = np.arange(logits_val.shape[0])
            true_logits = logits_val[epoch_ids[:, None], np.arange(logits_val.shape[1])[None, :], y_val[None, :]]
            logits_wo_true = logits_val.copy()
            logits_wo_true[epoch_ids[:, None], np.arange(logits_val.shape[1])[None, :], y_val[None, :]] = -np.inf
            rival_max = np.max(logits_wo_true, axis=2)
            margin = true_logits - rival_max

            early_idx, _, late_idx = resolve_epoch_windows(logits_val.shape[0])
            m_early = np.mean(margin[early_idx], axis=0)
            m_late = np.mean(margin[late_idx], axis=0)
            delta_m = np.maximum(m_late - m_early, 0.0).astype(np.float32)

            raw = np.zeros(val_idx.shape[0], dtype=np.float32)
            sample_iter = tqdm(
                range(val_idx.shape[0]),
                desc=f"Computing D fold {f_idx} neighborhood",
                unit="sample",
                leave=False,
            )
            for local_i in sample_iter:
                yi = int(y_val[local_i])
                gi = g[val_idx[local_i]]

                same_mask = y_train == yi
                rival_mask = y_train != yi
                same_pool = train_idx[same_mask]
                rival_pool = train_idx[rival_mask]

                if same_pool.size == 0:
                    same_pool = train_idx
                if rival_pool.size == 0:
                    rival_pool = train_idx

                same_feats = g[same_pool]
                rival_feats = g[rival_pool]
                d_same_all = 1.0 - (same_feats @ gi)
                d_rival_all = 1.0 - (rival_feats @ gi)

                k_same = max(1, min(same_pool.size, int(math.ceil(self.k_ratio * same_pool.size))))
                k_rival = max(1, min(rival_pool.size, int(math.ceil(self.k_ratio * rival_pool.size))))

                d_same = float(np.mean(np.partition(d_same_all, k_same - 1)[:k_same]))
                d_rival = float(np.mean(np.partition(d_rival_all, k_rival - 1)[:k_rival]))
                b_i = d_same / (d_rival + EPS)
                raw[local_i] = float(np.nan_to_num(delta_m[local_i] * b_i, nan=0.0, posinf=0.0, neginf=0.0))

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
