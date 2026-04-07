from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamic_v2_utils import (
    FoldLogData,
    confusion_distribution_wo_true,
    quantile_minmax_v2,
    resolve_epoch_windows,
    softmax,
    true_class_probabilities,
)


@dataclass
class OOFPatternGapResult:
    """Result for D, aligned to global sample indices."""

    agg: np.ndarray
    norm2: np.ndarray
    foldwise_norm1: np.ndarray


class OOFPatternGapScore:
    """Legacy file/class name retained; now computes validation learnable boundary value.

    D_raw = b(i,f) * max(delta_r(i,f), 0), where b is same/rival cosine-distance ratio
    in static feature space from training fold structure, and delta_r is OOF learnability gain.
    """

    def compute(
        self,
        folds: list[FoldLogData],
        labels_all: np.ndarray,
        static_features: np.ndarray,
    ) -> OOFPatternGapResult:
        num_samples = labels_all.shape[0]
        num_folds = len(folds)
        eps = 1e-8

        if static_features.ndim != 2 or static_features.shape[0] != num_samples:
            raise ValueError(
                f"static_features must be shape (num_samples, dim); got {static_features.shape}, num_samples={num_samples}."
            )

        norms = np.linalg.norm(static_features, axis=1, keepdims=True)
        safe_norms = np.where(norms > eps, norms, 1.0)
        g = (static_features / safe_norms).astype(np.float32)

        foldwise_norm1 = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        agg = np.full(num_samples, np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(folds):
            train_idx = fold.train_indices
            val_idx = fold.val_indices
            y_train = labels_all[train_idx]
            y_val = labels_all[val_idx]

            probs = softmax(fold.val_logits)
            r = true_class_probabilities(probs, y_val)
            q = confusion_distribution_wo_true(probs, y_val)

            early_slice, _, late_slice = resolve_epoch_windows(fold.val_logits.shape[0])
            early_mean = np.mean(r[early_slice], axis=0).astype(np.float32)
            late_mean = np.mean(r[late_slice], axis=0).astype(np.float32)
            delta_r = np.maximum(late_mean - early_mean, 0.0)

            qbar = np.mean(q[early_slice.stop :], axis=0).astype(np.float32)
            c_star = np.argmax(qbar, axis=1).astype(np.int64)

            raw = np.zeros(val_idx.shape[0], dtype=np.float32)
            for local_i, global_i in enumerate(val_idx):
                yi = int(y_val[local_i])
                rival = int(c_star[local_i])

                same_train_mask = y_train == yi
                rival_train_mask = y_train == rival

                same_feats = g[train_idx[same_train_mask]]
                rival_feats = g[train_idx[rival_train_mask]]

                if same_feats.shape[0] == 0:
                    same_feats = g[train_idx]
                if rival_feats.shape[0] == 0:
                    rival_feats = g[train_idx]

                if same_feats.shape[0] == 0 or rival_feats.shape[0] == 0:
                    raise ValueError(f"Fold {f_idx}: empty training features prevent D computation for sample {global_i}.")

                k_same = max(1, min(same_feats.shape[0], int(math.ceil(0.05 * same_feats.shape[0]))))
                k_rival = max(1, min(rival_feats.shape[0], int(math.ceil(0.05 * rival_feats.shape[0]))))

                g_i = g[global_i]
                d_same_all = 1.0 - (same_feats @ g_i)
                d_rival_all = 1.0 - (rival_feats @ g_i)

                d_same = float(np.mean(np.partition(d_same_all, k_same - 1)[:k_same]))
                d_rival = float(np.mean(np.partition(d_rival_all, k_rival - 1)[:k_rival]))

                b = d_same / (d_rival + eps)
                raw[local_i] = np.float32(b * delta_r[local_i])

            norm1 = quantile_minmax_v2(raw)
            foldwise_norm1[f_idx, val_idx] = norm1
            agg[val_idx] = norm1

        if np.any(~np.isfinite(agg)):
            missing = np.where(~np.isfinite(agg))[0]
            raise ValueError(f"Some samples missing D assignment in validation folds: {missing[:10]}.")

        norm2 = quantile_minmax_v2(agg)
        return OOFPatternGapResult(agg=agg.astype(np.float32), norm2=norm2, foldwise_norm1=foldwise_norm1)
