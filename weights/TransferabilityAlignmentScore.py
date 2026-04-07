from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    EPS,
    DynamicComponentResult,
    FoldLogData,
    aggregate_train_fold_component,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
    softmax,
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return 0.0
    xx = x[finite]
    yy = y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


class TransferabilityAlignmentScore:
    """D: alignment between train margin advances and same-class validation improvements."""

    def __init__(self, tau_d: float = 1.0) -> None:
        if tau_d <= 0:
            raise ValueError("tau_d must be positive.")
        self.tau_d = float(tau_d)

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        num_samples = labels_all.shape[0]
        raw_foldwise = np.full((len(folds), num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((len(folds), num_samples), np.nan, dtype=np.float32)

        for f_idx, fold in enumerate(tqdm(folds, desc="Computing D by fold", unit="fold")):
            train_idx = fold.train_indices
            val_idx = fold.val_indices
            y_train = labels_all[train_idx]
            y_val = labels_all[val_idx]

            logits_train = np.nan_to_num(fold.train_logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
            logits_val = np.nan_to_num(fold.val_logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)

            epoch_ids = np.arange(logits_train.shape[0])[:, None]
            sample_ids_train = np.arange(logits_train.shape[1])[None, :]
            true_logits_train = logits_train[epoch_ids, sample_ids_train, y_train[None, :]]
            masked_train = logits_train.copy()
            masked_train[epoch_ids, sample_ids_train, y_train[None, :]] = -np.inf
            rival_logits_train = np.max(masked_train, axis=2)
            margin_train = true_logits_train - rival_logits_train

            delta_m = margin_train[1:] - margin_train[:-1]
            g_train = np.log1p(np.exp(delta_m / self.tau_d)) * self.tau_d

            probs_val = softmax(logits_val.astype(np.float32))
            true_prob_val = probs_val[np.arange(probs_val.shape[0])[:, None], np.arange(probs_val.shape[1])[None, :], y_val[None, :]]
            true_prob_val = np.clip(true_prob_val.astype(np.float64), EPS, 1.0)
            loss_val = -np.log(true_prob_val)

            num_epochs = logits_train.shape[0]
            _, mid_idx, late_idx = resolve_epoch_windows(num_epochs)
            selected_epochs = np.concatenate([mid_idx, late_idx])
            selected_epochs = selected_epochs[selected_epochs >= 1]
            if selected_epochs.size == 0:
                selected_epochs = np.array([num_epochs - 1], dtype=np.int64)
            delta_idx = selected_epochs - 1

            class_improve: dict[int, np.ndarray] = {}
            for cls in np.unique(y_train):
                cls_mask = y_val == int(cls)
                if not np.any(cls_mask):
                    class_improve[int(cls)] = np.zeros(num_epochs - 1, dtype=np.float64)
                    continue
                class_loss = np.mean(loss_val[:, cls_mask], axis=1)
                v_val = np.maximum(class_loss[:-1] - class_loss[1:], 0.0)
                class_improve[int(cls)] = v_val

            raw = np.zeros(train_idx.shape[0], dtype=np.float32)
            for local_i in range(train_idx.shape[0]):
                yi = int(y_train[local_i])
                corr = _safe_corr(g_train[delta_idx, local_i], class_improve[yi][delta_idx])
                corr01 = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)
                raw[local_i] = np.float32(corr01)

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
