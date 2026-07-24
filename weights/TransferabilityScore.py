from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .dynamic_utils import (
    EPS,
    DynamicComponentResult,
    FoldLogData,
    aggregate_train_fold_component,
    resolve_epoch_windows,
    standard_zscore_dynamic,
)


def _true_label_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Numerically stable per-epoch, per-sample true-label cross entropy."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, samples, classes).")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[1]:
        raise ValueError("labels shape mismatch for validation logits.")
    safe_logits = np.nan_to_num(logits.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    max_logits = np.max(safe_logits, axis=2, keepdims=True)
    logsumexp = np.squeeze(max_logits, axis=2) + np.log(np.sum(np.exp(safe_logits - max_logits), axis=2))
    epoch_idx = np.arange(safe_logits.shape[0])[:, None]
    sample_idx = np.arange(safe_logits.shape[1])[None, :]
    true_logits = safe_logits[epoch_idx, sample_idx, labels[None, :]]
    loss = logsumexp - true_logits
    return np.nan_to_num(loss, nan=0.0, posinf=50.0, neginf=-50.0)


class TransferabilityScore:
    """T: signed leave-one-out transferability from untruncated validation loss improvements."""

    def compute(self, folds: list[FoldLogData], labels_all: np.ndarray) -> DynamicComponentResult:
        labels_all = np.asarray(labels_all, dtype=np.int64)
        num_samples = int(labels_all.shape[0])
        num_folds = len(folds)
        if num_folds == 0:
            raise ValueError("folds must not be empty.")

        raw_foldwise = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        fold_normalized = np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        in_sum = np.zeros(num_samples, dtype=np.float64)
        in_count = np.zeros(num_samples, dtype=np.int64)
        out_values = np.full(num_samples, np.nan, dtype=np.float64)
        out_count = np.zeros(num_samples, dtype=np.int64)

        for fold in tqdm(folds, desc="Computing T utilities", unit="fold"):
            train_idx = np.asarray(fold.train_indices, dtype=np.int64)
            val_idx = np.asarray(fold.val_indices, dtype=np.int64)
            y_train = labels_all[train_idx]
            y_val = labels_all[val_idx]
            val_logits = np.asarray(fold.val_logits)
            num_epochs = int(val_logits.shape[0])
            if num_epochs <= 0:
                raise ValueError("fold val_logits must contain at least one epoch.")

            _, mid_idx, late_idx = resolve_epoch_windows(num_epochs)
            selected_epochs = np.unique(np.concatenate([mid_idx, late_idx]).astype(np.int64))
            selected_epochs = selected_epochs[selected_epochs >= 1]
            if selected_epochs.size == 0:
                selected_epochs = np.array([num_epochs - 1], dtype=np.int64)
            selected_epochs = selected_epochs[selected_epochs < num_epochs]
            if selected_epochs.size == 0:
                selected_epochs = np.array([max(num_epochs - 1, 0)], dtype=np.int64)
            previous_epochs = selected_epochs - 1

            loss_val = _true_label_cross_entropy(val_logits, y_val)
            q = np.mean(loss_val[previous_epochs] - loss_val[selected_epochs], axis=0)
            q = np.nan_to_num(q.astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)

            class_utility: dict[int, float] = {}
            class_sum: dict[int, float] = {}
            class_count: dict[int, int] = {}
            for cls in np.unique(labels_all):
                cls = int(cls)
                mask = y_val == cls
                count = int(np.sum(mask))
                if count == 0:
                    class_utility[cls] = 0.0
                    class_sum[cls] = 0.0
                    class_count[cls] = 0
                else:
                    total = float(np.sum(q[mask]))
                    class_sum[cls] = total
                    class_count[cls] = count
                    class_utility[cls] = total / count

            for cls in np.unique(y_train):
                cls = int(cls)
                idx = train_idx[y_train == cls]
                in_sum[idx] += class_utility.get(cls, 0.0)
                in_count[idx] += 1

            for local_i, global_i in enumerate(val_idx):
                cls = int(y_val[local_i])
                count = class_count.get(cls, 0)
                if count <= 1:
                    out_values[global_i] = class_utility.get(cls, 0.0)
                else:
                    out_values[global_i] = (class_sum[cls] - float(q[local_i])) / float(count - 1)
                out_count[global_i] += 1

        if not np.all(in_count > 0):
            raise ValueError("each sample must appear in at least one training fold.")
        if not np.all(out_count == 1):
            raise ValueError("each sample must appear in exactly one validation fold.")

        u_in = in_sum / in_count
        u_out = np.nan_to_num(out_values, nan=0.0, posinf=1.0, neginf=-1.0)
        t_raw = (u_in - u_out) / (np.abs(u_in) + np.abs(u_out) + EPS)
        t_raw = np.nan_to_num(t_raw, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        for f_idx, fold in enumerate(folds):
            train_idx = np.asarray(fold.train_indices, dtype=np.int64)
            raw_foldwise[f_idx, train_idx] = t_raw[train_idx]
            fold_normalized[f_idx, train_idx] = standard_zscore_dynamic(t_raw[train_idx])

        aggregated = aggregate_train_fold_component(fold_normalized, folds)
        final_normalized = standard_zscore_dynamic(aggregated)
        return DynamicComponentResult(raw_foldwise=raw_foldwise, fold_normalized=fold_normalized, aggregated=aggregated, final_normalized=final_normalized)
