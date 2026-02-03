"""PersistentDifficultyScore implementation for k-fold proxy logs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.score_utils import quantile_minmax, stable_sigmoid
from weights.TransferGainScore import get_labels_by_indices


class PersistentDifficultyScore:
    """Compute PersistentDifficultyScore (V) from k-fold validation logits."""

    def __init__(
        self,
        *,
        late_ratio: float = 0.5,
        tau_m: float = 0.10,
        eps: float = 1e-12,
        q_low: float = 0.01,
        q_high: float = 0.99,
        min_class_count: int = 20,
        verbose: bool = False,
    ) -> None:
        self.late_ratio = float(late_ratio)
        self.tau_m = float(tau_m)
        self.eps = float(eps)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.min_class_count = int(min_class_count)
        self.verbose = bool(verbose)

    @staticmethod
    def _load_meta(log_dir: Path) -> dict:
        meta_path = log_dir / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        meta_npz = log_dir / "meta.npz"
        if meta_npz.exists():
            data = np.load(meta_npz, allow_pickle=True)
            return {key: data[key].tolist() for key in data.files}
        return {}

    @staticmethod
    def _softmax(logits: np.ndarray, eps: float) -> np.ndarray:
        logits_max = logits.max(axis=2, keepdims=True)
        shifted = logits - logits_max
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=2, keepdims=True)
        return exp_shifted / (sum_exp + eps)

    def _compute_fold_scores(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        if val_logits.ndim != 3:
            raise ValueError("val_logits must have shape (epochs, num_samples, num_classes).")
        if val_labels.ndim != 1:
            raise ValueError("labels must have shape (num_samples,).")
        if val_logits.shape[2] != num_classes:
            raise ValueError("num_classes mismatch with logits.")
        if val_logits.shape[1] != val_labels.shape[0]:
            raise ValueError("val_logits/labels length mismatch.")

        num_epochs = val_logits.shape[0]
        if num_epochs == 0:
            raise ValueError("val_logits must include at least one epoch.")
        if not 0.0 < self.late_ratio <= 1.0:
            raise ValueError("late_ratio must be in (0, 1].")
        if self.tau_m <= 0:
            raise ValueError("tau_m must be positive.")

        start_idx = int(np.floor(num_epochs * (1.0 - self.late_ratio)))
        start_idx = max(0, min(start_idx, num_epochs - 1))

        probs = self._softmax(val_logits.astype(np.float64), self.eps)
        label_idx = val_labels.reshape(1, -1, 1)
        p_true = np.take_along_axis(probs, label_idx, axis=2).squeeze(2)
        probs_other = probs.copy()
        probs_other[:, np.arange(val_labels.size), val_labels] = -np.inf
        p_other_max = probs_other.max(axis=2)
        margin = p_true - p_other_max

        margin_late = margin[start_idx:]
        dm = stable_sigmoid((-margin_late / self.tau_m).astype(np.float64)).mean(axis=0)

        log_probs = np.log(probs + self.eps)
        entropy = -(probs * log_probs).sum(axis=2)
        entropy_norm = entropy / np.log(num_classes)
        dh = entropy_norm[start_idx:].mean(axis=0)

        return (dm + dh).astype(np.float32)

    def _normalize_by_class(
        self,
        values: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("q_low/q_high must satisfy 0 <= q_low < q_high <= 1.")
        if values.ndim != 1 or labels.ndim != 1:
            raise ValueError("values and labels must be 1D arrays.")
        if values.shape[0] != labels.shape[0]:
            raise ValueError("labels length must match values length.")

        global_norm = quantile_minmax(values, q_low=self.q_low, q_high=self.q_high)
        output = np.empty_like(values, dtype=np.float32)
        labels = labels.astype(np.int64)
        for cls in np.unique(labels):
            mask = labels == cls
            count = int(np.sum(mask))
            if count == 0:
                continue
            if count < self.min_class_count:
                output[mask] = global_norm[mask]
                continue
            class_vals = values[mask]
            lo = float(np.quantile(class_vals, self.q_low))
            hi = float(np.quantile(class_vals, self.q_high))
            if hi <= lo:
                output[mask] = global_norm[mask]
                continue
            clipped = np.clip(class_vals, lo, hi)
            output[mask] = ((clipped - lo) / (hi - lo + 1e-8)).astype(np.float32)
        output = np.clip(output, 0.0, 1.0)
        return output.astype(np.float32)

    def compute(
        self,
        cv_log_dir: str | Path,
        dataset,
        num_classes: int | None = None,
    ) -> dict:
        log_dir = Path(cv_log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f"cv_log_dir not found: {log_dir}")

        meta = self._load_meta(log_dir)
        num_samples = int(meta.get("num_samples", len(dataset)))
        if len(dataset) != num_samples:
            num_samples = len(dataset)
        if num_classes is None:
            num_classes = int(meta.get("num_classes", getattr(dataset, "num_classes", 0)))
        if num_classes <= 0:
            raise ValueError("num_classes must be provided or available in meta/dataset.")

        fold_paths = sorted(log_dir.glob("fold_*.npz"))
        if not fold_paths:
            raise FileNotFoundError(f"No fold_*.npz files found in {log_dir}")

        v_raw = np.full(num_samples, np.nan, dtype=np.float32)
        labels_full = np.full(num_samples, -1, dtype=np.int64)
        for fold_path in fold_paths:
            data = np.load(fold_path)
            val_indices = data["val_indices"].astype(np.int64)
            val_logits = data["val_logits"].astype(np.float32)

            y_val = get_labels_by_indices(dataset, val_indices)
            labels_full[val_indices] = y_val

            fold_scores = self._compute_fold_scores(val_logits, y_val, num_classes)
            if fold_scores.shape[0] != val_indices.shape[0]:
                raise ValueError("fold score length mismatch with val indices.")
            if np.any(~np.isnan(v_raw[val_indices])):
                raise ValueError("Duplicate val indices detected across folds.")
            v_raw[val_indices] = fold_scores

        if np.any(np.isnan(v_raw)):
            missing = np.where(np.isnan(v_raw))[0]
            raise RuntimeError(f"Missing scores for {missing.size} samples.")
        if np.any(labels_full < 0):
            missing_labels = np.where(labels_full < 0)[0]
            labels_full[missing_labels] = get_labels_by_indices(dataset, missing_labels)

        v_norm = self._normalize_by_class(v_raw.astype(np.float32), labels_full.astype(np.int64))

        result = {
            "score": v_norm.astype(np.float32),
            "name": "PersistentDifficultyScore",
            "meta": {
                "late_ratio": self.late_ratio,
                "tau_m": self.tau_m,
                "q_low": self.q_low,
                "q_high": self.q_high,
                "min_class_count": self.min_class_count,
                "num_samples": num_samples,
                "num_classes": num_classes,
                "seed": meta.get("seed"),
                "k_folds": meta.get("k_folds"),
            },
        }

        if self.verbose:
            print(
                "PersistentDifficultyScore scores min/mean/max: "
                f"{v_norm.min():.6f}, {v_norm.mean():.6f}, {v_norm.max():.6f}"
            )

        return result


__all__ = ["PersistentDifficultyScore"]
