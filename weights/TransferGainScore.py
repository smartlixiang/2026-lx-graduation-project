"""TransferGainScore implementation for k-fold proxy logs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def get_labels_by_indices(dataset, indices: np.ndarray) -> np.ndarray:
    """Fetch labels by global indices, preferring dataset.targets/labels."""
    indices = np.asarray(indices, dtype=np.int64)
    num_samples = len(dataset)
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if len(values) == num_samples:
                return np.asarray(values, dtype=np.int64)[indices]
    labels = np.empty(indices.shape[0], dtype=np.int64)
    for pos, idx in enumerate(indices.tolist()):
        _, label = dataset[idx]
        if hasattr(label, "item"):
            label = label.item()
        labels[pos] = int(label)
    return labels


class TransferGainScore:
    """Compute TransferGainScore (T) from k-fold proxy training logs."""

    def __init__(
        self,
        *,
        tau_p_mode: str = "percentile",
        tau_p: float = 0.10,
        tau_p_percentile: float = 90.0,
        tau_p_min: float = 1e-3,
        eps: float = 1e-12,
        agg: str = "median",
        cache_dir: str | Path = "weights/cache",
        verbose: bool = False,
    ) -> None:
        self.tau_p_mode = str(tau_p_mode)
        self.tau_p = float(tau_p)
        self.tau_p_percentile = float(tau_p_percentile)
        self.tau_p_min = float(tau_p_min)
        self.eps = float(eps)
        self.agg = agg
        self.cache_dir = Path(cache_dir)
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

    @staticmethod
    def _softplus(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)

    def _resolve_tau_p(self, dp: np.ndarray) -> float:
        if self.tau_p_mode == "fixed":
            return max(self.tau_p_min, self.tau_p)
        if self.tau_p_mode == "percentile":
            values = np.abs(dp).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return self.tau_p_min
            tau = float(np.percentile(values, self.tau_p_percentile))
            return max(self.tau_p_min, tau)
        raise ValueError(f"Unsupported tau_p_mode: {self.tau_p_mode}")

    @staticmethod
    def _iter_classes(num_classes: int) -> Iterable[int]:
        return range(int(num_classes))

    def _compute_fold_scores(
        self,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        if train_logits.ndim != 3 or val_logits.ndim != 3:
            raise ValueError("logits must have shape (epochs, num_samples, num_classes).")
        if train_labels.ndim != 1 or val_labels.ndim != 1:
            raise ValueError("labels must have shape (num_samples,).")
        if train_logits.shape[2] != num_classes or val_logits.shape[2] != num_classes:
            raise ValueError("num_classes mismatch with logits.")
        if train_logits.shape[0] != val_logits.shape[0]:
            raise ValueError("train/val logits must share the same epoch count.")

        num_epochs = train_logits.shape[0]
        if num_epochs == 0:
            raise ValueError("logits must include at least one epoch.")

        val_probs = self._softmax(val_logits.astype(np.float64), self.eps)
        val_label_idx = val_labels.reshape(1, -1, 1)
        val_p_true = np.take_along_axis(val_probs, val_label_idx, axis=2).squeeze(2)
        val_loss = -np.log(np.maximum(val_p_true, self.eps))
        val_loss = np.log1p(val_loss)

        v_curve = np.zeros((num_epochs, num_classes), dtype=np.float64)
        for cls in self._iter_classes(num_classes):
            mask = val_labels == cls
            if not np.any(mask):
                continue
            v_curve[:, cls] = val_loss[:, mask].mean(axis=1)

        delta_v = np.zeros_like(v_curve)
        if num_epochs > 1:
            delta_v[1:] = np.maximum(0.0, v_curve[:-1] - v_curve[1:])

        train_probs = self._softmax(train_logits.astype(np.float64), self.eps)
        train_label_idx = train_labels.reshape(1, -1, 1)
        train_p_true = np.take_along_axis(train_probs, train_label_idx, axis=2).squeeze(2)
        train_probs_other = train_probs.copy()
        train_probs_other[:, np.arange(train_labels.size), train_labels] = -np.inf
        p_other_max = train_probs_other.max(axis=2)
        gap = train_p_true - p_other_max

        delta_gap = np.zeros_like(gap)
        dp = np.empty((0, gap.shape[1]), dtype=gap.dtype)
        if num_epochs > 1:
            dp = gap[1:] - gap[:-1]
            delta_gap[1:] = dp
        tau_p = self._resolve_tau_p(dp)
        scaled = delta_gap / tau_p
        delta_gap_pos = self._softplus(scaled) * tau_p
        delta_gap_pos[0] = 0.0

        fold_scores = np.zeros(train_labels.shape[0], dtype=np.float32)
        for cls in self._iter_classes(num_classes):
            sample_mask = train_labels == cls
            if not np.any(sample_mask):
                continue
            a = delta_gap_pos[:, sample_mask].astype(np.float64)
            b = delta_v[:, cls].astype(np.float64)
            std_b = b.std()
            if std_b < 1e-8:
                fold_scores[sample_mask] = 0.5
                continue
            mean_b = b.mean()
            mean_a = a.mean(axis=0)
            std_a = a.std(axis=0)
            denom = std_a * std_b
            centered_a = a - mean_a
            centered_b = b[:, None] - mean_b
            cov = (centered_a * centered_b).mean(axis=0)
            corr = np.zeros_like(mean_a)
            valid = denom >= 1e-8
            corr[valid] = cov[valid] / denom[valid]
            corr = np.clip(corr, -1.0, 1.0)
            fold_scores[sample_mask] = 0.5 * (corr + 1.0)

        return fold_scores.astype(np.float32)

    def compute(
        self,
        cv_log_dir: str | Path,
        dataset,
        num_classes: int | None = None,
        save_cache: bool = True,
    ) -> dict:
        if self.tau_p <= 0:
            raise ValueError("tau_p must be positive.")
        if self.tau_p_min <= 0:
            raise ValueError("tau_p_min must be positive.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")

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

        bucket: list[list[float]] = [[] for _ in range(num_samples)]
        for fold_path in fold_paths:
            data = np.load(fold_path)
            train_indices = data["train_indices"].astype(np.int64)
            val_indices = data["val_indices"].astype(np.int64)
            train_logits = data["train_logits"].astype(np.float32)
            val_logits = data["val_logits"].astype(np.float32)

            y_train = get_labels_by_indices(dataset, train_indices)
            y_val = get_labels_by_indices(dataset, val_indices)

            fold_scores = self._compute_fold_scores(
                train_logits,
                y_train,
                val_logits,
                y_val,
                num_classes,
            )
            for idx, score in zip(train_indices.tolist(), fold_scores.tolist(), strict=True):
                if idx < 0 or idx >= num_samples:
                    raise ValueError("train_indices out of range.")
                bucket[idx].append(score)

        scores = np.full(num_samples, 0.5, dtype=np.float32)
        for idx, values in enumerate(bucket):
            if not values:
                continue
            if self.agg == "median":
                scores[idx] = float(np.median(values))
            elif self.agg == "mean":
                scores[idx] = float(np.mean(values))
            else:
                raise ValueError(f"Unsupported agg method: {self.agg}")

        seed = meta.get("seed")
        k_folds = meta.get("k_folds")
        if save_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            seed_tag = f"seed{seed}" if seed is not None else "seedunknown"
            cache_path = self.cache_dir / f"transfer_gain_{seed_tag}.npy"
            np.save(cache_path, scores)

        result = {
            "score": scores.astype(np.float32),
            "name": "TransferGainScore",
            "meta": {
                "tau_p_mode": self.tau_p_mode,
                "tau_p": self.tau_p,
                "tau_p_percentile": self.tau_p_percentile,
                "tau_p_min": self.tau_p_min,
                "agg": self.agg,
                "seed": seed,
                "k_folds": k_folds,
                "num_samples": num_samples,
                "num_classes": num_classes,
            },
        }

        if self.verbose:
            print(
                "TransferGainScore scores min/mean/max: "
                f"{scores.min():.6f}, {scores.mean():.6f}, {scores.max():.6f}"
            )

        return result


__all__ = ["TransferGainScore", "get_labels_by_indices"]
