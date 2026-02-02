"""CoverageGainScore implementation based on proxy training dynamics."""

# Key interfaces:
# - CoverageGainScore.compute() -> CoverageGainResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import quantile_minmax_by_class, stable_sigmoid


@dataclass
class CoverageGainResult:
    """Container for CoverageGainScore outputs."""

    scores: np.ndarray
    knn_distance: np.ndarray
    q_confusion: np.ndarray
    alpha_sum: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray


class CoverageGainScore:
    """Compute CoverageGainScore from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        tau_g: float = 0.15,
        s_g: float = 0.07,
        k: int = 10,
        q_low: float = 0.01,
        q_high: float = 0.99,
        eps: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.tau_g = float(tau_g)
        self.s_g = float(s_g)
        self.k = int(k)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self.verbose = bool(verbose)

    @staticmethod
    def _validate_logits(logits: np.ndarray, labels: np.ndarray) -> None:
        if logits.ndim != 3:
            raise ValueError("logits array should have shape (epochs, num_samples, num_classes).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError("labels length must match number of samples.")

    def _load_logits(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "logits" not in data:
            raise ValueError("proxy log must include 'logits' for CoverageGainScore.")
        if "labels" not in data:
            raise ValueError("labels are required to compute CoverageGainScore from logits.")
        logits = data["logits"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        indices = data["indices"] if "indices" in data else np.arange(logits.shape[1])
        self._validate_logits(logits, labels)
        if indices.shape[0] != logits.shape[1]:
            raise ValueError("indices length must match number of samples.")
        return logits, labels, indices

    def _compute_softmax(self, logits: np.ndarray) -> np.ndarray:
        logits_max = logits.max(axis=2, keepdims=True)
        shifted = logits - logits_max
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=2, keepdims=True)
        return exp_shifted / (sum_exp + self.eps)

    def _compute_p_true(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        label_idx = labels.reshape(1, -1, 1)
        return np.take_along_axis(probs, label_idx, axis=2).squeeze(-1)

    def _compute_knn_mean_distance(
        self,
        q_class: np.ndarray,
        k: int,
        chunk_size: int = 512,
    ) -> np.ndarray:
        num_samples = q_class.shape[0]
        if num_samples <= 1:
            return np.zeros(num_samples, dtype=np.float32)
        k = max(1, min(k, num_samples - 1))
        q_class = q_class.astype(np.float64, copy=False)
        norms = np.sum(q_class**2, axis=1)
        distances = np.empty(num_samples, dtype=np.float64)
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            block = q_class[start:end]
            block_norms = np.sum(block**2, axis=1, keepdims=True)
            dot = block @ q_class.T
            dist2 = block_norms + norms[None, :] - 2.0 * dot
            dist2 = np.maximum(dist2, 0.0)
            for local_idx, global_idx in enumerate(range(start, end)):
                dist2[local_idx, global_idx] = np.inf
            nearest = np.partition(dist2, k, axis=1)[:, :k]
            distances[start:end] = np.sqrt(nearest).mean(axis=1)
        return distances.astype(np.float32)

    def compute(self) -> CoverageGainResult:
        if self.s_g <= 0.0:
            raise ValueError("s_g must be positive.")
        if self.k <= 0:
            raise ValueError("k must be positive.")
        if not 0.0 <= self.q_low < self.q_high <= 1.0:
            raise ValueError("q_low/q_high must satisfy 0 <= q_low < q_high <= 1.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        data = np.load(self.npz_path)
        logits, labels, indices = self._load_logits(data)

        num_epochs, num_samples, num_classes = logits.shape
        if num_epochs == 0:
            raise ValueError("logits array must contain at least one epoch.")

        probs = self._compute_softmax(logits).astype(np.float32)
        p_true = self._compute_p_true(probs, labels)

        probs_other = probs.copy()
        probs_other[:, np.arange(num_samples), labels] = -np.inf
        p_other_max = probs_other.max(axis=2)
        gap = p_true - p_other_max
        alpha = stable_sigmoid((self.tau_g - gap) / self.s_g).astype(np.float32)
        alpha_sum = alpha.sum(axis=0).astype(np.float32)

        q_sum = np.zeros((num_samples, num_classes), dtype=np.float64)
        for t in range(num_epochs):
            tmp = probs[t].astype(np.float64, copy=True)
            tmp[np.arange(num_samples), labels] = 0.0
            denom = tmp.sum(axis=1, keepdims=True)
            tmp = tmp / (denom + self.eps)
            q_sum += alpha[t].astype(np.float64)[:, None] * tmp
        q_confusion = (q_sum / (alpha_sum.astype(np.float64)[:, None] + self.eps)).astype(
            np.float32
        )

        knn_distance = np.zeros(num_samples, dtype=np.float32)
        class_k: dict[int, int] = {}
        for cls in np.unique(labels):
            mask = labels == cls
            idx = np.where(mask)[0]
            count = idx.size
            if count <= 1:
                class_k[int(cls)] = 0
                continue
            k_eff = self.k if count > self.k else min(5, count - 1)
            class_k[int(cls)] = k_eff
            q_class = q_confusion[idx]
            distances = self._compute_knn_mean_distance(q_class, k_eff)
            knn_distance[idx] = distances
        score_raw = quantile_minmax_by_class(
            knn_distance, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )
        scores = np.clip(score_raw, 0.0, 1.0).astype(np.float32)

        if self.verbose:
            print(
                "CoverageGainScore scores min/mean/max: "
                f"{scores.min():.6f}, {scores.mean():.6f}, {scores.max():.6f}"
            )
            for cls in sorted(class_k):
                print(f"CoverageGainScore class {cls}: n={np.sum(labels == cls)}, k={class_k[cls]}")

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            knn_distance = knn_distance[order]
            q_confusion = q_confusion[order]
            alpha_sum = alpha_sum[order]
            indices = indices[order]
            labels = labels[order]

        return CoverageGainResult(
            scores=scores,
            knn_distance=knn_distance,
            q_confusion=q_confusion,
            alpha_sum=alpha_sum,
            labels=labels,
            indices=indices,
        )


__all__ = ["CoverageGainResult", "CoverageGainScore"]
