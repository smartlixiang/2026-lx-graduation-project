"""InformativenessScore (B) implementation based on proxy training dynamics."""

# Key interfaces:
# - InformativenessScore.compute() -> InformativenessResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import (
    quantile_minmax_by_class,
    resolve_window_length,
    stable_sigmoid,
)


@dataclass
class InformativenessResult:
    """Container for InformativenessScore outputs."""

    scores: np.ndarray
    hardness: np.ndarray
    delta_gap: np.ndarray
    raw_score: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    early_epochs: int
    late_epochs: int


class InformativenessScore:
    """Compute informativeness score B from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        tau_g: float = 0.10,
        s_g: float = 0.03,
        tau_delta: float = 0.05,
        q_low: float = 0.01,
        q_high: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.tau_g = float(tau_g)
        self.s_g = float(s_g)
        self.tau_delta = float(tau_delta)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)

    def _load_logits(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits_key = "logits_over_epochs" if "logits_over_epochs" in data else "logits"
        if logits_key not in data:
            raise ValueError("proxy log must include 'logits' or 'logits_over_epochs'.")
        if "labels" not in data:
            raise ValueError("labels are required to compute InformativenessScore.")
        logits = data[logits_key].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        indices = data["indices"] if "indices" in data else np.arange(logits.shape[1])
        if logits.ndim != 3:
            raise ValueError("logits array should have shape (epochs, num_samples, num_classes).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError("labels length must match number of samples.")
        if indices.shape[0] != logits.shape[1]:
            raise ValueError("indices length must match number of samples.")
        return logits, labels, indices

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits_max = logits.max(axis=2, keepdims=True)
        shifted = logits - logits_max
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=2, keepdims=True)
        return exp_shifted / (sum_exp + self.eps)

    def compute(self, proxy_logs: Optional[dict[str, np.ndarray]] = None) -> InformativenessResult:
        if self.s_g <= 0.0:
            raise ValueError("s_g must be positive.")
        if self.tau_delta <= 0.0:
            raise ValueError("tau_delta must be positive.")
        data = proxy_logs if proxy_logs is not None else np.load(self.npz_path)
        logits, labels, indices = self._load_logits(data)

        num_epochs = logits.shape[0]
        early_epochs = resolve_window_length(num_epochs, ratio=0.2, min_epochs=5)
        late_epochs = resolve_window_length(num_epochs, ratio=0.2, min_epochs=5)

        probs = self._softmax(logits)
        label_idx = labels.reshape(1, -1, 1)
        p_true = np.take_along_axis(probs, label_idx, axis=2).squeeze(-1)

        probs_other = probs.copy()
        probs_other[:, np.arange(p_true.shape[1]), labels] = -np.inf
        p_other_max = probs_other.max(axis=2)
        gap = p_true - p_other_max

        alpha = stable_sigmoid((self.tau_g - gap) / self.s_g)

        early_slice = slice(0, early_epochs)
        late_slice = slice(num_epochs - late_epochs, num_epochs)

        hardness = alpha[late_slice].mean(axis=0).astype(np.float32)
        delta_gap = (
            gap[late_slice].mean(axis=0) - gap[early_slice].mean(axis=0)
        ).astype(np.float32)
        improve = stable_sigmoid(delta_gap / self.tau_delta)
        raw_score = (hardness * improve).astype(np.float32)

        scores = quantile_minmax_by_class(
            raw_score, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            hardness = hardness[order]
            delta_gap = delta_gap[order]
            raw_score = raw_score[order]
            indices = indices[order]
            labels = labels[order]

        return InformativenessResult(
            scores=scores.astype(np.float32),
            hardness=hardness,
            delta_gap=delta_gap,
            raw_score=raw_score,
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
            late_epochs=late_epochs,
        )


__all__ = ["InformativenessResult", "InformativenessScore"]
