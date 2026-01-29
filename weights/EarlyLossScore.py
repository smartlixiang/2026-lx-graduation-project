"""EarlyLossScore implementation based on proxy training dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class EarlyLossResult:
    """Container for EarlyLossScore outputs."""

    scores: np.ndarray
    early_loss: np.ndarray
    level: np.ndarray
    progress: np.ndarray
    raw_score: np.ndarray
    standardized_score: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    early_epochs: int


class EarlyLossScore:
    """Compute EarlyLossScore from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        early_epochs: int | None = None,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        tau: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.early_epochs = early_epochs
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.eps = float(eps)

    def _resolve_early_epochs(self, num_epochs: int) -> int:
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive.")
        if self.early_epochs is not None:
            if self.early_epochs <= 0:
                raise ValueError("early_epochs must be positive.")
            return min(self.early_epochs, num_epochs)
        default_epochs = num_epochs // 3 if num_epochs >= 3 else 1
        return max(1, min(10, default_epochs))

    @staticmethod
    def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
        positive = values >= 0
        negative = ~positive
        result = np.empty_like(values, dtype=np.float32)
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_vals = np.exp(values[negative])
        result[negative] = exp_vals / (1.0 + exp_vals)
        return result

    def compute(self) -> EarlyLossResult:
        data = np.load(self.npz_path)
        losses = data["loss"]
        labels = data["labels"] if "labels" in data else None
        indices = data["indices"] if "indices" in data else np.arange(losses.shape[1])

        if losses.ndim != 2:
            raise ValueError("loss array should have shape (epochs, num_samples).")
        if indices.shape[0] != losses.shape[1]:
            raise ValueError("indices length must match number of samples.")

        early_epochs = self._resolve_early_epochs(losses.shape[0])
        early_losses = np.log1p(losses[:early_epochs])
        medians = np.median(early_losses, axis=1, keepdims=True)
        mads = np.median(np.abs(early_losses - medians), axis=1, keepdims=True)
        z_scores = (early_losses - medians) / (mads + self.eps)

        level = z_scores.mean(axis=0)
        if early_epochs == 1:
            progress = np.zeros_like(level)
        else:
            progress = z_scores[0, :] - z_scores[-1, :]

        raw_score = self.alpha * level + self.beta * progress
        raw_median = np.median(raw_score)
        raw_mad = np.median(np.abs(raw_score - raw_median))
        standardized = (raw_score - raw_median) / (raw_mad + self.eps)
        scores = self._stable_sigmoid(standardized / self.tau)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            raw_score = raw_score[order]
            standardized = standardized[order]
            level = level[order]
            progress = progress[order]
            indices = indices[order]
            if labels is not None:
                labels = labels[order]

        return EarlyLossResult(
            scores=scores,
            early_loss=raw_score,
            level=level,
            progress=progress,
            raw_score=raw_score,
            standardized_score=standardized,
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
        )


__all__ = ["EarlyLossResult", "EarlyLossScore"]
