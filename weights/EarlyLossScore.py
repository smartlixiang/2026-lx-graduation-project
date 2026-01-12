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
    labels: Optional[np.ndarray]
    indices: np.ndarray
    early_epochs: int


class EarlyLossScore:
    """Compute EarlyLossScore from proxy training logs (.npz)."""

    def __init__(self, npz_path: str | Path, early_epochs: int | None = None) -> None:
        self.npz_path = Path(npz_path)
        self.early_epochs = early_epochs

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
    def _min_max_normalize(values: np.ndarray) -> np.ndarray:
        min_val = float(values.min())
        max_val = float(values.max())
        if np.isclose(max_val, min_val):
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_val) / (max_val - min_val)).astype(np.float32)

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
        early_mean = early_losses.mean(axis=0)
        scores = self._min_max_normalize(early_mean)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            early_mean = early_mean[order]
            indices = indices[order]
            if labels is not None:
                labels = labels[order]

        return EarlyLossResult(
            scores=scores,
            early_loss=early_mean,
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
        )


__all__ = ["EarlyLossResult", "EarlyLossScore"]
