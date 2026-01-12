"""ForgettingScore implementation based on proxy training dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ForgettingResult:
    """Container for ForgettingScore outputs."""

    scores: np.ndarray
    accuracy: np.ndarray
    forgetting_counts: np.ndarray
    forgetting_normalized: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    start_epoch: int


class ForgettingScore:
    """Compute ForgettingScore from proxy training logs (.npz)."""

    def __init__(self, npz_path: str | Path) -> None:
        self.npz_path = Path(npz_path)

    @staticmethod
    def _min_max_normalize(values: np.ndarray) -> np.ndarray:
        min_val = float(values.min())
        max_val = float(values.max())
        if np.isclose(max_val, min_val):
            return np.zeros_like(values, dtype=np.float32)
        return ((values - min_val) / (max_val - min_val)).astype(np.float32)

    @staticmethod
    def _bilinear_score(accuracy: np.ndarray, forgetting: np.ndarray) -> np.ndarray:
        """Interpolate scores from the four corner cases described in the plan."""
        return (
            (1 - accuracy) * (1 - forgetting) * 0.9
            + accuracy * (1 - forgetting) * 0.7
            + (1 - accuracy) * forgetting * 0.1
            + accuracy * forgetting * 0.2
        ).astype(np.float32)

    @staticmethod
    def _validate_shapes(correct: np.ndarray) -> None:
        if correct.ndim != 2:
            raise ValueError("correct array should have shape (epochs, num_samples).")

    def compute(self) -> ForgettingResult:
        data = np.load(self.npz_path)
        correct = data["correct"]
        labels = data["labels"] if "labels" in data else None
        indices = data["indices"] if "indices" in data else np.arange(correct.shape[1])

        self._validate_shapes(correct)
        if indices.shape[0] != correct.shape[1]:
            raise ValueError("indices length must match number of samples.")

        num_epochs = correct.shape[0]
        if num_epochs == 0:
            raise ValueError("correct array must contain at least one epoch.")

        correct = correct.astype(bool)
        accuracy = correct.mean(axis=0).astype(np.float32)

        start_epoch = num_epochs // 2
        if num_epochs < 2:
            forgetting_counts = np.zeros(correct.shape[1], dtype=np.int32)
        else:
            transitions = correct[start_epoch:-1] & ~correct[start_epoch + 1 :]
            forgetting_counts = transitions.sum(axis=0).astype(np.int32)

        forgetting_normalized = self._min_max_normalize(forgetting_counts)
        scores = self._bilinear_score(accuracy, forgetting_normalized)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            accuracy = accuracy[order]
            forgetting_counts = forgetting_counts[order]
            forgetting_normalized = forgetting_normalized[order]
            indices = indices[order]
            if labels is not None:
                labels = labels[order]

        return ForgettingResult(
            scores=scores,
            accuracy=accuracy,
            forgetting_counts=forgetting_counts,
            forgetting_normalized=forgetting_normalized,
            labels=labels,
            indices=indices,
            start_epoch=start_epoch,
        )


__all__ = ["ForgettingResult", "ForgettingScore"]
