"""MarginScore implementation based on proxy training dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MarginResult:
    """Container for MarginScore outputs."""

    scores: np.ndarray
    margins: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    delta: float


class MarginScore:
    """Compute MarginScore from proxy training logs (.npz)."""

    def __init__(self, npz_path: str | Path, delta: float = 1.0) -> None:
        self.npz_path = Path(npz_path)
        self.delta = float(delta)

    @staticmethod
    def _validate_shapes(logits: np.ndarray, correct: np.ndarray, labels: np.ndarray) -> None:
        if logits.ndim != 3:
            raise ValueError("logits array should have shape (epochs, num_samples, num_classes).")
        if correct.ndim != 2:
            raise ValueError("correct array should have shape (epochs, num_samples).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if logits.shape[0] != correct.shape[0] or logits.shape[1] != correct.shape[1]:
            raise ValueError("logits and correct arrays must share (epochs, num_samples).")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError("labels length must match number of samples.")

    def compute(self) -> MarginResult:
        data = np.load(self.npz_path)
        logits = data["logits"]
        correct = data["correct"]
        labels = data["labels"] if "labels" in data else None
        indices = data["indices"] if "indices" in data else np.arange(logits.shape[1])

        if labels is None:
            raise ValueError("labels are required to compute margins.")

        self._validate_shapes(logits, correct, labels)

        num_epochs, num_samples, _ = logits.shape
        labels = labels.astype(np.int64)
        correct = correct.astype(bool)

        label_idx = labels.reshape(1, num_samples, 1)
        true_logits = np.take_along_axis(logits, label_idx, axis=2).squeeze(-1)

        logits_other = logits.copy()
        logits_other[:, np.arange(num_samples), labels] = -np.inf
        max_other = logits_other.max(axis=2)

        margins = true_logits - max_other

        scores_per_epoch = np.zeros((num_epochs, num_samples), dtype=np.float32)
        scores_per_epoch[~correct] = -1.0
        boundary_hits = correct & (margins <= self.delta)
        scores_per_epoch[boundary_hits] = 1.0

        mean_scores = scores_per_epoch.mean(axis=0)
        scores = ((mean_scores + 1.0) / 2.0).astype(np.float32)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            margins = margins[:, order]
            indices = indices[order]
            labels = labels[order]

        return MarginResult(
            scores=scores,
            margins=margins,
            labels=labels,
            indices=indices,
            delta=self.delta,
        )


__all__ = ["MarginResult", "MarginScore"]
