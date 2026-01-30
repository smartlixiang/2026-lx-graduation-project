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

    def __init__(
        self,
        npz_path: str | Path,
        delta: float | None = None,
        tau_m: float = 1.0,
        start_ratio: float = 0.2,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.tau_m = float(tau_m)
        if delta is not None and start_ratio == 0.2 and delta != 1.0:
            start_ratio = float(delta)
        self.start_ratio = float(start_ratio)
        # delta field is retained for backward-compatible result structure.
        # It now stores the start_ratio for MarginScore aggregation.
        self.delta = float(self.start_ratio)

    @staticmethod
    def _validate_shapes(logits: np.ndarray, labels: np.ndarray) -> None:
        if logits.ndim != 3:
            raise ValueError("logits array should have shape (epochs, num_samples, num_classes).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError("labels length must match number of samples.")

    def compute(self) -> MarginResult:
        data = np.load(self.npz_path)
        logits = data["logits"]
        labels = data["labels"] if "labels" in data else None
        indices = data["indices"] if "indices" in data else np.arange(logits.shape[1])
        _ = data["correct"] if "correct" in data else None

        if labels is None:
            raise ValueError("labels are required to compute margins.")

        self._validate_shapes(logits, labels)

        if self.tau_m <= 0:
            raise ValueError("tau_m must be positive.")

        num_epochs, num_samples, _ = logits.shape
        labels = labels.astype(np.int64)

        scaled_logits = logits.astype(np.float32) / self.tau_m
        scaled_logits = scaled_logits - scaled_logits.max(axis=2, keepdims=True)
        exp_logits = np.exp(scaled_logits)
        prob = exp_logits / exp_logits.sum(axis=2, keepdims=True)

        label_idx = labels.reshape(1, num_samples, 1)
        p_true = np.take_along_axis(prob, label_idx, axis=2).squeeze(-1)

        prob_other = prob.copy()
        prob_other[:, np.arange(num_samples), labels] = -np.inf
        p_other_max = prob_other.max(axis=2)

        margins = p_true - p_other_max

        t0 = int(np.floor(self.start_ratio * num_epochs))
        t0 = min(max(t0, 0), max(num_epochs - 1, 0))
        mean_margin = margins[t0:, :].mean(axis=0)
        scores = np.clip((mean_margin + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)

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
