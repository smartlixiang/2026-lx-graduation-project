"""StabilityScore implementation based on proxy training dynamics.

Score formula (learnable samples only):
score = a * S + b * (L * S) + c * (L * (1 - S))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class StabilityResult:
    """Container for StabilityScore outputs."""

    scores: np.ndarray
    learn_time: np.ndarray
    learn_time_normalized: np.ndarray
    post_stability: np.ndarray
    learnable_mask: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    window: int
    stable_weight: float
    late_bonus: float
    unstable_weight: float


class StabilityScore:
    """Compute StabilityScore from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        window: int = 10,
        stable_weight: float = 0.85,
        late_bonus: float = 0.15,
        unstable_weight: float = 0.20,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.window = int(window)
        self.stable_weight = float(stable_weight)
        self.late_bonus = float(late_bonus)
        self.unstable_weight = float(unstable_weight)

    @staticmethod
    def _validate_correct(correct: np.ndarray) -> None:
        if correct.ndim != 2:
            raise ValueError("correct array should have shape (epochs, num_samples).")

    @staticmethod
    def _validate_logits(logits: np.ndarray, labels: np.ndarray) -> None:
        if logits.ndim != 3:
            raise ValueError("logits array should have shape (epochs, num_samples, num_classes).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if logits.shape[1] != labels.shape[0]:
            raise ValueError("labels length must match number of samples.")

    def _build_correct(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if "correct" in data:
            correct = data["correct"]
            labels = data["labels"] if "labels" in data else None
            return correct, labels

        if "logits" not in data:
            raise ValueError("proxy log must include either 'correct' or 'logits'.")

        labels = data["labels"] if "labels" in data else None
        if labels is None:
            raise ValueError("labels are required to compute correct from logits.")
        logits = data["logits"]
        self._validate_logits(logits, labels)

        labels = labels.astype(np.int64)
        preds = logits.argmax(axis=2)
        correct = preds == labels.reshape(1, -1)
        return correct, labels

    def compute(self) -> StabilityResult:
        if self.window <= 0:
            raise ValueError("window must be a positive integer.")
        if self.stable_weight < 0 or self.late_bonus < 0 or self.unstable_weight < 0:
            raise ValueError("stability weights must be non-negative.")
        if not np.isclose(self.stable_weight + self.late_bonus, 1.0):
            raise ValueError("stable_weight + late_bonus must equal 1.")

        data = np.load(self.npz_path)
        correct, labels = self._build_correct(data)
        indices = data["indices"] if "indices" in data else np.arange(correct.shape[1])

        self._validate_correct(correct)
        if indices.shape[0] != correct.shape[1]:
            raise ValueError("indices length must match number of samples.")

        num_epochs, num_samples = correct.shape
        if num_epochs == 0:
            raise ValueError("correct array must contain at least one epoch.")

        correct = correct.astype(bool)
        learnable_mask = np.zeros(num_samples, dtype=bool)
        learn_time = np.full(num_samples, -1, dtype=np.int32)
        learn_time_normalized = np.ones(num_samples, dtype=np.float32)
        post_stability = np.zeros(num_samples, dtype=np.float32)
        scores = np.zeros(num_samples, dtype=np.float32)

        if num_epochs >= self.window:
            correct_int = correct.astype(np.int32)
            cumsum = np.cumsum(correct_int, axis=0)
            pad = np.zeros((1, num_samples), dtype=cumsum.dtype)
            window_sum = cumsum[self.window - 1:] - np.concatenate(
                [pad, cumsum[: -self.window]], axis=0
            )
            learnable = window_sum == self.window
            learnable_mask = learnable.any(axis=0)

            first_idx = np.argmax(learnable, axis=0)
            t_learn0 = np.where(learnable_mask, first_idx, -1)
            learn_time = np.where(learnable_mask, t_learn0 + 1, -1).astype(np.int32)

            denom = max(1, num_epochs - self.window)
            learn_time_normalized = np.where(
                learnable_mask,
                t_learn0.astype(np.float32) / float(denom),
                1.0,
            ).astype(np.float32)

            correct_float = correct.astype(np.float32)
            cumsum_tail = np.cumsum(correct_float[::-1], axis=0)[::-1]
            lengths = np.arange(num_epochs, 0, -1, dtype=np.float32)
            sample_indices = np.arange(num_samples)
            valid_samples = sample_indices[learnable_mask]
            if valid_samples.size > 0:
                post_stability[valid_samples] = (
                    cumsum_tail[t_learn0[valid_samples], valid_samples]
                    / lengths[t_learn0[valid_samples]]
                )

            learnable_float = learnable_mask.astype(np.float32)
            raw_scores = (
                self.stable_weight * post_stability
                + self.late_bonus * (learn_time_normalized * post_stability)
                + self.unstable_weight * (learn_time_normalized * (1.0 - post_stability))
            )
            raw_scores = np.clip(raw_scores, 0.0, 1.0)
            scores = (learnable_float * raw_scores).astype(np.float32)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            learn_time = learn_time[order]
            learn_time_normalized = learn_time_normalized[order]
            post_stability = post_stability[order]
            learnable_mask = learnable_mask[order]
            indices = indices[order]
            if labels is not None:
                labels = labels[order]

        return StabilityResult(
            scores=scores,
            learn_time=learn_time,
            learn_time_normalized=learn_time_normalized,
            post_stability=post_stability,
            learnable_mask=learnable_mask,
            labels=labels,
            indices=indices,
            window=self.window,
            stable_weight=self.stable_weight,
            late_bonus=self.late_bonus,
            unstable_weight=self.unstable_weight,
        )


__all__ = ["StabilityResult", "StabilityScore"]
