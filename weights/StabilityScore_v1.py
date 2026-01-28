"""StabilityScore v1 with normalized gain based on global accuracy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class StabilityResultV1:
    """Container for StabilityScore v1 outputs."""

    scores: np.ndarray
    learn_time: np.ndarray
    learn_time_normalized: np.ndarray
    post_stability: np.ndarray
    learnability: np.ndarray
    learnable_mask: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    window: int
    threshold: float
    temperature: float
    u0: float
    u1: float
    v0: float
    v1: float
    gamma: float
    eta: float
    s_global: np.ndarray
    gain: np.ndarray


class StabilityScore:
    """Compute StabilityScore v1 from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        window: int = 5,
        threshold: float = 0.8,
        temperature: float = 0.04,
        u0: float = 0.65,
        u1: float = 0.35,
        v0: float = 0.05,
        v1: float = 0.20,
        gamma: float = 1.0,
        eta: float = 0.25,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.window = int(window)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.u0 = float(u0)
        self.u1 = float(u1)
        self.v0 = float(v0)
        self.v1 = float(v1)
        self.gamma = float(gamma)
        self.eta = float(eta)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

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

    def compute(self) -> StabilityResultV1:
        if self.window <= 0:
            raise ValueError("window must be a positive integer.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if min(self.u0, self.u1, self.v0, self.v1) < 0:
            raise ValueError("u0/u1/v0/v1 must be non-negative.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")

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
        learn_time_normalized = np.zeros(num_samples, dtype=np.float32)
        post_stability = np.zeros(num_samples, dtype=np.float32)
        learnability = np.zeros(num_samples, dtype=np.float32)
        scores = np.zeros(num_samples, dtype=np.float32)

        effective_window = min(self.window, num_epochs)
        correct_int = correct.astype(np.int32)
        cumsum = np.cumsum(correct_int, axis=0)
        pad = np.zeros((1, num_samples), dtype=cumsum.dtype)
        window_sum = cumsum[effective_window - 1:] - np.concatenate(
            [pad, cumsum[: -effective_window]], axis=0
        )
        r_values = window_sum.astype(np.float32) / float(effective_window)
        learnability = self._sigmoid((r_values.max(axis=0) - self.threshold) / self.temperature)

        reach = r_values >= self.threshold
        learnable_mask = reach.any(axis=0)
        t_reach0 = np.argmax(reach, axis=0)
        t_star0 = np.argmax(r_values, axis=0)
        t_anchor0 = np.where(learnable_mask, t_reach0, t_star0).astype(np.int32)

        learn_time = (t_anchor0 + 1).astype(np.int32)
        denom = max(1, num_epochs - effective_window)
        learn_time_normalized = (t_anchor0.astype(np.float32) / float(denom)).astype(np.float32)

        correct_float = correct.astype(np.float32)
        cumsum_tail = np.cumsum(correct_float[::-1], axis=0)[::-1]
        lengths = np.arange(num_epochs, 0, -1, dtype=np.float32)
        sample_indices = np.arange(num_samples)
        post_stability[sample_indices] = (
            cumsum_tail[t_anchor0, sample_indices] / lengths[t_anchor0]
        )

        s_global = correct_float.mean(axis=0)
        eps = np.finfo(np.float32).eps
        gain = (post_stability - s_global) / (1.0 - s_global + eps)
        gain = np.clip(gain, 0.0, 1.0).astype(np.float32)
        s_eff = np.clip(post_stability + self.eta * gain, 0.0, 1.0)

        upper = self.u0 + self.u1 * learn_time_normalized
        lower = self.v0 + self.v1 * learn_time_normalized
        raw_scores = lower + (upper - lower) * s_eff
        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        scores = (learnability * raw_scores).astype(np.float32)
        if not np.isclose(self.gamma, 1.0):
            scores = np.power(scores, self.gamma, dtype=np.float32)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            learn_time = learn_time[order]
            learn_time_normalized = learn_time_normalized[order]
            post_stability = post_stability[order]
            learnability = learnability[order]
            learnable_mask = learnable_mask[order]
            s_global = s_global[order]
            gain = gain[order]
            indices = indices[order]
            if labels is not None:
                labels = labels[order]

        return StabilityResultV1(
            scores=scores,
            learn_time=learn_time,
            learn_time_normalized=learn_time_normalized,
            post_stability=post_stability,
            learnability=learnability,
            learnable_mask=learnable_mask,
            labels=labels,
            indices=indices,
            window=self.window,
            threshold=self.threshold,
            temperature=self.temperature,
            u0=self.u0,
            u1=self.u1,
            v0=self.v0,
            v1=self.v1,
            gamma=self.gamma,
            eta=self.eta,
            s_global=s_global.astype(np.float32),
            gain=gain,
        )


__all__ = ["StabilityResultV1", "StabilityScore"]
