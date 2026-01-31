"""StabilityScore implementation based on proxy training dynamics (v5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class StabilityResult:
    """Minimal container for StabilityScore outputs used in the main pipeline."""

    scores: np.ndarray
    learn_time_normalized: np.ndarray
    post_stability: np.ndarray
    g_learn: np.ndarray
    learnable_mask: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray


class StabilityScore:
    """Compute StabilityScore v5 from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        beta: float = 0.9,
        theta: float = 0.7,
        lam: float = 1.0,
        t_learn: float = 0.2,
        b_min: float = 0.75,
        b_max: float = 1.00,
        gamma_s: float = 1.0,
        delta_max: float = 0.15,
        gamma_delta: float = 2.0,
        pen: float = 0.15,
        gamma_pen: float = 2.0,
        u_max: float = 0.20,
        gamma_u: float = 2.5,
        eps: float = 1e-8,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.beta = float(beta)
        self.theta = float(theta)
        self.lam = float(lam)
        self.t_learn = float(t_learn)
        self.b_min = float(b_min)
        self.b_max = float(b_max)
        self.gamma_s = float(gamma_s)
        self.delta_max = float(delta_max)
        self.gamma_delta = float(gamma_delta)
        self.pen = float(pen)
        self.gamma_pen = float(gamma_pen)
        self.u_max = float(u_max)
        self.gamma_u = float(gamma_u)
        self.eps = float(eps)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

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
            raise ValueError("proxy log must include 'logits' for StabilityScore.")
        if "labels" not in data:
            raise ValueError("labels are required to compute StabilityScore from logits.")
        logits = data["logits"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        indices = data["indices"] if "indices" in data else np.arange(logits.shape[1])
        self._validate_logits(logits, labels)
        if indices.shape[0] != logits.shape[1]:
            raise ValueError("indices length must match number of samples.")
        return logits, labels, indices

    def _compute_p_true(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        logits_max = logits.max(axis=2, keepdims=True)
        shifted = logits - logits_max
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=2)
        label_idx = labels.reshape(1, -1, 1)
        true_exp = np.take_along_axis(exp_shifted, label_idx, axis=2).squeeze(-1)
        return true_exp / (sum_exp + self.eps)

    def _compute_b11(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        num_epochs, num_samples, _ = logits.shape
        label_idx = labels.reshape(1, num_samples, 1)
        true_logits = np.take_along_axis(logits, label_idx, axis=2).squeeze(-1)
        logits_other = logits.copy()
        logits_other[:, np.arange(num_samples), labels] = -np.inf
        max_other = logits_other.max(axis=2)
        return true_logits - max_other

    def compute(self) -> StabilityResult:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1).")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1].")
        if self.t_learn <= 0.0:
            raise ValueError("t_learn must be positive.")
        if self.lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        data = np.load(self.npz_path)
        logits, labels, indices = self._load_logits(data)

        num_epochs, num_samples, _ = logits.shape
        if num_epochs == 0:
            raise ValueError("logits array must contain at least one epoch.")

        p_true = self._compute_p_true(logits, labels)

        ema = np.zeros_like(p_true, dtype=np.float32)
        ema[0] = p_true[0]
        for t in range(1, num_epochs):
            ema[t] = self.beta * ema[t - 1] + (1.0 - self.beta) * p_true[t]

        a_max = ema.max(axis=0)
        reach = ema >= self.theta
        reach_any = reach.any(axis=0)
        t_reach = np.argmax(reach, axis=0)
        t_star = np.argmax(ema, axis=0)
        t_anchor = np.where(reach_any, t_reach, t_star).astype(np.int32)

        denom = max(1, num_epochs - 1)
        learn_time_normalized = (t_anchor.astype(np.float32) / float(denom)).astype(np.float32)

        b11 = self._compute_b11(logits, labels)
        mu_post = np.zeros(num_samples, dtype=np.float32)
        sigma_post = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            start = t_anchor[i]
            post = b11[start:, i]
            mu_post[i] = float(np.mean(post))
            sigma_post[i] = float(np.std(post))

        post_stability_raw = mu_post - self.lam * sigma_post
        post_stability = np.zeros_like(post_stability_raw, dtype=np.float32)
        for cls in np.unique(labels):
            mask = labels == cls
            if not np.any(mask):
                continue
            class_raw = post_stability_raw[mask]
            lo = float(np.percentile(class_raw, 1))
            hi = float(np.percentile(class_raw, 99))
            post_stability[mask] = np.clip(
                (class_raw - lo) / (hi - lo + self.eps), 0.0, 1.0
            ).astype(np.float32)

        b_term = self.b_min + (self.b_max - self.b_min) * (post_stability**self.gamma_s)
        delta_term = self.delta_max * (post_stability**self.gamma_delta)
        penalty = self.pen * ((1.0 - post_stability) ** self.gamma_pen)
        score_learn = b_term + delta_term * learn_time_normalized - penalty

        score_unlearn = self.u_max * (post_stability**self.gamma_u)
        score_unlearn = np.clip(score_unlearn, 0.0, self.u_max)

        g_learn = self._sigmoid((a_max - self.theta) / self.t_learn).astype(np.float32)
        score_raw = g_learn * score_learn + (1.0 - g_learn) * score_unlearn
        scores = np.clip(score_raw, 0.0, 1.0).astype(np.float32)
        learnable_mask = reach_any.astype(bool)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            learn_time_normalized = learn_time_normalized[order]
            post_stability = post_stability[order]
            g_learn = g_learn[order]
            learnable_mask = learnable_mask[order]
            indices = indices[order]
            labels = labels[order]

        return StabilityResult(
            scores=scores,
            learn_time_normalized=learn_time_normalized,
            post_stability=post_stability,
            g_learn=g_learn,
            learnable_mask=learnable_mask,
            labels=labels,
            indices=indices,
        )


__all__ = ["StabilityResult", "StabilityScore"]
