"""Noise-aware Boundary Informativeness score based on proxy training dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BoundaryInfoResult:
    """Container for BoundaryInfoScore outputs.

    This score captures boundary informativeness under the assumption that the
    sample is learnable: it favors samples that stay close to the decision
    boundary yet are not consistently noisy or conflicting.
    """

    scores: np.ndarray
    boundary_closeness: np.ndarray
    noise_penalty: np.ndarray
    g_learn: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray


class BoundaryInfoScore:
    """Compute Noise-aware Boundary Informativeness from proxy training logs."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        beta: float = 0.9,
        theta: float = 0.5,
        t_learn: float = 0.3,
        late_epochs: int = 30,
        sigma: float = 1.0,
        sigma_q: float = 60.0,
        tau_n: float = 1.0,
        eps: float = 1e-8,
        use_prob_gap: bool = False,
        verbose: bool = False,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.beta = float(beta)
        self.theta = float(theta)
        self.t_learn = float(t_learn)
        self.late_epochs = int(late_epochs)
        self.sigma = float(sigma)
        self.sigma_q = float(sigma_q)
        self.tau_n = float(tau_n)
        self.eps = float(eps)
        self.use_prob_gap = bool(use_prob_gap)
        self.verbose = bool(verbose)

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
            raise ValueError("proxy log must include 'logits' for BoundaryInfoScore.")
        if "labels" not in data:
            raise ValueError("labels are required to compute BoundaryInfoScore from logits.")
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

    def _compute_prob_gap(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float64)
        logits_max = logits.max(axis=2, keepdims=True)
        shifted = logits - logits_max
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum(axis=2, keepdims=True)
        probs = exp_shifted / (sum_exp + self.eps)

        num_epochs, num_samples, _ = probs.shape
        label_idx = labels.reshape(1, num_samples, 1)
        p_true = np.take_along_axis(probs, label_idx, axis=2).squeeze(-1)
        probs_other = probs.copy()
        probs_other[:, np.arange(num_samples), labels] = -np.inf
        p_second = probs_other.max(axis=2)
        return (p_true - p_second).astype(np.float64)

    def _compute_margins(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        num_epochs, num_samples, _ = logits.shape
        label_idx = labels.reshape(1, num_samples, 1)
        true_logits = np.take_along_axis(logits, label_idx, axis=2).squeeze(-1)
        logits_other = logits.copy()
        logits_other[:, np.arange(num_samples), labels] = -np.inf
        max_other = logits_other.max(axis=2)
        return true_logits - max_other

    def compute(self) -> BoundaryInfoResult:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1).")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1].")
        if self.t_learn <= 0.0:
            raise ValueError("t_learn must be positive.")
        if self.late_epochs <= 0:
            raise ValueError("late_epochs must be positive.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if not 0.0 < self.sigma_q < 100.0:
            raise ValueError("sigma_q must be in (0, 100).")
        if self.tau_n <= 0.0:
            raise ValueError("tau_n must be positive.")
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
        g_learn = self._sigmoid((a_max - self.theta) / self.t_learn).astype(np.float32)

        margins = self._compute_prob_gap(logits, labels) if self.use_prob_gap else self._compute_margins(
            logits, labels
        )
        start = max(0, num_epochs - self.late_epochs)
        late_margins = margins[start:, :]

        pos = np.maximum(late_margins, 0.0)
        pos_all = pos[pos > 0]
        if pos_all.size == 0:
            sigma_eff = max(self.sigma, self.eps)
        else:
            sigma_eff = float(np.percentile(pos_all, self.sigma_q))
            sigma_eff = max(sigma_eff, self.eps)
        x = pos.astype(np.float64) / (sigma_eff + self.eps)
        boundary_closeness = np.mean(np.exp(-x), axis=0).astype(np.float32)
        noise_penalty = np.mean(self._sigmoid((-late_margins) / self.tau_n), axis=0).astype(
            np.float32
        )

        raw = g_learn * boundary_closeness * (1.0 - noise_penalty)
        if self.verbose:
            bc_q = np.percentile(boundary_closeness, [1, 50, 99])
            raw_q = np.percentile(raw, [1, 50, 99])
            print(
                "BoundaryInfoScore boundary_closeness percentiles (1/50/99): "
                f"{bc_q[0]:.6f}, {bc_q[1]:.6f}, {bc_q[2]:.6f}"
            )
            print(
                "BoundaryInfoScore raw percentiles (1/50/99): "
                f"{raw_q[0]:.6f}, {raw_q[1]:.6f}, {raw_q[2]:.6f}"
            )
        lo = float(np.percentile(raw, 1))
        hi = float(np.percentile(raw, 99))
        scores = np.clip((raw - lo) / (hi - lo + self.eps), 0.0, 1.0).astype(np.float32)

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            boundary_closeness = boundary_closeness[order]
            noise_penalty = noise_penalty[order]
            g_learn = g_learn[order]
            indices = indices[order]
            labels = labels[order]

        return BoundaryInfoResult(
            scores=scores,
            boundary_closeness=boundary_closeness,
            noise_penalty=noise_penalty,
            g_learn=g_learn,
            labels=labels,
            indices=indices,
        )


__all__ = ["BoundaryInfoResult", "BoundaryInfoScore"]
