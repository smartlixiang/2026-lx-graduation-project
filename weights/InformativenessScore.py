"""InformativenessScore (B) implementation based on proxy training dynamics."""

# Key interfaces:
# - InformativenessScore.compute() -> InformativenessResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import quantile_minmax_by_class, resolve_early_late_slices


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
        q_low: float = 0.002,
        q_high: float = 0.998,
        mu_percentile: float = 30.0,
        stats_by_class: bool = True,
        iqr_p0: float = 25.0,
        iqr_p1: float = 75.0,
        iqr_scale: float = 0.5,
        tau_p_mode: str = "percentile",
        tau_p: float = 0.02,
        tau_p_percentile: float = 90.0,
        tau_p_min: float = 1e-3,
        early_late_ratio: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.mu_percentile = float(mu_percentile)
        self.stats_by_class = bool(stats_by_class)
        self.iqr_p0 = float(iqr_p0)
        self.iqr_p1 = float(iqr_p1)
        self.iqr_scale = float(iqr_scale)
        self.tau_p_mode = str(tau_p_mode)
        self.tau_p = float(tau_p)
        self.tau_p_percentile = float(tau_p_percentile)
        self.tau_p_min = float(tau_p_min)
        self.early_late_ratio = float(early_late_ratio)
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

    def _compute_mu_sigma(
        self, gL: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.stats_by_class:
            mu_arr = np.empty_like(gL, dtype=np.float64)
            sigma_arr = np.empty_like(gL, dtype=np.float64)
            for cls in np.unique(labels):
                mask = labels == cls
                gLc = gL[mask]
                if gLc.size == 0:
                    continue
                mu_c = np.percentile(gLc, self.mu_percentile)
                p0 = np.percentile(gLc, self.iqr_p0)
                p1 = np.percentile(gLc, self.iqr_p1)
                sigma_c = self.iqr_scale * (p1 - p0) + self.eps
                mu_arr[mask] = mu_c
                sigma_arr[mask] = sigma_c
            return mu_arr.astype(np.float64), sigma_arr.astype(np.float64)

        mu = np.percentile(gL, self.mu_percentile)
        p0 = np.percentile(gL, self.iqr_p0)
        p1 = np.percentile(gL, self.iqr_p1)
        sigma = self.iqr_scale * (p1 - p0) + self.eps
        mu_arr = np.full_like(gL, mu, dtype=np.float64)
        sigma_arr = np.full_like(gL, sigma, dtype=np.float64)
        return mu_arr, sigma_arr

    def _resolve_tau_p(self, dp: np.ndarray) -> float:
        if self.tau_p_mode == "fixed":
            return max(self.tau_p_min, self.tau_p)
        if self.tau_p_mode == "percentile":
            values = np.abs(dp).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return self.tau_p_min
            tau = float(np.percentile(values, self.tau_p_percentile))
            return max(self.tau_p_min, tau)
        raise ValueError(f"Unsupported tau_p_mode: {self.tau_p_mode}")

    def compute(self, proxy_logs: Optional[dict[str, np.ndarray]] = None) -> InformativenessResult:
        data = proxy_logs if proxy_logs is not None else np.load(self.npz_path)
        logits, labels, indices = self._load_logits(data)

        num_epochs = logits.shape[0]
        _, late_slice, window = resolve_early_late_slices(
            num_epochs, ratio=self.early_late_ratio, min_epochs=5, skip_first=True
        )
        early_epochs = window
        late_epochs = window

        probs = self._softmax(logits)
        label_idx = labels.reshape(1, -1, 1)
        p_true = np.take_along_axis(probs, label_idx, axis=2).squeeze(-1)

        probs_other = probs.copy()
        probs_other[:, np.arange(p_true.shape[1]), labels] = -np.inf
        p_other_max = probs_other.max(axis=2)
        gap = p_true - p_other_max

        gL = gap[late_slice].mean(axis=0)
        mu_arr, sigma_arr = self._compute_mu_sigma(gL, labels)
        w = np.exp(-0.5 * ((gap - mu_arr[None, :]) / (sigma_arr[None, :] + self.eps)) ** 2)

        dp = gap[1:] - gap[:-1]
        tau_p = self._resolve_tau_p(dp)
        scaled_dp = dp / tau_p
        softplus = np.log1p(np.exp(-np.abs(scaled_dp))) + np.maximum(scaled_dp, 0.0)
        delta_pos = softplus * tau_p

        raw_score = (w[:-1] * delta_pos).mean(axis=0).astype(np.float32)

        scores = quantile_minmax_by_class(
            raw_score, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            gL = gL[order]
            mu_arr = mu_arr[order]
            sigma_arr = sigma_arr[order]
            delta_pos = delta_pos[:, order]
            raw_score = raw_score[order]
            indices = indices[order]
            labels = labels[order]

        return InformativenessResult(
            scores=scores.astype(np.float32),
            hardness=gL.astype(np.float32),
            # hardness stores gL: late-window mean gap (boundary location proxy).
            delta_gap=delta_pos.mean(axis=0).astype(np.float32),
            raw_score=raw_score,
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
            late_epochs=late_epochs,
        )


__all__ = ["InformativenessResult", "InformativenessScore"]
