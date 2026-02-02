"""RiskScore (R) implementation based on proxy training dynamics."""

# Key interfaces:
# - RiskScore.compute() -> RiskResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import (
    quantile_minmax_by_class,
    resolve_window_length,
    robust_z_by_class,
    stable_sigmoid,
)


@dataclass
class RiskResult:
    """Container for RiskScore outputs."""

    scores: np.ndarray
    risk_z: np.ndarray
    raw_score: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    late_epochs: int


class RiskScore:
    """Compute risk score R from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        q_low: float = 0.01,
        q_high: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)

    def _load_loss(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "loss" not in data:
            raise ValueError("proxy log must include 'loss' for RiskScore.")
        if "labels" not in data:
            raise ValueError("labels are required to compute RiskScore.")
        loss = data["loss"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        indices = data["indices"] if "indices" in data else np.arange(loss.shape[1])
        if loss.ndim != 2:
            raise ValueError("loss array should have shape (epochs, num_samples).")
        if labels.ndim != 1:
            raise ValueError("labels array should have shape (num_samples,).")
        if labels.shape[0] != loss.shape[1]:
            raise ValueError("labels length must match number of samples.")
        if indices.shape[0] != loss.shape[1]:
            raise ValueError("indices length must match number of samples.")
        return loss, labels, indices

    def compute(self, proxy_logs: Optional[dict[str, np.ndarray]] = None) -> RiskResult:
        data = proxy_logs if proxy_logs is not None else np.load(self.npz_path)
        loss, labels, indices = self._load_loss(data)

        num_epochs = loss.shape[0]
        late_epochs = resolve_window_length(num_epochs, ratio=0.2, min_epochs=5)
        late_log = np.log1p(loss[-late_epochs:])
        r_value = late_log.mean(axis=0)
        risk_z = robust_z_by_class(r_value, labels, eps=self.eps)
        raw_score = stable_sigmoid(risk_z)

        scores = quantile_minmax_by_class(
            raw_score, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            risk_z = risk_z[order]
            raw_score = raw_score[order]
            indices = indices[order]
            labels = labels[order]

        return RiskResult(
            scores=scores.astype(np.float32),
            risk_z=risk_z.astype(np.float32),
            raw_score=raw_score.astype(np.float32),
            labels=labels,
            indices=indices,
            late_epochs=late_epochs,
        )


__all__ = ["RiskResult", "RiskScore"]
