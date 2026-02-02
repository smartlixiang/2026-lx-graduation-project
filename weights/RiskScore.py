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
    """Container for RiskScore outputs.

    Note: risk_z stores the risk_core z-combination (late_z - lambda_improve * improve_z).
    """

    scores: np.ndarray
    risk_z: np.ndarray
    raw_score: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    early_epochs: int
    late_epochs: int


class RiskScore:
    """Compute risk score R from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        q_low: float = 0.01,
        q_high: float = 0.99,
        lambda_improve: float = 0.7,
        temp: float = 2.5,
        eps: float = 1e-6,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.lambda_improve = float(lambda_improve)
        self.temp = float(temp)
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
        early_epochs = resolve_window_length(num_epochs, ratio=0.2, min_epochs=5)
        late_epochs = resolve_window_length(num_epochs, ratio=0.2, min_epochs=5)
        early_log = np.log1p(loss[:early_epochs])
        late_log = np.log1p(loss[-late_epochs:])
        early_value = early_log.mean(axis=0)
        late_value = late_log.mean(axis=0)
        improve_value = early_value - late_value

        late_z = robust_z_by_class(late_value, labels, eps=self.eps)
        improve_z = robust_z_by_class(improve_value, labels, eps=self.eps)
        risk_core = late_z - self.lambda_improve * improve_z
        raw_score = stable_sigmoid(risk_core / self.temp)

        scores = quantile_minmax_by_class(
            raw_score, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            risk_core = risk_core[order]
            raw_score = raw_score[order]
            indices = indices[order]
            labels = labels[order]

        return RiskResult(
            scores=scores.astype(np.float32),
            risk_z=risk_core.astype(np.float32),
            raw_score=raw_score.astype(np.float32),
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
            late_epochs=late_epochs,
        )


__all__ = ["RiskResult", "RiskScore"]
