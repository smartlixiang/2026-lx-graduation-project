"""RiskScore (R) implementation based on proxy training dynamics."""

# Key interfaces:
# - RiskScore.compute() -> RiskResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import resolve_early_late_slices, robust_z_by_class


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
        lambda_improve: float = 0.7,
        tail_q0: float = 0.95,
        tail_q1: float = 0.995,
        early_late_ratio: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.lambda_improve = float(lambda_improve)
        self.tail_q0 = float(tail_q0)
        self.tail_q1 = float(tail_q1)
        self.early_late_ratio = float(early_late_ratio)
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
        early_slice, late_slice, window = resolve_early_late_slices(
            num_epochs, ratio=self.early_late_ratio, min_epochs=5, skip_first=True
        )
        early_epochs = window
        late_epochs = window
        early_log = np.log1p(loss[early_slice])
        late_log = np.log1p(loss[late_slice])
        early_value = early_log.mean(axis=0)
        late_value = late_log.mean(axis=0)
        improve_value = early_value - late_value

        late_z = robust_z_by_class(late_value, labels, eps=self.eps)
        improve_z = robust_z_by_class(improve_value, labels, eps=self.eps)
        risk_core = late_z - self.lambda_improve * improve_z
        raw_score = np.zeros_like(risk_core, dtype=np.float32)
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_vals = risk_core[cls_mask]
            if cls_vals.size == 0:
                continue
            t0 = np.quantile(cls_vals, self.tail_q0)
            t1 = np.quantile(cls_vals, self.tail_q1)
            if t1 <= t0 + self.eps:
                t1 = t0 + 1.0
            denom = t1 - t0 + self.eps
            raw_score[cls_mask] = np.clip((cls_vals - t0) / denom, 0.0, 1.0)

        scores = raw_score

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
