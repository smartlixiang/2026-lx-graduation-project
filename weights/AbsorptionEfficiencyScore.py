"""AbsorptionEfficiencyScore (A) implementation based on proxy training dynamics."""

# Key interfaces:
# - AbsorptionEfficiencyScore.compute() -> AbsorptionEfficiencyResult with scores in [0, 1].
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from utils.score_utils import (
    quantile_minmax_by_class,
    resolve_early_late_slices,
    robust_z_by_class,
    stable_sigmoid,
)


@dataclass
class AbsorptionEfficiencyResult:
    """Container for AbsorptionEfficiencyScore outputs."""

    scores: np.ndarray
    level_z: np.ndarray
    progress_z: np.ndarray
    raw_score: np.ndarray
    labels: Optional[np.ndarray]
    indices: np.ndarray
    early_epochs: int


class AbsorptionEfficiencyScore:
    """Compute absorption efficiency score A from proxy training logs (.npz)."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        q_low: float = 0.002,
        q_high: float = 0.998,
        temp_progress: float = 2.0,
        sigma_level: float = 2.0,
        early_late_ratio: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.temp_progress = float(temp_progress)
        self.sigma_level = float(sigma_level)
        self.early_late_ratio = float(early_late_ratio)
        self.eps = float(eps)

    def _load_loss(self, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "loss" not in data:
            raise ValueError("proxy log must include 'loss' for AbsorptionEfficiencyScore.")
        if "labels" not in data:
            raise ValueError("labels are required to compute AbsorptionEfficiencyScore.")
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

    def compute(self, proxy_logs: Optional[dict[str, np.ndarray]] = None) -> AbsorptionEfficiencyResult:
        data = proxy_logs if proxy_logs is not None else np.load(self.npz_path)
        loss, labels, indices = self._load_loss(data)

        num_epochs = loss.shape[0]
        early_slice, _, early_epochs = resolve_early_late_slices(
            num_epochs, ratio=self.early_late_ratio, min_epochs=5, skip_first=True
        )

        early_log = np.log1p(loss[early_slice])
        level = early_log.mean(axis=0)
        if early_epochs == 1:
            progress = np.zeros_like(level)
        else:
            progress = early_log[0] - early_log[-1]

        level_z = robust_z_by_class(level, labels, eps=self.eps)
        progress_z = robust_z_by_class(progress, labels, eps=self.eps)

        speed = stable_sigmoid(progress_z / self.temp_progress)
        moderate = np.exp(-0.5 * ((level_z / self.sigma_level) ** 2)).astype(np.float32)
        raw_score = np.sqrt(np.clip(speed * moderate, 0.0, 1.0) + self.eps).astype(np.float32)

        scores = quantile_minmax_by_class(
            raw_score, labels, q_low=self.q_low, q_high=self.q_high, eps=self.eps
        )

        if not np.array_equal(indices, np.arange(len(indices))):
            order = np.argsort(indices)
            scores = scores[order]
            level_z = level_z[order]
            progress_z = progress_z[order]
            raw_score = raw_score[order]
            indices = indices[order]
            labels = labels[order]

        return AbsorptionEfficiencyResult(
            scores=scores.astype(np.float32),
            level_z=level_z.astype(np.float32),
            progress_z=progress_z.astype(np.float32),
            raw_score=raw_score.astype(np.float32),
            labels=labels,
            indices=indices,
            early_epochs=early_epochs,
        )


__all__ = ["AbsorptionEfficiencyResult", "AbsorptionEfficiencyScore"]
