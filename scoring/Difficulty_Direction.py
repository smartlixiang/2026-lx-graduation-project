"""类内难度方向得分（Difficulty Direction Score, DDS）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from model.adapter import AdapterMLP, CLIPFeatureExtractor
from utils.global_config import CONFIG


@dataclass
class DDSResult:
    """DDS 计算结果容器。"""

    scores: torch.Tensor
    labels: torch.Tensor
    image_features: torch.Tensor
    class_names: List[str]
    k: float
    resolved_k: int

    def classwise_mean(self) -> List[float]:
        """按类别返回平均 DDS 分值。"""

        means = []
        for class_idx in range(len(self.class_names)):
            mask = self.labels == class_idx
            if mask.any():
                means.append(self.scores[mask].mean().item())
            else:  # pragma: no cover - 防御性分支
                means.append(float("nan"))
        return means


class DifficultyDirection:
    """计算图像样本的类内难度方向得分 (DDS)。"""

    def __init__(
        self,
        class_names: Sequence[str],
        k: float = 10,
        clip_model: str = "ViT-B/32",
        device: torch.device | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k 必须为正数。")
        self.class_names = [str(name) for name in class_names]
        self.k = float(k)
        self.device = torch.device(device) if device is not None else CONFIG.global_device
        self.extractor = CLIPFeatureExtractor(model_name=clip_model, device=self.device)

    def _encode_images(
        self, dataloader: DataLoader, adapter: AdapterMLP | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        adapter_device = None
        if adapter is not None:
            adapter_device = next(adapter.parameters()).device
        target_device = adapter_device or self.extractor.device

        for images, batch_labels in dataloader:
            image_features = self.extractor.encode_image(images)
            if adapter is not None:
                image_features = adapter(image_features.to(adapter_device))
            image_features = image_features.to(target_device)
            feats.append(image_features)
            labels.append(batch_labels.to(target_device))

        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    def _resolve_k(self, feature_dim: int) -> int:
        if float(self.k).is_integer():
            return max(1, min(int(self.k), feature_dim))
        if 0 < self.k < 1:
            return max(1, min(int(math.ceil(self.k * feature_dim)), feature_dim))
        raise ValueError("当 k 为小数时，需满足 0 < k < 1，用于表示方向比例。")

    @staticmethod
    def _dds_from_pca(
        class_features: torch.Tensor, low_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples, feat_dim = class_features.shape
        if num_samples <= 1:
            return torch.zeros(num_samples, device=class_features.device), torch.zeros(
                feat_dim, device=class_features.device
            )

        mean = class_features.mean(dim=0)
        centered = class_features - mean
        cov = centered.T @ centered / (num_samples - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        low_k = max(1, min(low_k, eigenvectors.shape[1]))
        low_dirs = eigenvectors[:, :low_k]
        projections = centered @ low_dirs
        scores = projections.abs().sum(dim=1)
        return scores, mean

    @staticmethod
    def _min_max_normalize(values: torch.Tensor) -> torch.Tensor:
        min_val = values.min()
        max_val = values.max()
        if torch.isclose(min_val, max_val):
            return torch.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def score_dataset(
        self, dataloader: DataLoader, adapter: AdapterMLP | None = None
    ) -> DDSResult:
        if adapter is not None:
            adapter.eval()

        image_features, labels = self._encode_images(dataloader, adapter)
        scores = torch.zeros(labels.shape[0], device=image_features.device)
        resolved_k = self._resolve_k(image_features.shape[1])

        for class_idx in range(len(self.class_names)):
            mask = labels == class_idx
            if not mask.any():
                continue
            class_features = image_features[mask]
            class_scores, _ = self._dds_from_pca(class_features, resolved_k)
            scores[mask] = class_scores

        scores = self._min_max_normalize(scores)

        return DDSResult(
            scores=scores.detach().cpu(),
            labels=labels.detach().cpu(),
            image_features=image_features.detach().cpu(),
            class_names=self.class_names,
            k=self.k,
            resolved_k=resolved_k,
        )


__all__ = ["DDSResult", "DifficultyDirection"]
