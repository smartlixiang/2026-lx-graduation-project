"""多样性覆盖度（Div）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from model.adapter import AdapterMLP, CLIPFeatureExtractor
from utils.global_config import CONFIG


@dataclass
class DivResult:
    """Div 计算结果容器。"""

    scores: torch.Tensor
    labels: torch.Tensor
    image_features: torch.Tensor
    k_distances: torch.Tensor
    class_names: List[str]
    k: float

    def classwise_mean(self) -> List[float]:
        """按类别返回平均 Div 分值。"""

        means = []
        for class_idx in range(len(self.class_names)):
            mask = self.labels == class_idx
            if mask.any():
                means.append(self.scores[mask].mean().item())
            else:  # pragma: no cover - 防御性分支
                means.append(float("nan"))
        return means


class Div:
    """计算图像样本的多样性覆盖度 (Div)。

    使用类内前 k 个近邻距离均值作为原始度量，并进行类内分位点归一化。
    """

    def __init__(
        self,
        class_names: Sequence[str],
        k: float = 0.05,
        clip_model: str = "ViT-B/32",
        device: torch.device | None = None,
        chunk_size: int = 1024,
    ) -> None:
        if k <= 0:
            raise ValueError("k 必须为正数。")
        self.class_names = [str(name) for name in class_names]
        self.k = float(k)
        self.chunk_size = chunk_size
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

    @staticmethod
    def _resolve_k(k: float, num_samples: int) -> int:
        if num_samples <= 1:
            return 1
        if float(k).is_integer():
            return max(1, min(int(k), num_samples - 1))
        if 0 < k < 1:
            return max(1, min(int(math.ceil(k * num_samples)), num_samples - 1))
        raise ValueError("当 k 为小数时，需满足 0 < k < 1，用于表示类别内样本比例。")

    def _knn_mean_distance(self, features: torch.Tensor, k: float) -> torch.Tensor:
        num_samples = features.shape[0]
        if num_samples <= 1:
            return torch.zeros(num_samples, device=features.device)

        effective_k = self._resolve_k(k, num_samples)
        distances = torch.empty(num_samples, device=features.device)

        all_features = features
        total = all_features.shape[0]
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            chunk = all_features[start:end]
            sims = chunk @ all_features.T
            dists = 2.0 - 2.0 * sims

            row_indices = torch.arange(start, end, device=features.device)
            local_indices = torch.arange(end - start, device=features.device)
            dists[local_indices, row_indices] = float("inf")

            nearest = torch.topk(dists, k=effective_k, largest=False).values
            nearest = torch.sqrt(torch.clamp(nearest, min=0.0))
            distances[start:end] = nearest.mean(dim=1)

        return distances

    @staticmethod
    def _quantile_normalize(
        values: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        low_q: float = 0.002,
        high_q: float = 0.998,
    ) -> torch.Tensor:
        scores = torch.zeros_like(values)
        for class_idx in range(num_classes):
            mask = labels == class_idx
            if not mask.any():
                continue
            class_values = values[mask]
            q_low = torch.quantile(class_values, low_q)
            q_high = torch.quantile(class_values, high_q)
            if torch.isclose(q_low, q_high):
                scores[mask] = 0.5
                continue
            scaled = (class_values - q_low) / (q_high - q_low)
            scores[mask] = torch.clamp(scaled, 0.0, 1.0)
        return scores

    def score_dataset(
        self, dataloader: DataLoader, adapter: AdapterMLP | None = None
    ) -> DivResult:
        if adapter is not None:
            adapter.eval()

        image_features, labels = self._encode_images(dataloader, adapter)
        k_distances = torch.zeros(labels.shape[0], device=image_features.device)

        for class_idx in range(len(self.class_names)):
            mask = labels == class_idx
            if not mask.any():
                continue
            class_features = image_features[mask]
            class_distances = self._knn_mean_distance(class_features, self.k)
            k_distances[mask] = class_distances

        scores = self._quantile_normalize(k_distances, labels, len(self.class_names))

        return DivResult(
            scores=scores.detach().cpu(),
            labels=labels.detach().cpu(),
            image_features=image_features.detach().cpu(),
            k_distances=k_distances.detach().cpu(),
            class_names=self.class_names,
            k=self.k,
        )


__all__ = ["Div", "DivResult"]
