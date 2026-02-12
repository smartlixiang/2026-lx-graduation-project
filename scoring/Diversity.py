"""多样性覆盖度（Div）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

import numpy as np

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

    def _knn_mean_distance_to_reference(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
        k: float,
        query_indices: torch.Tensor | None = None,
        reference_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_query = query_features.shape[0]
        num_reference = reference_features.shape[0]
        if num_query == 0:
            return torch.zeros(0, device=query_features.device)
        if num_reference == 0:
            return torch.zeros(num_query, device=query_features.device)

        effective_k = min(self._resolve_k(k, max(2, num_reference)), num_reference)
        distances = torch.empty(num_query, device=query_features.device)

        for start in range(0, num_query, self.chunk_size):
            end = min(start + self.chunk_size, num_query)
            chunk = query_features[start:end]
            dists = 2.0 - 2.0 * (chunk @ reference_features.T)

            if query_indices is not None and reference_indices is not None:
                chunk_indices = query_indices[start:end]
                same_mask = chunk_indices.unsqueeze(1) == reference_indices.unsqueeze(0)
                dists = dists.masked_fill(same_mask, float("inf"))

            nearest = torch.topk(dists, k=effective_k, largest=False).values
            finite_mask = torch.isfinite(nearest)
            safe_nearest = torch.where(finite_mask, nearest, torch.zeros_like(nearest))
            counts = finite_mask.sum(dim=1).clamp(min=1)
            mean_sq = safe_nearest.sum(dim=1) / counts
            distances[start:end] = torch.sqrt(torch.clamp(mean_sq, min=0.0))

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

    def score_dataset_dynamic(
        self,
        dataloader: DataLoader,
        adapter: AdapterMLP | None = None,
        selected_mask: np.ndarray | None = None,
        image_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> DivResult:
        if adapter is not None:
            adapter.eval()

        if image_features is None or labels is None:
            image_features, labels = self._encode_images(dataloader, adapter)
        else:
            image_features = image_features.to(self.device)
            labels = labels.to(self.device)

        if selected_mask is None:
            return self.score_dataset(dataloader, adapter)

        if selected_mask.shape[0] != labels.shape[0]:
            raise ValueError("selected_mask 长度必须与数据集样本数一致。")

        selected_tensor = torch.as_tensor(selected_mask, dtype=torch.bool, device=labels.device)
        sample_indices = torch.arange(labels.shape[0], device=labels.device)
        k_distances = torch.zeros(labels.shape[0], device=image_features.device)

        for class_idx in range(len(self.class_names)):
            class_mask = labels == class_idx
            if not class_mask.any():
                continue

            class_indices = sample_indices[class_mask]
            class_features = image_features[class_mask]

            reference_mask = class_mask & selected_tensor
            if reference_mask.any():
                reference_indices = sample_indices[reference_mask]
                reference_features = image_features[reference_mask]
            else:
                reference_indices = class_indices
                reference_features = class_features

            class_distances = self._knn_mean_distance_to_reference(
                query_features=class_features,
                reference_features=reference_features,
                k=self.k,
                query_indices=class_indices,
                reference_indices=reference_indices,
            )
            k_distances[class_mask] = class_distances

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
