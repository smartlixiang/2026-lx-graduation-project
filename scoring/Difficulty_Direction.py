"""类内难度方向得分（Difficulty Direction Score, DDS）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
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
    k: int
    eigval_lower_bound: float
    eigval_upper_bound: float
    pca_cov_reg: float

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
        k: int = 5,
        clip_model: str = "ViT-B/32",
        device: torch.device | None = None,
        eigval_lower_bound: float = 0.02,
        eigval_upper_bound: float = 0.2,
        pca_cov_reg: float = 1e-6,
    ) -> None:
        if k < 1:
            raise ValueError("k 必须为正整数，用于控制最少选择方向数。")
        if not (0 <= eigval_lower_bound < eigval_upper_bound <= 1):
            raise ValueError("需满足 0 <= eigval_lower_bound < eigval_upper_bound <= 1。")
        if pca_cov_reg < 0:
            raise ValueError("pca_cov_reg 需为非负数。")

        self.class_names = [str(name) for name in class_names]
        self.k = int(k)
        self.device = torch.device(device) if device is not None else CONFIG.global_device
        self.extractor = CLIPFeatureExtractor(model_name=clip_model, device=self.device)
        self.eigval_lower_bound = float(eigval_lower_bound)
        self.eigval_upper_bound = float(eigval_upper_bound)
        self.pca_cov_reg = float(pca_cov_reg)

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

    def _select_difficulty_dirs(
        self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor
    ) -> torch.Tensor:
        feat_dim = eigenvalues.shape[0]
        if feat_dim == 0:
            return eigenvectors[:, :0]

        eigvals = torch.clamp(eigenvalues, min=0.0)
        total = eigvals.sum()
        if total.item() <= 0:
            return eigenvectors[:, : min(self.k, feat_dim)]

        cumulative = torch.cumsum(eigvals, dim=0) / total

        if self.eigval_lower_bound > 0:
            below_lower = torch.nonzero(cumulative < self.eigval_lower_bound, as_tuple=False).flatten()
            if below_lower.numel() == 0:
                start = 1 if feat_dim > 1 else 0
            else:
                start = int(below_lower[-1].item()) + 1
        else:
            start = 0

        start = min(start, feat_dim - 1)
        upper_budget = self.eigval_upper_bound * total

        chosen_indices: list[int] = []
        chosen_sum = torch.zeros((), dtype=eigvals.dtype, device=eigvals.device)

        for idx in range(start, feat_dim):
            next_sum = chosen_sum + eigvals[idx]
            if next_sum <= upper_budget:
                chosen_indices.append(idx)
                chosen_sum = next_sum
            else:
                break

        idx = start + len(chosen_indices)
        while len(chosen_indices) < self.k and idx < feat_dim:
            next_sum = chosen_sum + eigvals[idx]
            if next_sum <= upper_budget:
                chosen_indices.append(idx)
                chosen_sum = next_sum
                idx += 1
                continue
            break

        if not chosen_indices:
            return eigenvectors[:, :0]

        index_tensor = torch.tensor(chosen_indices, device=eigenvectors.device)
        return eigenvectors.index_select(dim=1, index=index_tensor)

    def _dds_from_pca(self, class_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples, feat_dim = class_features.shape
        if num_samples <= 1:
            return torch.zeros(num_samples, device=class_features.device), torch.zeros(
                feat_dim, device=class_features.device
            )

        mean = class_features.mean(dim=0)
        centered = class_features - mean
        cov = centered.T @ centered / (num_samples - 1)
        cov = cov + self.pca_cov_reg * torch.eye(
            feat_dim, device=class_features.device, dtype=class_features.dtype
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        selected_dirs = self._select_difficulty_dirs(eigenvalues, eigenvectors)
        if selected_dirs.shape[1] == 0:
            return torch.zeros(num_samples, device=class_features.device), mean

        projections = centered @ selected_dirs
        scores = projections.abs().sum(dim=1)
        return scores, mean

    def _dds_from_reference_pca(
        self,
        class_features: torch.Tensor,
        reference_features: torch.Tensor,
    ) -> torch.Tensor:
        num_samples = class_features.shape[0]
        feat_dim = class_features.shape[1]
        ref_num = reference_features.shape[0]
        if num_samples == 0:
            return torch.zeros(0, device=class_features.device)
        if ref_num <= 1:
            return torch.zeros(num_samples, device=class_features.device)

        ref_mean = reference_features.mean(dim=0)
        ref_centered = reference_features - ref_mean
        cov = ref_centered.T @ ref_centered / (ref_num - 1)
        cov = cov + self.pca_cov_reg * torch.eye(
            feat_dim, device=class_features.device, dtype=class_features.dtype
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        selected_dirs = self._select_difficulty_dirs(eigenvalues, eigenvectors)
        if selected_dirs.shape[1] == 0:
            return torch.zeros(num_samples, device=class_features.device)

        centered = class_features - ref_mean
        projections = centered @ selected_dirs
        return projections.abs().sum(dim=1)

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
    ) -> DDSResult:
        if adapter is not None:
            adapter.eval()

        image_features, labels = self._encode_images(dataloader, adapter)
        scores = torch.zeros(labels.shape[0], device=image_features.device)

        for class_idx in range(len(self.class_names)):
            mask = labels == class_idx
            if not mask.any():
                continue
            class_features = image_features[mask]
            class_scores, _ = self._dds_from_pca(class_features)
            scores[mask] = class_scores

        scores = self._quantile_normalize(scores, labels, len(self.class_names))

        return DDSResult(
            scores=scores.detach().cpu(),
            labels=labels.detach().cpu(),
            image_features=image_features.detach().cpu(),
            class_names=self.class_names,
            k=self.k,
            eigval_lower_bound=self.eigval_lower_bound,
            eigval_upper_bound=self.eigval_upper_bound,
            pca_cov_reg=self.pca_cov_reg,
        )

    def score_dataset_dynamic(
        self,
        dataloader: DataLoader,
        adapter: AdapterMLP | None = None,
        selected_mask: np.ndarray | None = None,
        image_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> DDSResult:
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
        scores = torch.zeros(labels.shape[0], device=image_features.device)

        for class_idx in range(len(self.class_names)):
            class_mask = labels == class_idx
            if not class_mask.any():
                continue

            class_features = image_features[class_mask]
            ref_mask = class_mask & selected_tensor
            if ref_mask.any():
                ref_features = image_features[ref_mask]
            else:
                ref_features = class_features

            class_scores = self._dds_from_reference_pca(class_features, ref_features)
            scores[class_mask] = class_scores

        scores = self._quantile_normalize(scores, labels, len(self.class_names))
        return DDSResult(
            scores=scores.detach().cpu(),
            labels=labels.detach().cpu(),
            image_features=image_features.detach().cpu(),
            class_names=self.class_names,
            k=self.k,
            eigval_lower_bound=self.eigval_lower_bound,
            eigval_upper_bound=self.eigval_upper_bound,
            pca_cov_reg=self.pca_cov_reg,
        )


__all__ = ["DDSResult", "DifficultyDirection"]
