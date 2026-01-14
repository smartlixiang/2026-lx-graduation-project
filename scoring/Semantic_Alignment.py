"""语义对齐度（Semantic Alignment, SA）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from model.adapter import AdapterMLP, CLIPFeatureExtractor
from utils.global_config import CONFIG


@dataclass
class SAResult:
    """SA 计算结果容器。"""

    scores: torch.Tensor
    labels: torch.Tensor
    image_features: torch.Tensor
    text_features: torch.Tensor
    class_names: List[str]

    def classwise_mean(self) -> List[float]:
        """按类别返回平均 SA 分值。"""

        means = []
        for class_idx in range(len(self.class_names)):
            mask = self.labels == class_idx
            if mask.any():
                means.append(self.scores[mask].mean().item())
            else:  # pragma: no cover - 防御性分支
                means.append(float("nan"))
        return means


class SemanticAlignment:
    """计算图像样本的语义对齐度 (SA)。

    SA 定义为图像嵌入与对应类别文本嵌入的余弦相似度与最相近非目标类别之间的差值，
    分值越大表示样本语义越贴合其标签且与其他类别区分度越高。
    """

    def __init__(
        self,
        class_names: Sequence[str],
        clip_model: str = "ViT-B/32",
        prompt_template: str = "a photo of a {}",
        device: torch.device | None = None,
    ) -> None:
        self.class_names = [str(name) for name in class_names]
        self.prompt_template = prompt_template
        self.device = torch.device(device) if device is not None else CONFIG.global_device
        self.extractor = CLIPFeatureExtractor(model_name=clip_model, device=self.device)

    def _build_text_features(self) -> torch.Tensor:
        prompts = [self.prompt_template.format(name) for name in self.class_names]
        return self.extractor.encode_text(prompts)

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
    def _margin_similarity(
        image_features: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        sims = image_features @ text_features.T
        target_sim = sims.gather(1, labels.view(-1, 1)).squeeze(1)

        mask = torch.ones_like(sims, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), False)
        negative_max = sims.masked_fill(~mask, float("-inf")).max(dim=1).values
        return target_sim - negative_max

    @staticmethod
    def _quantile_normalize(
        values: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        low_q: float = 0.01,
        high_q: float = 0.99,
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
    ) -> SAResult:
        text_features = self._build_text_features().to(self.device)

        if adapter is not None:
            adapter.eval()

        image_features, labels = self._encode_images(dataloader, adapter)
        text_features = text_features.to(image_features.device)
        scores = self._margin_similarity(image_features, text_features, labels)
        scores = self._quantile_normalize(scores, labels, len(self.class_names))

        return SAResult(
            scores=scores.detach().cpu(),
            labels=labels.detach().cpu(),
            image_features=image_features.detach().cpu(),
            text_features=text_features.detach().cpu(),
            class_names=self.class_names,
        )


__all__ = ["SAResult", "SemanticAlignment"]
