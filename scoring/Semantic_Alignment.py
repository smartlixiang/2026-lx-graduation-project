"""语义对齐度（Semantic Alignment, SA）评分实现。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader

from model.adapter import AdapterMLP, CLIPFeatureExtractor


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
        self.extractor = CLIPFeatureExtractor(model_name=clip_model, device=device)

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

        for images, batch_labels in dataloader:
            image_features = self.extractor.encode_image(images)
            if adapter is not None:
                image_features = adapter(image_features.to(adapter_device))
            feats.append(image_features.cpu())
            labels.append(batch_labels)

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

    def score_dataset(
        self, dataloader: DataLoader, adapter: AdapterMLP | None = None
    ) -> SAResult:
        text_features = self._build_text_features()

        if adapter is not None:
            adapter.eval()

        image_features, labels = self._encode_images(dataloader, adapter)
        scores = self._margin_similarity(image_features, text_features, labels)

        return SAResult(
            scores=scores.cpu(),
            labels=labels.cpu(),
            image_features=image_features.cpu(),
            text_features=text_features.cpu(),
            class_names=self.class_names,
        )


__all__ = ["SAResult", "SemanticAlignment"]
