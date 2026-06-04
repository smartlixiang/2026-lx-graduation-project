"""Swin Transformer model factories."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import swin_t as torchvision_swin_t


class _SmallInputSafeSwin(nn.Module):
    """Wrap torchvision Swin to avoid invalid spatial sizes on very small inputs."""

    def __init__(self, model: nn.Module, min_size: int = 32) -> None:
        super().__init__()
        self.model = model
        self.min_size = min_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height < self.min_size or width < self.min_size:
            target_height = max(height, self.min_size)
            target_width = max(width, self.min_size)
            x = F.interpolate(
                x,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        return self.model(x)


def swin_t(num_classes: int = 10) -> nn.Module:
    """Create a randomly initialized Swin-Tiny classifier.

    The model does not load ImageNet weights. It accepts standard image tensors with
    shape ``[B, 3, H, W]`` and returns logits with shape ``[B, num_classes]``.
    """

    model = torchvision_swin_t(weights=None, num_classes=num_classes)
    return _SmallInputSafeSwin(model)
