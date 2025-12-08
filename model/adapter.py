"""Adapter MLP for CLIP feature alignment and a lightweight CLIP wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'clip' package is required. Install via `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc


class AdapterMLP(nn.Module):
    """Two-layer MLP adapter with output L2 normalization."""

    def __init__(self, input_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


@dataclass
class CLIPFeatureExtractor:
    """Thin wrapper around the CLIP model for feature extraction."""

    model_name: str = "ViT-B/32"
    device: torch.device | None = None

    def __post_init__(self) -> None:  # pragma: no cover - device binding
        self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    @property
    def embed_dim(self) -> int:
        return self.model.visual.output_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images and return L2-normalized features."""

        with torch.no_grad():
            image_features = self.model.encode_image(images.to(self.device))
            return F.normalize(image_features, dim=-1)

    def encode_text(self, texts: Sequence[str] | Iterable[str]) -> torch.Tensor:
        """Encode texts with CLIP and return L2-normalized features."""

        tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            return F.normalize(text_features, dim=-1)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CLIP's preprocess transform to a batch of raw images."""

        return torch.stack([self.preprocess(img) for img in images])

    def to(self, device: torch.device | str) -> "CLIPFeatureExtractor":  # pragma: no cover - delegating
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self


__all__ = ["AdapterMLP", "CLIPFeatureExtractor"]
