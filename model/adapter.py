"""Adapter MLP for CLIP feature alignment and a lightweight CLIP wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip  # type: ignore

from utils.global_config import CONFIG


class AdapterMLP(nn.Module):
    """Two-layer MLP adapter with output L2 normalization."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
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
    tokenizer: Callable[[Sequence[str] | Iterable[str]], torch.Tensor] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - device binding
        if self.device is None:
            self.device = CONFIG.global_device
        else:
            self.device = torch.device(self.device)
        self._load_model_and_preprocess()
        self.model.eval()

    @property
    def embed_dim(self) -> int:
        return self.model.visual.output_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images and return L2-normalized features."""

        with torch.no_grad():
            image_features = self.model.encode_image(images.to(self.device))
            image_features = image_features.float()
            return F.normalize(image_features, dim=-1)

    def encode_text(self, texts: Sequence[str] | Iterable[str]) -> torch.Tensor:
        """Encode texts with CLIP and return L2-normalized features."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized; call after __post_init__ executes.")

        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features.float()
            return F.normalize(text_features, dim=-1)

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CLIP's preprocess transform to a batch of raw images."""

        return torch.stack([self.preprocess(img) for img in images])

    def to(self, device: torch.device | str) -> "CLIPFeatureExtractor":  # pragma: no cover - delegating
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def _load_model_and_preprocess(self) -> None:
        """Load CLIP model + preprocess via the official `clip` package."""

        cache_dir = CONFIG.ensure_pretrained_clip_dir()
        self.model, self.preprocess = clip.load(
            self.model_name, device=self.device, download_root=str(cache_dir)
        )
        self.tokenizer = clip.tokenize


def resolve_adapter_dir(dataset_name: str, seed: int) -> Path:
    """Resolve adapter directory for a dataset/seed pair."""

    dataset_dir = CONFIG.ensure_adapter_dir(dataset_name)
    seed_dir = dataset_dir / str(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def resolve_adapter_paths(
    dataset_name: str,
    seed: int,
    adapter_image_path: str | Path | None = None,
    adapter_text_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve adapter image/text weight paths with dataset/seed defaults."""

    base_dir = resolve_adapter_dir(dataset_name, seed)
    image_path = Path(adapter_image_path) if adapter_image_path else base_dir / "adapter_image.pt"
    text_path = Path(adapter_text_path) if adapter_text_path else base_dir / "adapter_context.pt"
    return image_path, text_path


def _load_adapter_from_path(
    path: Path,
    input_dim: int,
    hidden_dim: int | None = None,
    map_location: torch.device | str | None = None,
) -> AdapterMLP:
    if hidden_dim is None:
        meta_path = path.parent / "meta.json"
        if meta_path.exists():
            try:
                import json

                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                hidden_dim = int(meta.get("hidden_dim", 256))
            except (ValueError, TypeError, json.JSONDecodeError):  # pragma: no cover - best effort
                hidden_dim = 256
        else:
            hidden_dim = 256

    adapter = AdapterMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    state_dict = torch.load(path, map_location=map_location)
    adapter.load_state_dict(state_dict)
    return adapter


def load_trained_adapters(
    dataset_name: str,
    clip_model: str,
    input_dim: int,
    seed: int,
    hidden_dim: int | None = None,
    map_location: torch.device | str | None = None,
    adapter_image_path: str | Path | None = None,
    adapter_text_path: str | Path | None = None,
) -> tuple[AdapterMLP, AdapterMLP, dict[str, Path]]:
    """根据数据集名称/随机种子加载图像与文本 Adapter。"""

    image_path, text_path = resolve_adapter_paths(
        dataset_name, seed, adapter_image_path, adapter_text_path
    )
    if not image_path.exists():
        raise FileNotFoundError(f"未找到图像 adapter 权重: {image_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"未找到文本 adapter 权重: {text_path}")

    meta_path = image_path.parent / "meta.json"
    if meta_path.exists():
        try:
            import json

            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_clip = meta.get("clip_model")
            if meta_clip and meta_clip != clip_model:
                raise ValueError(
                    f"Adapter clip_model={meta_clip} 与当前 clip_model={clip_model} 不一致。"
                )
        except json.JSONDecodeError:  # pragma: no cover - meta optional
            pass

    image_adapter = _load_adapter_from_path(
        image_path, input_dim=input_dim, hidden_dim=hidden_dim, map_location=map_location
    )
    text_adapter = _load_adapter_from_path(
        text_path, input_dim=input_dim, hidden_dim=hidden_dim, map_location=map_location
    )
    return image_adapter, text_adapter, {
        "image_path": image_path,
        "text_path": text_path,
        "meta_path": meta_path,
    }


__all__ = [
    "AdapterMLP",
    "CLIPFeatureExtractor",
    "load_trained_adapters",
    "resolve_adapter_dir",
    "resolve_adapter_paths",
]
