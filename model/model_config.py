"""Model registry utilities."""
from __future__ import annotations

from collections.abc import Callable

from torch import nn

from model.resnet import resnet18, resnet50

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}


def get_model(name: str) -> Callable[..., nn.Module]:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


__all__ = ["MODEL_REGISTRY", "get_model"]
