"""Model registry utilities."""
from __future__ import annotations

from collections.abc import Callable

from torch import nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Register a model factory under a name."""

    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' is already registered")
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def get_model(name: str) -> Callable[..., nn.Module]:
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


__all__ = ["MODEL_REGISTRY", "register_model", "get_model"]
