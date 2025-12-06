"""Normalization helpers for common vision datasets."""
from __future__ import annotations

from typing import Iterable, Tuple

from torchvision import transforms

# Statistics sourced from widely-used torchvision references
CIFAR10_MEAN_STD: Tuple[Iterable[float], Iterable[float]] = (
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616),
)
CIFAR100_MEAN_STD: Tuple[Iterable[float], Iterable[float]] = (
    (0.5071, 0.4865, 0.4409),
    (0.2673, 0.2564, 0.2761),
)


def get_normalization(mean: Iterable[float], std: Iterable[float]) -> transforms.Normalize:
    """Create a normalization transform."""

    return transforms.Normalize(mean=mean, std=std)


def get_cifar_normalization(dataset: str) -> transforms.Normalize:
    """Return normalization transform for CIFAR dataset variants."""

    key = dataset.lower()
    if key in {"cifar10", "cifar-10"}:
        mean, std = CIFAR10_MEAN_STD
    elif key in {"cifar100", "cifar-100"}:
        mean, std = CIFAR100_MEAN_STD
    else:
        raise ValueError(f"Unsupported CIFAR dataset '{dataset}'.")
    return get_normalization(mean, std)
