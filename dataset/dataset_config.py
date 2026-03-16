"""Dataset name registration for consistent usage across the project."""
from __future__ import annotations

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
TINY_IMAGENET = "tiny_imagenet"

AVAILABLE_DATASETS = (CIFAR10, CIFAR100, TINY_IMAGENET)

__all__ = ["CIFAR10", "CIFAR100", "TINY_IMAGENET", "AVAILABLE_DATASETS"]
