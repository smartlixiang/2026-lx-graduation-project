"""Dataset name registration for consistent usage across the project."""
from __future__ import annotations

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

AVAILABLE_DATASETS = (CIFAR10, CIFAR100)

__all__ = ["CIFAR10", "CIFAR100", "AVAILABLE_DATASETS"]
