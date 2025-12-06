"""Reusable normalization and transform utilities."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from torchvision import transforms

DatasetStats = Dict[str, Sequence[float]]


DATASET_STATS: Dict[str, DatasetStats] = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "cifar100": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    # Synthetic dataset used for quick functional checks without downloads.
    "fake": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}


def get_normalization(dataset_name: str) -> Tuple[List[float], List[float]]:
    """Return mean and std for a dataset."""

    key = dataset_name.lower()
    if key not in DATASET_STATS:
        raise KeyError(f"Normalization parameters for '{dataset_name}' are not defined.")
    stats = DATASET_STATS[key]
    return list(stats["mean"]), list(stats["std"])


def build_train_transforms(dataset_name: str, normalize: bool = True, augment: bool = True, image_size: int = 32) -> transforms.Compose:
    """Create training transforms with optional augmentation and normalization."""

    ops: List[transforms.Transform] = []

    if augment:
        ops.extend(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

    ops.append(transforms.ToTensor())

    if normalize:
        mean, std = get_normalization(dataset_name)
        ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def build_eval_transforms(dataset_name: str, normalize: bool = True) -> transforms.Compose:
    """Create evaluation (validation/test) transforms."""

    ops = [transforms.ToTensor()]
    if normalize:
        mean, std = get_normalization(dataset_name)
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def register_dataset_stats(name: str, mean: Sequence[float], std: Sequence[float]) -> None:
    """Register normalization statistics for a new dataset."""

    DATASET_STATS[name.lower()] = {"mean": list(mean), "std": list(std)}


__all__ = [
    "DATASET_STATS",
    "build_eval_transforms",
    "build_train_transforms",
    "get_normalization",
    "register_dataset_stats",
]
