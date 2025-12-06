"""Dataset loaders for CIFAR datasets with reproducible splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from utils.normalizer import get_cifar_normalization


TransformBuilder = Callable[[bool], transforms.Compose]


@dataclass
class DatasetBundle:
    """Container for train/validation/test datasets."""

    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class DataLoaders:
    """Container for train/validation/test dataloaders."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def _build_cifar_transforms(dataset: str, augment: bool = True) -> TransformBuilder:
    """Create a transform builder for CIFAR datasets.

    Args:
        dataset: Dataset name (``"cifar10"`` or ``"cifar100"``).
        augment: Whether to include standard training augmentations.

    Returns:
        A callable that produces a :class:`~torchvision.transforms.Compose` given a
        boolean indicating training mode.
    """

    normalization = get_cifar_normalization(dataset)

    def builder(is_train: bool) -> transforms.Compose:
        common: Iterable[transforms.Transform] = [transforms.ToTensor(), normalization]

        if not is_train:
            return transforms.Compose(list(common))

        if not augment:
            return transforms.Compose(list(common))

        train_transforms: Iterable[transforms.Transform] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        return transforms.Compose([*train_transforms, *common])

    return builder


def _split_dataset(
    dataset: Dataset, val_split: int, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Split dataset into train and validation subsets deterministically."""

    generator = torch.Generator().manual_seed(seed)
    train_size = len(dataset) - val_split
    return random_split(dataset, [train_size, val_split], generator=generator)


def _default_collate(batch):
    """Default collate function exposed for extensibility."""

    return torch.utils.data.default_collate(batch)


def _create_cifar_datasets(
    dataset_cls: Callable[..., Dataset],
    root: str,
    val_split: int = 5000,
    seed: int = 42,
    transform_builder: Optional[TransformBuilder] = None,
) -> DatasetBundle:
    """Internal helper to build CIFAR datasets and deterministic split."""

    transform_builder = transform_builder or _build_cifar_transforms(dataset_cls.__name__.lower())

    train_full = dataset_cls(root=root, train=True, download=True, transform=transform_builder(True))
    test = dataset_cls(root=root, train=False, download=True, transform=transform_builder(False))
    train, val = _split_dataset(train_full, val_split=val_split, seed=seed)
    return DatasetBundle(train=train, val=val, test=test)


def create_cifar10_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: int = 5000,
    seed: int = 42,
    augment: bool = True,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoaders:
    """Create CIFAR-10 dataloaders with deterministic train/val split."""

    transform_builder = _build_cifar_transforms("cifar10", augment=augment)
    datasets_bundle = _create_cifar_datasets(
        datasets.CIFAR10,
        root=root,
        val_split=val_split,
        seed=seed,
        transform_builder=transform_builder,
    )

    collate_fn = collate_fn or _default_collate
    train_loader = DataLoader(
        datasets_bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets_bundle.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        datasets_bundle.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


def create_cifar100_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: int = 5000,
    seed: int = 42,
    augment: bool = True,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoaders:
    """Create CIFAR-100 dataloaders with deterministic train/val split."""

    transform_builder = _build_cifar_transforms("cifar100", augment=augment)
    datasets_bundle = _create_cifar_datasets(
        datasets.CIFAR100,
        root=root,
        val_split=val_split,
        seed=seed,
        transform_builder=transform_builder,
    )

    collate_fn = collate_fn or _default_collate
    train_loader = DataLoader(
        datasets_bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets_bundle.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        datasets_bundle.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


DATASET_REGISTRY: Dict[str, Callable[..., DataLoaders]] = {
    "cifar10": create_cifar10_loaders,
    "cifar-10": create_cifar10_loaders,
    "cifar100": create_cifar100_loaders,
    "cifar-100": create_cifar100_loaders,
}


def get_dataloaders(name: str, **kwargs) -> DataLoaders:
    """Factory method to obtain dataloaders for the requested dataset."""

    key = name.lower()
    if key not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unsupported dataset '{name}'. Available options: {supported}.")
    return DATASET_REGISTRY[key](**kwargs)


def list_datasets() -> Tuple[str, ...]:
    """List available dataset keys."""

    return tuple(sorted(DATASET_REGISTRY))
