"""Dataset and dataloader utilities.

This module provides a registry-driven interface for loading common datasets
with configurable train/val/test splits. The abstractions focus on class-based
APIs so adding a new dataset requires subclassing :class:`BaseDataset` and
registering it with :func:`register_dataset`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Type

if importlib.util.find_spec("torch") is None:
    raise ImportError("PyTorch 未安装，请先执行 `pip install torch torchvision` 后再使用数据集模块。")

import torch
from torch.utils.data import DataLoader, Dataset, random_split

if importlib.util.find_spec("torchvision") is None:
    raise ImportError("torchvision 未安装，请先执行 `pip install torchvision` 后再使用数据集模块。")

from torchvision import datasets

from utils.global_config import CONFIG as GLOBAL_CFG
from utils.normalizer import NORMALIZER


DATASET_REGISTRY: Dict[str, Type["BaseDataset"]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset implementation.

    Args:
        name: Name used to retrieve the dataset class via :class:`BaseDataLoader`.
    """

    def decorator(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class SplitConfig:
    """Configuration for dataset splits and loading."""

    data_path: Path = GLOBAL_CFG.data_root
    batch_size: int = GLOBAL_CFG.default_batch_size
    num_workers: int = GLOBAL_CFG.num_workers
    val_split: float = 0.1
    download: bool = True
    augment: bool = True
    normalize: bool = True
    seed: int = GLOBAL_CFG.global_seed
    pin_memory: bool = GLOBAL_CFG.pin_memory


class BaseDataset(ABC):
    """Abstract dataset definition with train/val/test support."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self.config.data_path.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset used for registry lookup and transforms."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of target classes in the dataset."""

    @abstractmethod
    def _build_train_set(self) -> Dataset:
        """Return the raw training dataset (without validation split)."""

    @abstractmethod
    def _build_test_set(self) -> Dataset:
        """Return the evaluation dataset."""

    def _split_train_val(self, train_set: Dataset) -> Tuple[Dataset, Dataset]:
        """Split the training dataset into train/validation subsets."""

        if self.config.val_split <= 0:
            return train_set, DatasetSubset(train_set, [])

        val_size = int(len(train_set) * self.config.val_split)
        train_size = len(train_set) - val_size
        generator = torch.Generator().manual_seed(self.config.seed)
        return tuple(random_split(train_set, [train_size, val_size], generator=generator))  # type: ignore[return-value]

    def build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Construct dataloaders for train, val, and test splits."""

        train_set = self._build_train_set()
        test_set = self._build_test_set()
        train_subset, val_subset = self._split_train_val(train_set)

        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }

        train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader


class DatasetSubset(Dataset):
    """A lightweight subset wrapper that exposes length/indexing."""

    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self.indices)

    def __getitem__(self, idx: int):  # pragma: no cover - trivial proxy
        return self.dataset[self.indices[idx]]


@register_dataset("cifar10")
class Cifar10Dataset(BaseDataset):
    _dataset_name = "cifar10"
    _num_classes = 10

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_train_set(self) -> Dataset:
        transform = NORMALIZER.build_train_transforms(
            self.dataset_name,
            normalize=self.config.normalize,
            augment=self.config.augment,
        )
        return datasets.CIFAR10(
            root=str(self.config.data_path),
            train=True,
            download=self.config.download,
            transform=transform,
        )

    def _build_test_set(self) -> Dataset:
        transform = NORMALIZER.build_eval_transforms(
            self.dataset_name,
            normalize=self.config.normalize,
        )
        return datasets.CIFAR10(
            root=str(self.config.data_path),
            train=False,
            download=self.config.download,
            transform=transform,
        )


@register_dataset("cifar100")
class Cifar100Dataset(BaseDataset):
    _dataset_name = "cifar100"
    _num_classes = 100

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_train_set(self) -> Dataset:
        transform = NORMALIZER.build_train_transforms(
            self.dataset_name,
            normalize=self.config.normalize,
            augment=self.config.augment,
        )
        return datasets.CIFAR100(
            root=str(self.config.data_path),
            train=True,
            download=self.config.download,
            transform=transform,
        )

    def _build_test_set(self) -> Dataset:
        transform = NORMALIZER.build_eval_transforms(
            self.dataset_name,
            normalize=self.config.normalize,
        )
        return datasets.CIFAR100(
            root=str(self.config.data_path),
            train=False,
            download=self.config.download,
            transform=transform,
        )


class BaseDataLoader:
    """Factory-driven data loader for registered datasets.

    Example::

        loader = BaseDataLoader("cifar10", data_path=Path("./data"))
        train_loader, val_loader, test_loader = loader.load()
    """

    def __init__(
        self,
        dataset: str,
        data_path: str | Path | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        val_split: float = 0.1,
        download: bool = True,
        augment: bool = True,
        normalize: bool = True,
        seed: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        if dataset not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{dataset}' is not registered. Available: {list(DATASET_REGISTRY.keys())}")

        config = SplitConfig(
            data_path=Path(data_path) if data_path is not None else GLOBAL_CFG.ensure_data_dir(),
            batch_size=batch_size if batch_size is not None else GLOBAL_CFG.default_batch_size,
            num_workers=num_workers if num_workers is not None else GLOBAL_CFG.num_workers,
            val_split=val_split,
            download=download,
            augment=augment,
            normalize=normalize,
            seed=seed if seed is not None else GLOBAL_CFG.global_seed,
            pin_memory=pin_memory if pin_memory is not None else GLOBAL_CFG.pin_memory,
        )

        dataset_cls = DATASET_REGISTRY[dataset]
        self.dataset: BaseDataset = dataset_cls(config)

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def load(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Instantiate dataloaders for the configured dataset."""

        return self.dataset.build_loaders()

    @staticmethod
    def available_datasets() -> Iterable[str]:
        return DATASET_REGISTRY.keys()


__all__ = [
    "BaseDataset",
    "BaseDataLoader",
    "Cifar10Dataset",
    "Cifar100Dataset",
    "DATASET_REGISTRY",
    "DatasetSubset",
    "register_dataset",
    "SplitConfig",
]
