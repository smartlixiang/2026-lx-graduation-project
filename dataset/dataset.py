"""Dataset and dataloader utilities.

This module provides a registry-driven interface for loading common datasets
with configurable train/val/test splits. The abstractions focus on class-based
APIs so adding a new dataset requires subclassing :class:`BaseDataset` and
registering it with :func:`register_dataset`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import importlib.util
import shutil
import uuid
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

from dataset.dataset_config import CIFAR10, CIFAR100, TINY_IMAGENET
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


@register_dataset(CIFAR10)
class Cifar10Dataset(BaseDataset):
    _dataset_name = CIFAR10
    _num_classes = 10

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_train_set(self) -> Dataset:
        transform = NORMALIZER.train_tfms(
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
        transform = NORMALIZER.eval_tfms(
            self.dataset_name,
            normalize=self.config.normalize,
        )
        return datasets.CIFAR10(
            root=str(self.config.data_path),
            train=False,
            download=self.config.download,
            transform=transform,
        )


@register_dataset(CIFAR100)
class Cifar100Dataset(BaseDataset):
    _dataset_name = CIFAR100
    _num_classes = 100

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _build_train_set(self) -> Dataset:
        transform = NORMALIZER.train_tfms(
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
        transform = NORMALIZER.eval_tfms(
            self.dataset_name,
            normalize=self.config.normalize,
        )
        return datasets.CIFAR100(
            root=str(self.config.data_path),
            train=False,
            download=self.config.download,
            transform=transform,
        )


@register_dataset(TINY_IMAGENET)
class TinyImageNetDataset(BaseDataset):
    _dataset_name = TINY_IMAGENET
    _num_classes = 200
    _image_size = 64

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def _dataset_dir(self) -> Path:
        return self.config.data_path / "tiny-imagenet-200"

    def _build_train_set(self) -> Dataset:
        transform = NORMALIZER.train_tfms(
            self.dataset_name,
            normalize=self.config.normalize,
            augment=self.config.augment,
            image_size=self._image_size,
        )
        train_root = self._dataset_dir() / "train"
        if not train_root.exists():
            raise FileNotFoundError(
                "tiny-imagenet train split not found. Expected directory: "
                f"{train_root}. Please prepare tiny-imagenet-200 under data root."
            )
        return datasets.ImageFolder(root=str(train_root), transform=transform)

    def _train_class_to_idx(self) -> Dict[str, int]:
        train_root = self._dataset_dir() / "train"
        train_folder = datasets.ImageFolder(root=str(train_root))
        return train_folder.class_to_idx

    def _is_reorganized_val_layout(self, val_root: Path, train_classes: set[str]) -> bool:
        class_dirs = sorted(path.name for path in val_root.iterdir() if path.is_dir())
        if len(class_dirs) != self._num_classes:
            return False
        return set(class_dirs) == train_classes

    def _validate_imagefolder_readable(self, root: Path) -> None:
        folder = datasets.ImageFolder(root=str(root))
        if len(folder.classes) != self._num_classes:
            raise RuntimeError(
                f"tiny-imagenet 验证集类别数异常，期望 {self._num_classes}，实际 {len(folder.classes)}: {root}"
            )

    def _parse_val_annotations(self, annotations_path: Path) -> list[tuple[str, str]]:
        mapping: list[tuple[str, str]] = []
        with annotations_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_no, row in enumerate(reader, start=1):
                if len(row) < 2:
                    raise ValueError(
                        f"val_annotations.txt 第 {row_no} 行格式错误（至少需要 image 与 wnid 两列）。"
                    )
                image_name, wnid = row[0].strip(), row[1].strip()
                if not image_name or not wnid:
                    raise ValueError(f"val_annotations.txt 第 {row_no} 行存在空字段。")
                mapping.append((image_name, wnid))
        if not mapping:
            raise ValueError("val_annotations.txt 为空，无法构建 tiny-imagenet 验证集。")
        return mapping

    def _validate_raw_val_layout(
        self,
        val_root: Path,
        train_class_to_idx: Dict[str, int],
    ) -> list[tuple[str, str]]:
        images_dir = val_root / "images"
        annotations_path = val_root / "val_annotations.txt"
        if not images_dir.is_dir() or not annotations_path.is_file():
            raise RuntimeError(
                "tiny-imagenet 的 val 既不是重排后的按类别目录结构，也不是合法 raw layout。"
            )

        annotations = self._parse_val_annotations(annotations_path)
        unknown_classes = sorted({wnid for _, wnid in annotations if wnid not in train_class_to_idx})
        if unknown_classes:
            raise RuntimeError(
                "val_annotations.txt 中存在无法与 train/class_to_idx 对齐的 wnid："
                f"{unknown_classes[:5]}{'...' if len(unknown_classes) > 5 else ''}"
            )

        missing_images = [image for image, _ in annotations if not (images_dir / image).is_file()]
        if missing_images:
            raise RuntimeError(
                "val/images 中缺少 val_annotations.txt 标注的图片，例如："
                f"{missing_images[:5]}{'...' if len(missing_images) > 5 else ''}"
            )
        return annotations

    def _reorganize_raw_val_layout(
        self,
        val_root: Path,
        train_classes: set[str],
        annotations: list[tuple[str, str]],
    ) -> None:
        images_dir = val_root / "images"
        tmp_root = val_root / f".val_reorg_tmp_{uuid.uuid4().hex}"
        tmp_root.mkdir(parents=True, exist_ok=False)

        try:
            for class_name in sorted(train_classes):
                (tmp_root / class_name).mkdir(parents=True, exist_ok=False)

            for image_name, wnid in annotations:
                src = images_dir / image_name
                dst = tmp_root / wnid / image_name
                shutil.copy2(src, dst)

            self._validate_imagefolder_readable(tmp_root)

            for class_name in sorted(train_classes):
                (tmp_root / class_name).replace(val_root / class_name)

            shutil.rmtree(images_dir, ignore_errors=True)
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            raise

    def _build_test_set(self) -> Dataset:
        transform = NORMALIZER.eval_tfms(
            self.dataset_name,
            normalize=self.config.normalize,
            image_size=self._image_size,
        )
        val_root = self._dataset_dir() / "val"
        if not val_root.exists():
            raise FileNotFoundError(
                "tiny-imagenet validation split not found. Expected directory: "
                f"{val_root}. Please prepare tiny-imagenet-200 under data root."
            )
        train_class_to_idx = self._train_class_to_idx()
        train_classes = set(train_class_to_idx.keys())

        if self._is_reorganized_val_layout(val_root, train_classes):
            print("[info] tiny-imagenet val 已是与 train 对齐的重排结构，直接用于最终评估。")
            self._validate_imagefolder_readable(val_root)
            return datasets.ImageFolder(root=str(val_root), transform=transform)

        annotations = self._validate_raw_val_layout(val_root, train_class_to_idx)
        print("[info] tiny-imagenet val 当前为 raw layout，准备自动重排为按类别目录结构。")
        self._reorganize_raw_val_layout(val_root, train_classes, annotations)

        if not self._is_reorganized_val_layout(val_root, train_classes):
            raise RuntimeError("tiny-imagenet val 自动重排后结构校验失败，请检查数据目录。")

        self._validate_imagefolder_readable(val_root)
        print("[info] tiny-imagenet val 自动重排完成并通过校验，将用于最终评估。")
        return datasets.ImageFolder(root=str(val_root), transform=transform)


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
    "TinyImageNetDataset",
    "DATASET_REGISTRY",
    "DatasetSubset",
    "register_dataset",
    "SplitConfig",
]
