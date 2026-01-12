"""可复用的标准化与变换构建器，采用类的形式便于扩展。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from importlib import import_module
from torchvision import transforms

# 通过模块导入全局配置
GLOBAL_CFG = import_module("utils.global_config").CONFIG
DATASET_CONFIG = import_module("dataset.dataset_config")


@dataclass
class DatasetStats:
    mean: Sequence[float]
    std: Sequence[float]


@dataclass
class Normalizer:
    """维护数据集统计并生成标准化/增强流水线。"""

    dataset_stats: Dict[str, DatasetStats] = field(
        default_factory=lambda: {
            DATASET_CONFIG.CIFAR10: DatasetStats(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
            DATASET_CONFIG.CIFAR100: DatasetStats(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        }
    )

    def _key(self, dataset_name: str) -> str:
        return dataset_name.lower()

    def register(self, name: str, mean: Sequence[float], std: Sequence[float]) -> None:
        """登记数据集的均值和方差。"""

        self.dataset_stats[self._key(name)] = DatasetStats(mean=list(mean), std=list(std))

    def stats(self, dataset_name: str) -> Tuple[List[float], List[float]]:
        """返回均值和标准差。"""

        key = self._key(dataset_name)
        if key not in self.dataset_stats:
            raise KeyError(f"Normalization parameters for '{dataset_name}' are not defined.")
        stats = self.dataset_stats[key]
        return list(stats.mean), list(stats.std)

    def train_tfms(
        self,
        dataset_name: str,
        normalize: bool = True,
        augment: bool = True,
        image_size: int | None = None,
    ) -> transforms.Compose:
        """训练阶段的增强与标准化流水线。"""

        ops: List[transforms.Transform] = []
        size = image_size or GLOBAL_CFG.image_size

        if augment:
            ops.extend(
                [
                    transforms.RandomCrop(size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        ops.append(transforms.ToTensor())

        if normalize:
            mean, std = self.stats(dataset_name)
            ops.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(ops)

    def eval_tfms(
        self, dataset_name: str, normalize: bool = True, image_size: int | None = None
    ) -> transforms.Compose:
        """验证/测试阶段的标准化流水线。"""

        _ = image_size  # 预留参数，当前评估阶段仅依赖 ToTensor/Normalize
        ops: List[transforms.Transform] = [transforms.ToTensor()]
        if normalize:
            mean, std = self.stats(dataset_name)
            ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)


NORMALIZER = Normalizer()

__all__ = ["DatasetStats", "Normalizer", "NORMALIZER"]
