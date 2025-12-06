"""可复用的标准化与变换构建器，采用类的形式便于扩展。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from importlib import import_module
from torchvision import transforms

# 避免与关键字冲突，使用 import_module 导入全局配置
GLOBAL_CFG = import_module("utils.global").CONFIG


@dataclass
class DatasetStats:
    mean: Sequence[float]
    std: Sequence[float]


@dataclass
class Normalizer:
    """维护数据集统计并生成标准化/增强流水线。"""

    dataset_stats: Dict[str, DatasetStats] = field(
        default_factory=lambda: {
            "cifar10": DatasetStats(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
            "cifar100": DatasetStats(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        }
    )

    def _key(self, dataset_name: str) -> str:
        return dataset_name.lower()

    def register_dataset_stats(self, name: str, mean: Sequence[float], std: Sequence[float]) -> None:
        """注册新的数据集统计信息。"""

        self.dataset_stats[self._key(name)] = DatasetStats(mean=list(mean), std=list(std))

    def get_normalization(self, dataset_name: str) -> Tuple[List[float], List[float]]:
        """返回指定数据集的均值和标准差。"""

        key = self._key(dataset_name)
        if key not in self.dataset_stats:
            raise KeyError(f"Normalization parameters for '{dataset_name}' are not defined.")
        stats = self.dataset_stats[key]
        return list(stats.mean), list(stats.std)

    def build_train_transforms(
        self,
        dataset_name: str,
        normalize: bool = True,
        augment: bool = True,
        image_size: int | None = None,
    ) -> transforms.Compose:
        """创建训练阶段的变换组合。"""

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
            mean, std = self.get_normalization(dataset_name)
            ops.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(ops)

    def build_eval_transforms(
        self, dataset_name: str, normalize: bool = True, image_size: int | None = None
    ) -> transforms.Compose:
        """创建验证/测试阶段的变换组合。"""

        _ = image_size  # 预留参数，当前评估阶段仅依赖 ToTensor/Normalize
        ops: List[transforms.Transform] = [transforms.ToTensor()]
        if normalize:
            mean, std = self.get_normalization(dataset_name)
            ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)


NORMALIZER = Normalizer()

__all__ = ["DatasetStats", "Normalizer", "NORMALIZER"]
