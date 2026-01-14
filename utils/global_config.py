"""集中管理全局可复用的实验超参数与路径配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class GlobalConfig:
    """项目级别的默认配置。"""

    data_root: Path = Path("./data")
    adapter_weights: Path = Path("./adapter_weights")
    pretrained_clip: Path = Path("./pretrained_clip")
    # 实验默认随机种子列表：exp_seeds=[22, 42, 96]
    exp_seeds: list[int] = field(default_factory=lambda: [22, 42, 96])
    global_seed: int = 42
    default_batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    image_size: int = 32
    global_device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def ensure_data_dir(self) -> Path:
        """确保数据目录存在并返回路径。"""

        self.data_root.mkdir(parents=True, exist_ok=True)
        return self.data_root

    def ensure_adapter_dir(self, dataset_name: str | None = None) -> Path:
        """确保 adapter 权重目录存在，可选按数据集划分子目录。"""

        self.adapter_weights.mkdir(parents=True, exist_ok=True)
        if dataset_name is None:
            return self.adapter_weights

        dataset_dir = self.adapter_weights / dataset_name.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def ensure_pretrained_clip_dir(self) -> Path:
        """确保本地 CLIP 预训练模型缓存目录存在并返回路径。"""

        self.pretrained_clip.mkdir(parents=True, exist_ok=True)
        return self.pretrained_clip


CONFIG = GlobalConfig()

__all__ = ["CONFIG", "GlobalConfig"]
