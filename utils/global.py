"""集中管理全局可复用的实验超参数与路径配置。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GlobalConfig:
    """项目级别的默认配置。"""

    data_root: Path = Path("./data")
    global_seed: int = 42
    default_batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    image_size: int = 32

    def ensure_data_dir(self) -> Path:
        """确保数据目录存在并返回路径。"""

        self.data_root.mkdir(parents=True, exist_ok=True)
        return self.data_root


CONFIG = GlobalConfig()

__all__ = ["CONFIG", "GlobalConfig"]
