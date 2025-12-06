"""Quick functional check for dataset/dataloader abstractions.

This script avoids network downloads by relying on the synthetic
`FakeImageDataset` registered in ``dataset.dataset``. It prints dataset
sizes and the shape of one training batch to demonstrate that the
registry, transforms, and split logic work end-to-end.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BaseDataLoader
import torch


def main() -> None:
    data_root = Path("./data")
    loader = BaseDataLoader(
        dataset="cifar10",
        data_path=data_root,
        batch_size=8,
        val_split=0.2,
        download=False,
        augment=True,
        normalize=True,
        seed=123,
    )

    train_loader, val_loader, test_loader = loader.load()
    print(f"已注册数据集: {list(BaseDataLoader.available_datasets())}")
    print(f"类别数: {loader.num_classes}")
    print(
        "数据量(train/val/test):",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    images, labels = next(iter(train_loader))
    print("训练批次张量形状:", tuple(images.shape), tuple(labels.shape))
    print("示例标签:", labels[:5].tolist())
    print("设备支持CUDA:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
