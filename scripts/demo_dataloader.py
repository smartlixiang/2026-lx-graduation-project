"""Quick functional check for dataset/dataloader abstractions on CIFAR-10."""
from pathlib import Path
import sys
from importlib import import_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BaseDataLoader
import torch

GLOBAL_CFG = import_module("utils.global_config").CONFIG
SEED_UTILS = import_module("utils.seed")


def main() -> None:
    parser = import_module("argparse").ArgumentParser(description="Demo dataset loader")
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in GLOBAL_CFG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    args = parser.parse_args()
    seeds = SEED_UTILS.parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    data_root = GLOBAL_CFG.ensure_data_dir()
    for seed in seeds:
        SEED_UTILS.set_seed(seed)
        loader = BaseDataLoader(
            dataset="cifar10",
            data_path=data_root,
            batch_size=GLOBAL_CFG.default_batch_size,
            val_split=0.2,
            download=False,
            augment=True,
            normalize=True,
            seed=seed,
        )

        train_loader, val_loader, test_loader = loader.load()
        prefix = f"[seed={seed}] " if multi_seed else ""
        print(f"{prefix}已注册数据集: {list(BaseDataLoader.available_datasets())}")
        print(f"{prefix}类别数: {loader.num_classes}")
        print(
            f"{prefix}数据量(train/val/test):",
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
        )

        images, labels = next(iter(train_loader))
        print(f"{prefix}训练批次张量形状:", tuple(images.shape), tuple(labels.shape))
        print(f"{prefix}示例标签:", labels[:5].tolist())
        print(f"{prefix}设备支持CUDA:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
