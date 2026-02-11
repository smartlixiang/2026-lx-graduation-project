"""Train after data selection and save evaluation results."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from dataset.dataset_config import AVAILABLE_DATASETS
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.path_rules import resolve_checkpoint_path, resolve_mask_path, resolve_result_path
from utils.seed import parse_seed_list, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=AVAILABLE_DATASETS,
    )
    parser.add_argument("--data_root", type=str, default=str(Path("data")))
    parser.add_argument(
        "--cr",  # cut ratios
        type=str,
        default="20,30,40,60,70,80,90",
        help="裁剪比例列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        help="数据选择方法名称（random 为随机采样）",
    )
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    parser.add_argument("--result_root", type=str, default="result")
    parser.add_argument(
        "--skip_saved",
        action="store_true",
        help="跳过已经保存的结果文件",
    )
    parser.add_argument(
        "--load_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练开始前是否加载已有 checkpoint",
    )
    return parser.parse_args()


def parse_ratio_list(ratio_text: str) -> list[int]:
    cleaned = ratio_text.strip()
    if not cleaned:
        return []
    if "," in cleaned:
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [cleaned]
    return [int(item) for item in items]


def _extract_labels(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.asarray([dataset[idx][1] for idx in range(len(dataset))])


def select_random_indices_by_class(
    labels: np.ndarray,
    num_classes: int,
    cut_ratio: int,
    seed: int,
) -> np.ndarray:
    if cut_ratio <= 0:
        raise ValueError("cut_ratio must be positive")
    if cut_ratio > 100:
        raise ValueError("cut_ratio must be <= 100")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    ratio = cut_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            continue
        if cut_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        chosen = rng.choice(class_indices, size=num_select, replace=False)
        selected.extend(chosen.tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_selection_mask(
    dataset_name: str,
    mode: str,
    cut_ratio: int,
    seed: int,
    model_name: str,
) -> np.ndarray:
    """Load a 0/1 mask for a selection method.

    The mask should have shape (N,) and values in {0, 1}, where 1 indicates
    the sample is selected.
    """
    mask_seed = CONFIG.global_seed if mode == "my_naive" else seed
    mask_path = resolve_mask_path(
        mode=mode,
        dataset=dataset_name,
        model=model_name,
        seed=mask_seed,
        cut_ratio=cut_ratio,
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"未找到 mask 文件: {mask_path}")
    with np.load(mask_path) as data:
        if "mask" in data:
            mask = data["mask"]
        elif len(data.files) == 1:
            mask = data[data.files[0]]
        else:
            raise ValueError(f"mask 文件格式不正确: {mask_path}")
    return np.asarray(mask)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


def prepare_selection_indices(
    dataset_name: str,
    mode: str,
    cut_ratio: int,
    seed: int,
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    model_name: str,
) -> np.ndarray:
    if mode == "random":
        labels = _extract_labels(dataset)
        return select_random_indices_by_class(labels, num_classes, cut_ratio, seed)

    mask = load_selection_mask(dataset_name, mode, cut_ratio, seed, model_name)
    mask = np.asarray(mask).astype(bool)
    return np.flatnonzero(mask)


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device

    data_loader = BaseDataLoader(
        args.dataset,
        data_path=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
        seed=seed,
    )
    train_loader, _, test_loader = data_loader.load()
    train_dataset = train_loader.dataset

    model_name = args.model
    cut_ratios = parse_ratio_list(args.cr)
    model_factory = get_model(model_name)

    for cut_ratio in cut_ratios:
        result_path = resolve_result_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            cut_ratio=cut_ratio,
            root=Path(args.result_root),
        )
        result_dir = result_path.parent
        checkpoint_path = resolve_checkpoint_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            cut_ratio=cut_ratio,
        )
        checkpoint_dir = checkpoint_path.parent
        if args.skip_saved and result_path.exists():
            continue

        start_time = time.time()
        selected_indices = prepare_selection_indices(
            args.dataset,
            args.mode,
            cut_ratio,
            seed,
            train_dataset,
            data_loader.num_classes,
            model_name,
        )
        subset = Subset(train_dataset, selected_indices.tolist())

        generator = torch.Generator().manual_seed(seed)
        subset_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            drop_last=False,
        )

        model = model_factory(num_classes=data_loader.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(
            model.parameters(),
            lr=args.init_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

        accuracy_samples: list[float] = []
        start_eval_epoch = max(1, args.epochs - 9)
        start_epoch = 1
        if args.load_checkpoint and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            accuracy_samples = list(checkpoint.get("accuracy_samples", []))
            start_epoch = int(checkpoint["epoch"]) + 1
            elapsed_time = float(checkpoint.get("elapsed_time", 0.0))
            start_time = time.time() - elapsed_time

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                subset_loader,
                optimizer,
                criterion,
                device,
                epoch,
                args.epochs,
            )
            scheduler.step()
            if epoch >= start_eval_epoch:
                eval_accuracy = round(
                    float(evaluate(model, test_loader, device)),
                    4,
                )
                accuracy_samples.append(eval_accuracy)
                print(f"Test accuracy at epoch {epoch}: {eval_accuracy:.4f}")
            if epoch % 20 == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "accuracy_samples": accuracy_samples,
                        "elapsed_time": time.time() - start_time,
                    },
                    checkpoint_path,
                )

        total_time = time.time() - start_time
        accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
        accuracy = round(accuracy, 4)

        result_dir.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "metadata": {
                "dataset": args.dataset,
                "data_root": str(Path(args.data_root)),
                "model": model_name,
                "cut_ratio": cut_ratio,
                "selection_method": args.mode,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "SGD",
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "init_lr": args.init_lr,
                "lr_schedule": {
                    "type": "MultiStepLR",
                    "milestones": [60, 120, 160],
                    "gamma": 0.2,
                },
                "num_selected": len(selected_indices),
                "num_total": len(train_dataset),
            },
            "accuracy": accuracy,
            "accuracy_samples": accuracy_samples,
            "time_seconds": total_time,
        }

        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if multi_seed:
            print(f"Saved result to {result_path}")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)


if __name__ == "__main__":
    main()
