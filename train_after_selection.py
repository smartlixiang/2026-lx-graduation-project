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
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.seed import parse_seed_list, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--data_root", type=str, default=str(Path("data")))
    parser.add_argument(
        "--cut_ratios",
        type=str,
        default="20,30,40,60,70,80,90,100",
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
    data_root: Path,
) -> np.ndarray:
    """Load a 0/1 mask for a selection method.

    TODO: Replace this stub with real loading logic once the selection artifacts
    (e.g., .npz masks) are finalized. The mask should have shape (N,) and values
    in {0, 1}, where 1 indicates the sample is selected.
    """
    raise NotImplementedError(
        f"Selection loading for mode='{mode}' is not implemented yet."
    )


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
    data_root: Path,
) -> np.ndarray:
    if mode == "random":
        labels = _extract_labels(dataset)
        return select_random_indices_by_class(labels, num_classes, cut_ratio, seed)

    mask = load_selection_mask(dataset_name, mode, cut_ratio, seed, data_root)
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
    cut_ratios = parse_ratio_list(args.cut_ratios)
    model_factory = get_model(model_name)

    for cut_ratio in cut_ratios:
        start_time = time.time()
        selected_indices = prepare_selection_indices(
            args.dataset,
            args.mode,
            cut_ratio,
            seed,
            train_dataset,
            data_loader.num_classes,
            Path(args.data_root),
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
        for epoch in range(1, args.epochs + 1):
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
                accuracy_samples.append(evaluate(model, test_loader, device))

        total_time = time.time() - start_time
        accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0

        result_dir = Path(args.result_root) / args.dataset / model_name / str(seed)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"result_{cut_ratio}_{args.mode}.json"

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
            "time_seconds": total_time,
        }

        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

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
