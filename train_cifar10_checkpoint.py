from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from model.model_config import get_model
from train_after_selection import (
    load_selection_mask,
    select_random_indices_by_class,
)
from utils.global_config import CONFIG
from utils.path_rules import resolve_checkpoint_path
from utils.seed import set_seed
from utils.training_defaults import apply_dataset_training_defaults


DATASET_NAME = "cifar10"
MODEL_NAME = "resnet50"
FIXED_SEED = 22
DATA_ROOT = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on CIFAR-10 selected subsets and save final checkpoints."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Comma-separated selection methods, e.g. random,MoSo,YangCLIP,learned_group",
    )
    parser.add_argument(
        "--kr",
        type=str,
        required=True,
        help="Keep ratio(s), e.g. 50 or 20,50",
    )
    return parser.parse_args()


def build_train_args() -> argparse.Namespace:
    args = argparse.Namespace()
    args.dataset = DATASET_NAME
    args.epochs = None
    args.batch_size = None
    args.init_lr = None
    args.momentum = None
    args.weight_decay = None
    args.lr_milestones = None
    args.lr_gamma = None
    args = apply_dataset_training_defaults(args, lr_attr="init_lr")
    return args


def parse_csv_items(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def extract_labels(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.asarray([dataset[idx][1] for idx in range(len(dataset))])


def prepare_selection_indices(
    mode: str,
    keep_ratio: int,
    train_dataset: torch.utils.data.Dataset,
    num_classes: int,
) -> np.ndarray:
    if mode == "random":
        labels = extract_labels(train_dataset)
        return select_random_indices_by_class(
            labels=labels,
            num_classes=num_classes,
            keep_ratio=keep_ratio,
            seed=FIXED_SEED,
        )

    mask = load_selection_mask(
        dataset_name=DATASET_NAME,
        mode=mode,
        keep_ratio=keep_ratio,
        seed=FIXED_SEED,
        model_name=MODEL_NAME,
    )
    mask = np.asarray(mask).astype(bool)
    return np.flatnonzero(mask)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_amp: bool,
    scaler: GradScaler,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


def save_final_checkpoint(
    mode: str,
    keep_ratio: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: MultiStepLR,
    scaler: GradScaler,
    final_accuracy: float,
    train_args: argparse.Namespace,
) -> Path:
    checkpoint_path = resolve_checkpoint_path(
        mode=mode,
        dataset=DATASET_NAME,
        model=MODEL_NAME,
        seed=FIXED_SEED,
        keep_ratio=keep_ratio,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "mode": mode,
            "keep_ratio": keep_ratio,
            "seed": FIXED_SEED,
            "epoch": train_args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
            "final_test_accuracy": float(final_accuracy),
            "batch_size": train_args.physical_batch_size,
            "effective_batch_size": train_args.effective_batch_size,
            "init_lr": train_args.init_lr,
            "momentum": train_args.momentum,
            "weight_decay": train_args.weight_decay,
            "lr_milestones": list(train_args.lr_milestones),
            "lr_gamma": train_args.lr_gamma,
            "use_amp": bool(train_args.use_amp),
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_single(mode: str, keep_ratio: int, train_args: argparse.Namespace) -> None:
    set_seed(FIXED_SEED)
    device = CONFIG.global_device

    data_loader = BaseDataLoader(
        DATASET_NAME,
        data_path=DATA_ROOT,
        batch_size=train_args.physical_batch_size,
        num_workers=CONFIG.num_workers,
        val_split=0.0,
        seed=FIXED_SEED,
    )
    train_loader_full, _, test_loader = data_loader.load()
    train_dataset = train_loader_full.dataset

    selected_indices = prepare_selection_indices(
        mode=mode,
        keep_ratio=keep_ratio,
        train_dataset=train_dataset,
        num_classes=data_loader.num_classes,
    )
    subset = Subset(train_dataset, selected_indices.tolist())

    generator = torch.Generator().manual_seed(FIXED_SEED)
    subset_loader = DataLoader(
        subset,
        batch_size=train_args.physical_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=CONFIG.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = get_model(MODEL_NAME)(num_classes=data_loader.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=train_args.init_lr,
        momentum=train_args.momentum,
        weight_decay=train_args.weight_decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=train_args.lr_milestones,
        gamma=train_args.lr_gamma,
    )
    scaler = GradScaler(enabled=bool(train_args.use_amp and device.type == "cuda"))

    print(
        f"\n[Start] mode={mode}, kr={keep_ratio}, "
        f"num_selected={len(selected_indices)}/{len(train_dataset)}"
    )

    for epoch in range(1, train_args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=subset_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=train_args.epochs,
            use_amp=bool(train_args.use_amp and device.type == "cuda"),
            scaler=scaler,
        )
        scheduler.step()

        if epoch in {1, 50, 100, 150, train_args.epochs}:
            test_acc = evaluate(model, test_loader, device)
            print(
                f"[Eval] mode={mode}, kr={keep_ratio}, epoch={epoch}, "
                f"loss={train_loss:.4f}, test_acc={test_acc:.4f}"
            )

    final_test_accuracy = evaluate(model, test_loader, device)
    checkpoint_path = save_final_checkpoint(
        mode=mode,
        keep_ratio=keep_ratio,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        final_accuracy=final_test_accuracy,
        train_args=train_args,
    )
    print(
        f"[Done] mode={mode}, kr={keep_ratio}, final_test_accuracy={final_test_accuracy:.4f}\n"
        f"Saved checkpoint to: {checkpoint_path}"
    )


def main() -> None:
    cli_args = parse_args()
    modes = parse_csv_items(cli_args.mode)
    keep_ratios = parse_csv_ints(cli_args.kr)

    if not modes:
        raise ValueError("No valid mode is provided.")
    if not keep_ratios:
        raise ValueError("No valid keep ratio is provided.")
    for kr in keep_ratios:
        if kr <= 0 or kr > 100:
            raise ValueError(f"Invalid keep ratio: {kr}")

    train_args = build_train_args()

    for mode in modes:
        for keep_ratio in keep_ratios:
            run_single(mode, keep_ratio, train_args)


if __name__ == "__main__":
    main()
