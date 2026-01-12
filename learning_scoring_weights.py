"""Train a proxy model and reserve hooks for scoring-weight learning."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import gc

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from dataset.dataset_config import CIFAR10
from model.resnet import resnet18
from utils.global_config import CONFIG


class IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset to return the sample index."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label, idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"])
    parser.add_argument("--data_root", type=str, default=str(Path("data")))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else CONFIG.global_device

    data_loader = BaseDataLoader(
        args.dataset,
        data_path=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
    )
    train_loader, _, test_loader = data_loader.load()

    indexed_train_dataset = IndexedDataset(train_loader.dataset)
    train_loader = torch.utils.data.DataLoader(
        indexed_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = resnet18(num_classes=data_loader.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    num_samples = len(indexed_train_dataset)
    num_classes = data_loader.num_classes
    log_dir = Path("weights") / "proxy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_stem = f"{args.dataset}_resnet18_{timestamp}"

    # Use memmap for low-RAM logging during training, then pack everything into a single .npz
    loss_path = log_dir / f"{log_stem}_loss.dat"
    correct_path = log_dir / f"{log_stem}_correct.dat"
    logits_path = log_dir / f"{log_stem}_logits.dat"
    labels_path = log_dir / f"{log_stem}_labels.dat"
    indices_path = log_dir / f"{log_stem}_indices.dat"
    dat_paths = [loss_path, correct_path, logits_path, labels_path, indices_path]

    loss_memmap = np.memmap(
        loss_path,
        mode="w+",
        dtype="float32",
        shape=(args.epochs, num_samples),
    )
    correct_memmap = np.memmap(
        correct_path,
        mode="w+",
        dtype="int8",
        shape=(args.epochs, num_samples),
    )
    logits_memmap = np.memmap(
        logits_path,
        mode="w+",
        dtype="float32",
        shape=(args.epochs, num_samples, num_classes),
    )
    labels_memmap = np.memmap(
        labels_path,
        mode="w+",
        dtype="int64",
        shape=(num_samples,),
    )
    labels_memmap[:] = -1
    indices_memmap = np.memmap(
        indices_path,
        mode="w+",
        dtype="int64",
        shape=(num_samples,),
    )
    indices_memmap[:] = np.arange(num_samples, dtype="int64")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        epoch_idx = epoch - 1

        for images, labels, indices in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            per_sample_loss = criterion(outputs, labels)
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()

            batch_indices = indices.detach().cpu().numpy().astype(np.int64)
            running_loss += per_sample_loss.sum().item()

            loss_memmap[epoch_idx, batch_indices] = per_sample_loss.detach().cpu().numpy()
            logits_memmap[epoch_idx, batch_indices, :] = outputs.detach().cpu().numpy()
            preds = outputs.argmax(dim=1)
            correct_memmap[epoch_idx, batch_indices] = (
                preds.eq(labels).to(dtype=torch.int8).cpu().numpy()
            )
            labels_memmap[batch_indices] = labels.detach().cpu().numpy()

        epoch_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, test_acc={test_acc:.4f}")

        loss_memmap.flush()
        correct_memmap.flush()
        logits_memmap.flush()
        labels_memmap.flush()
        indices_memmap.flush()

    out_path = log_dir / f"{log_stem}.npz"
    np.savez(
        out_path,
        loss=loss_memmap,
        correct=correct_memmap,
        logits=logits_memmap,
        labels=labels_memmap,
        indices=indices_memmap,
    )

    # .npz is the final artifact; remove intermediate .dat files to keep the log directory clean.
    for mm in (loss_memmap, correct_memmap, logits_memmap, labels_memmap, indices_memmap):
        try:
            mm.flush()
        except Exception:
            pass
        # Best-effort close (important on Windows)
        try:
            mm._mmap.close()  # type: ignore[attr-defined]
        except Exception:
            pass
    del loss_memmap, correct_memmap, logits_memmap, labels_memmap, indices_memmap
    gc.collect()

    for p in dat_paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
