"""Train a proxy model and reserve hooks for scoring-weight learning."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from dataset.dataset_config import AVAILABLE_DATASETS
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.path_rules import resolve_proxy_log_dir
from utils.seed import parse_seed_list, set_seed


DATASET_DEFAULT_TRAINING_CONFIGS = {
    "cifar10": {
        "epochs": 100,
        "batch_size": 128,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_milestones": [60, 80],
        "lr_gamma": 0.1,
        "execution": {
            "use_amp": False,
            "reference_effective_batch_size": None,
        },
    },
    "cifar100": {
        "epochs": 100,
        "batch_size": 128,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_milestones": [60, 80],
        "lr_gamma": 0.1,
        "execution": {
            "use_amp": False,
            "reference_effective_batch_size": None,
        },
    },
    "tiny-imagenet": {
        "epochs": 90,
        "batch_size": 256,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_milestones": [30, 60],
        "lr_gamma": 0.1,
        "execution": {
            "default_physical_batch_size": 64,
            "use_amp": True,
            "reference_effective_batch_size": 256,
        },
    },
}


class IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset to return the sample index."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label, idx


def get_default_training_config(dataset_name: str) -> dict[str, int | float | list[int]]:
    try:
        config = DATASET_DEFAULT_TRAINING_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset for training config: {dataset_name}") from exc
    return {
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "init_lr": float(config["init_lr"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "lr_milestones": list(config["lr_milestones"]),
        "lr_gamma": float(config["lr_gamma"]),
    }


def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = get_default_training_config(args.dataset)
    batch_size_unspecified = args.batch_size is None

    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.lr is None:
        args.lr = defaults["init_lr"]
    if args.momentum is None:
        args.momentum = defaults["momentum"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.lr_milestones is None:
        args.lr_milestones = defaults["lr_milestones"]
    if args.lr_gamma is None:
        args.lr_gamma = defaults["lr_gamma"]

    execution_config = DATASET_DEFAULT_TRAINING_CONFIGS[args.dataset]["execution"]
    if args.dataset == "tiny-imagenet":
        if batch_size_unspecified:
            args.batch_size = int(execution_config["default_physical_batch_size"])
        args.use_amp = bool(execution_config["use_amp"])
        args.reference_effective_batch_size = int(execution_config["reference_effective_batch_size"])
        args.physical_batch_size = int(args.batch_size)
        if args.physical_batch_size <= 64:
            args.grad_accum_steps = 4
        elif args.physical_batch_size <= 128:
            args.grad_accum_steps = 2
        else:
            args.grad_accum_steps = 1
        args.effective_batch_size = args.physical_batch_size * args.grad_accum_steps
    else:
        if batch_size_unspecified:
            args.batch_size = defaults["batch_size"]
        args.use_amp = bool(execution_config["use_amp"])
        args.reference_effective_batch_size = args.batch_size
        args.grad_accum_steps = 1
        args.physical_batch_size = args.batch_size
        args.effective_batch_size = args.batch_size
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=AVAILABLE_DATASETS)
    parser.add_argument("--data_root", type=str, default=str(Path("data")))
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument(
        "--lr_milestones",
        type=int,
        nargs="+",
        default=None,
        help="MultiStepLR milestones，例如: --lr_milestones 60 80",
    )
    parser.add_argument("--lr_gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    return parser.parse_args()


def extract_labels(train_dataset: torch.utils.data.Dataset) -> np.ndarray:
    num_samples = len(train_dataset)
    for attr in ("targets", "labels"):
        if hasattr(train_dataset, attr):
            values = getattr(train_dataset, attr)
            if len(values) == num_samples:
                return np.asarray(values, dtype=np.int64)
    labels = np.empty(num_samples, dtype=np.int64)
    for idx in range(num_samples):
        _, label = train_dataset[idx]
        if torch.is_tensor(label):
            label = label.item()
        labels[idx] = int(label)
    return labels


def build_stratified_folds(labels: np.ndarray, k_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    folds: list[list[int]] = [[] for _ in range(k_folds)]
    for cls in np.unique(labels):
        class_indices = np.where(labels == cls)[0]
        rng.shuffle(class_indices)
        for offset, idx in enumerate(class_indices):
            folds[offset % k_folds].append(int(idx))
    return [np.array(fold, dtype=np.int64) for fold in folds]


def build_index_mapping(num_samples: int, indices: list[int]) -> np.ndarray:
    mapping = np.full(num_samples, -1, dtype=np.int64)
    if indices:
        mapping[np.asarray(indices, dtype=np.int64)] = np.arange(len(indices), dtype=np.int64)
    return mapping


def run_for_seed(args: argparse.Namespace, seed: int) -> None:
    set_seed(seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device

    data_loader = BaseDataLoader(
        args.dataset,
        data_path=Path(args.data_root),
        batch_size=args.physical_batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
        seed=seed,
    )
    train_loader, _, _ = data_loader.load()

    train_dataset = train_loader.dataset
    indexed_train_dataset = IndexedDataset(train_dataset)

    num_samples = len(train_dataset)
    num_classes = data_loader.num_classes
    labels = extract_labels(train_dataset)
    folds = build_stratified_folds(labels, args.k_folds, seed)

    proxy_model_name = args.model
    log_dir = resolve_proxy_log_dir(
        args.dataset,
        seed,
        proxy_model=proxy_model_name,
        epochs=args.epochs,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    model_factory = get_model(proxy_model_name)
    runtime_use_amp = bool(args.use_amp and device.type == "cuda")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    val_indices_list = [fold.tolist() for fold in folds]
    meta_payload = {
        "k_folds": args.k_folds,
        "num_samples": num_samples,
        "epochs": args.epochs,
        "num_classes": num_classes,
        "dataset": args.dataset,
        "model": proxy_model_name,
        "seed": seed,
        "timestamp": timestamp,
        "val_indices": val_indices_list,
        "physical_batch_size": args.physical_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.effective_batch_size,
        "lr": args.lr,
        "lr_milestones": args.lr_milestones,
        "lr_gamma": args.lr_gamma,
        "use_amp": runtime_use_amp,
    }
    meta_path = log_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for fold_id, val_indices in enumerate(folds):
        seed_fold = seed * 100 + fold_id
        set_seed(seed_fold)

        val_set = set(val_indices.tolist())
        train_indices = [idx for idx in range(num_samples) if idx not in val_set]
        val_indices_list = val_indices.tolist()

        assert len(set(train_indices).intersection(val_set)) == 0
        assert len(train_indices) + len(val_indices_list) == num_samples

        train_subset = torch.utils.data.Subset(indexed_train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(indexed_train_dataset, val_indices_list)

        generator = torch.Generator().manual_seed(seed_fold)
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=args.physical_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=args.physical_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        model = model_factory(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=args.lr_milestones,
            gamma=args.lr_gamma,
        )
        scaler = GradScaler(enabled=runtime_use_amp)

        train_logits_path = log_dir / f"fold_{fold_id}_train_logits.dat"
        val_logits_path = log_dir / f"fold_{fold_id}_val_logits.dat"
        dat_paths = [train_logits_path, val_logits_path]

        train_logits_memmap = np.memmap(
            train_logits_path,
            mode="w+",
            dtype="float32",
            shape=(args.epochs, len(train_indices), num_classes),
        )
        val_logits_memmap = np.memmap(
            val_logits_path,
            mode="w+",
            dtype="float32",
            shape=(args.epochs, len(val_indices_list), num_classes),
        )

        train_pos_map = build_index_mapping(num_samples, train_indices)
        val_pos_map = build_index_mapping(num_samples, val_indices_list)

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            train_progress = tqdm(
                train_loader,
                desc=f"Fold {fold_id} Epoch {epoch}/{args.epochs}",
                unit="batch",
            )
            epoch_idx = epoch - 1
            total_batches = len(train_loader)
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (images, labels, indices) in enumerate(train_progress, start=1):
                images = images.to(device)
                labels = labels.to(device)

                with autocast(enabled=runtime_use_amp):
                    outputs = model(images)
                    per_sample_loss = criterion(outputs, labels)
                    loss = per_sample_loss.mean()

                loss_for_backward = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == total_batches)
                if should_step:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                batch_indices = indices.detach().cpu().numpy().astype(np.int64)
                positions = train_pos_map[batch_indices]
                if np.any(positions < 0):
                    raise ValueError("Train batch contains indices outside current fold.")
                running_loss += per_sample_loss.sum().item()

                train_logits_memmap[epoch_idx, positions, :] = outputs.detach().float().cpu().numpy()
                train_progress.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) else 0.0
            print(f"Fold {fold_id} Epoch {epoch}: train_loss={epoch_loss:.4f}")

            model.eval()
            with torch.no_grad():
                val_progress = tqdm(
                    val_loader,
                    desc=f"Fold {fold_id} Epoch {epoch}/{args.epochs} (val)",
                    unit="batch",
                )
                for images, labels, indices in val_progress:
                    images = images.to(device)
                    labels = labels.to(device)
                    with autocast(enabled=runtime_use_amp):
                        outputs = model(images)

                    batch_indices = indices.detach().cpu().numpy().astype(np.int64)
                    positions = val_pos_map[batch_indices]
                    if np.any(positions < 0):
                        raise ValueError("Val batch contains indices outside current fold.")

                    val_logits_memmap[epoch_idx, positions, :] = outputs.detach().float().cpu().numpy()

            train_logits_memmap.flush()
            val_logits_memmap.flush()

        rng = np.random.RandomState(seed_fold)
        if train_indices:
            train_check = rng.choice(train_indices, size=min(10, len(train_indices)), replace=False)
            for idx in train_check:
                pos = train_pos_map[idx]
                assert pos >= 0
                assert train_indices[pos] == idx
        if val_indices_list:
            val_check = rng.choice(val_indices_list, size=min(10, len(val_indices_list)), replace=False)
            for idx in val_check:
                pos = val_pos_map[idx]
                assert pos >= 0
                assert val_indices_list[pos] == idx

        fold_meta = {
            "epochs": args.epochs,
            "num_classes": num_classes,
            "fold_id": fold_id,
            "k_folds": args.k_folds,
            "dataset_name": args.dataset,
            "model": proxy_model_name,
            "seed": seed,
            "physical_batch_size": args.physical_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.effective_batch_size,
            "lr": args.lr,
            "lr_milestones": args.lr_milestones,
            "lr_gamma": args.lr_gamma,
            "use_amp": runtime_use_amp,
        }
        out_path = log_dir / f"fold_{fold_id}.npz"
        np.savez(
            out_path,
            train_indices=np.asarray(train_indices, dtype=np.int64),
            val_indices=np.asarray(val_indices_list, dtype=np.int64),
            train_logits=train_logits_memmap,
            val_logits=val_logits_memmap,
            meta=np.array(json.dumps(fold_meta, ensure_ascii=False), dtype=object),
        )

        for mm in (train_logits_memmap, val_logits_memmap):
            try:
                mm.flush()
            except Exception:
                pass
            try:
                mm._mmap.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        del train_logits_memmap, val_logits_memmap
        gc.collect()

        for path in dat_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    args = parse_args()
    args = apply_dataset_defaults(args)
    seeds = parse_seed_list(args.seed)
    for seed in seeds:
        run_for_seed(args, seed)


if __name__ == "__main__":
    main()
