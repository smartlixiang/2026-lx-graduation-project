"""Train after data selection and save evaluation results."""
from __future__ import annotations

import argparse
import json
import time
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
from dataset.dataset_config import AVAILABLE_DATASETS
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.path_rules import resolve_checkpoint_path, resolve_mask_path, resolve_result_path
from utils.progress import PersistentStatusLine, create_persistent_bar, create_transient_batch_bar
from utils.seed import parse_seed_list, set_seed
from utils.training_defaults import apply_dataset_training_defaults




def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    return apply_dataset_training_defaults(args, lr_attr="init_lr")


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
        "--kr",  # keep ratios
        type=str,
        default="20,30,40,50,60,70,80,90",
        help="裁剪比例列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        help="数据选择方法名称（random 为随机采样）",
    )
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument(
        "--lr_milestones",
        type=int,
        nargs="+",
        default=None,
        help="MultiStepLR milestones，例如: --lr_milestones 60 120 160",
    )
    parser.add_argument("--lr_gamma", type=float, default=None)
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
    keep_ratio: int,
    seed: int,
) -> np.ndarray:
    if keep_ratio <= 0:
        raise ValueError("keep_ratio must be positive")
    if keep_ratio > 100:
        raise ValueError("keep_ratio must be <= 100")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    ratio = keep_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            continue
        if keep_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        chosen = rng.choice(class_indices, size=num_select, replace=False)
        selected.extend(chosen.tolist())
    return np.sort(np.asarray(selected, dtype=np.int64))


def load_selection_mask(
    dataset_name: str,
    mode: str,
    keep_ratio: int,
    seed: int,
    model_name: str,
) -> np.ndarray:
    """Load a 0/1 mask for a selection method.

    Note that the stored mask path intentionally does not depend on
    ``model_name``: selection is model-agnostic for this stage, while the
    downstream train/test results are saved under a model-aware path.

    The mask should have shape (N,) and values in {0, 1}, where 1 indicates
    the sample is selected.
    """
    mask_seed = CONFIG.global_seed if mode == "my_naive" else seed
    mask_path = resolve_mask_path(
        mode=mode,
        dataset=dataset_name,
        model=model_name,
        seed=mask_seed,
        keep_ratio=keep_ratio,
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
    use_amp: bool,
    grad_accum_steps: int,
    scaler: GradScaler,
    position: int = 1,
) -> float:
    model.train()
    running_loss = 0.0
    progress = create_transient_batch_bar(
        loader,
        desc=f"Train batch {epoch}/{total_epochs}",
        position=position,
    )
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss_for_backward = loss / grad_accum_steps
        if scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (batch_idx % grad_accum_steps == 0) or (batch_idx == total_batches)
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def prepare_selection_indices(
    dataset_name: str,
    mode: str,
    keep_ratio: int,
    seed: int,
    dataset: torch.utils.data.Dataset,
    num_classes: int,
    model_name: str,
) -> np.ndarray:
    if mode == "random":
        labels = _extract_labels(dataset)
        return select_random_indices_by_class(labels, num_classes, keep_ratio, seed)

    mask = load_selection_mask(dataset_name, mode, keep_ratio, seed, model_name)
    mask = np.asarray(mask).astype(bool)
    return np.flatnonzero(mask)


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
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
    train_loader, _, test_loader = data_loader.load()
    train_dataset = train_loader.dataset

    model_name = args.model
    keep_ratios = parse_ratio_list(args.kr)
    model_factory = get_model(model_name)

    for keep_ratio in keep_ratios:
        # Result files are model-sensitive and therefore include the model name.
        result_path = resolve_result_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            keep_ratio=keep_ratio,
            root=Path(args.result_root),
        )
        result_dir = result_path.parent
        checkpoint_path = resolve_checkpoint_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            keep_ratio=keep_ratio,
        )
        checkpoint_dir = checkpoint_path.parent
        if args.skip_saved and result_path.exists():
            continue

        start_time = time.time()
        selected_indices = prepare_selection_indices(
            args.dataset,
            args.mode,
            keep_ratio,
            seed,
            train_dataset,
            data_loader.num_classes,
            model_name,
        )
        subset = Subset(train_dataset, selected_indices.tolist())

        generator = torch.Generator().manual_seed(seed)
        subset_loader = DataLoader(
            subset,
            batch_size=args.physical_batch_size,
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
        scheduler = MultiStepLR(
            optimizer,
            milestones=args.lr_milestones,
            gamma=args.lr_gamma,
        )
        runtime_use_amp = bool(args.use_amp and device.type == "cuda")
        scaler = GradScaler(enabled=runtime_use_amp)

        accuracy_samples: list[float] = []
        start_eval_epoch = max(1, args.epochs - 9)
        start_epoch = 1
        if args.load_checkpoint and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            accuracy_samples = list(checkpoint.get("accuracy_samples", []))
            if checkpoint.get("scaler_state") is not None and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            elapsed_time = float(checkpoint.get("elapsed_time", 0.0))
            start_time = time.time() - elapsed_time

        epoch_bar = create_persistent_bar(
            total=args.epochs,
            desc=f"[seed={seed}] mode={args.mode} kr={keep_ratio}",
            position=0,
            initial=max(0, start_epoch - 1),
        )
        test_status = PersistentStatusLine(
            "Last-10 test accs (0/10): []",
            position=2,
        )
        if accuracy_samples:
            formatted = ", ".join(f"{acc:.4f}" for acc in accuracy_samples)
            test_status.update(
                f"Last-10 test accs ({len(accuracy_samples)}/10): [{formatted}]"
            )

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_bar.set_postfix_str(f"epoch={epoch}/{args.epochs}")
            train_one_epoch(
                model,
                subset_loader,
                optimizer,
                criterion,
                device,
                epoch,
                args.epochs,
                runtime_use_amp,
                args.grad_accum_steps,
                scaler,
                position=1,
            )
            scheduler.step()
            if epoch >= start_eval_epoch:
                eval_accuracy = round(
                    float(evaluate(model, test_loader, device)),
                    4,
                )
                accuracy_samples.append(eval_accuracy)
                formatted = ", ".join(f"{acc:.4f}" for acc in accuracy_samples)
                test_status.update(
                    f"Last-10 test accs ({len(accuracy_samples)}/10): [{formatted}]"
                )
            if epoch % 20 == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                        "accuracy_samples": accuracy_samples,
                        "elapsed_time": time.time() - start_time,
                        "use_amp": runtime_use_amp,
                        "grad_accum_steps": args.grad_accum_steps,
                        "physical_batch_size": args.physical_batch_size,
                        "effective_batch_size": args.effective_batch_size,
                    },
                    checkpoint_path,
                )
            epoch_bar.update(1)

        total_time = time.time() - start_time
        accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
        accuracy_std = float(np.std(accuracy_samples, ddof=0)) if accuracy_samples else 0.0
        accuracy = round(accuracy, 4)
        accuracy_std = round(accuracy_std, 4)

        result_dir.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "metadata": {
                "dataset": args.dataset,
                "data_root": str(Path(args.data_root)),
                "model": model_name,
                "keep_ratio": keep_ratio,
                "selection_method": args.mode,
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.physical_batch_size,
                "physical_batch_size": args.physical_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.effective_batch_size,
                "reference_effective_batch_size": args.reference_effective_batch_size,
                "optimizer": "SGD",
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "init_lr": args.init_lr,
                "use_amp": runtime_use_amp,
                "lr_schedule": {
                    "type": "MultiStepLR",
                    "milestones": args.lr_milestones,
                    "gamma": args.lr_gamma,
                },
                "num_selected": len(selected_indices),
                "num_total": len(train_dataset),
            },
            "accuracy": accuracy,
            "accuracy_mean": accuracy,
            "accuracy_std": accuracy_std,
            "accuracy_samples": accuracy_samples,
            "time_seconds": total_time,
        }

        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
        epoch_bar.close()
        test_status.close()
        tqdm.write(f"Saved result to {result_path}")
        tqdm.write(
            f"Last-10 test accs: [{', '.join(f'{acc:.4f}' for acc in accuracy_samples)}]"
        )
        tqdm.write(f"Last-10 mean/std: mean={accuracy:.4f}, std={accuracy_std:.4f}")


def main() -> None:
    args = parse_args()
    args = apply_dataset_defaults(args)
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)


if __name__ == "__main__":
    main()
