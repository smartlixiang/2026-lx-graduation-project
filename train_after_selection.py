"""Train after data selection and save evaluation results.

Supported modes:
  1. Default clean-data experiment: train on the selected clean training subset
     and evaluate on the clean test split.
  2. ``--exp label_noise``: read ``noise/<dataset>/noise_list_<seed>.txt``,
     modify training labels only, keep the test split clean, and write outputs
     under ``result_label_noise/`` and ``checkpoint_label_noise/``.
  3. ``--exp corruption``: read
     ``corruption_data/<dataset>/corruption_list_<seed>.txt``, apply fixed image
     corruptions to listed training samples before the training transform, keep
     labels correct and the test split clean, and write outputs under
     ``result_corruption/`` and ``checkpoint_corruption/``.

All non-random experiments read selection masks from the project-level
``mask/`` directory via ``utils.path_rules.resolve_mask_path``. The common
layout is ``mask/<mode>/<dataset>/<seed>/mask_<keep_ratio>.npz``. ``random``
mode samples by class and does not read a mask file.

Examples:
  CUDA_VISIBLE_DEVICES=0 python train_after_selection.py \
      --dataset tiny-imagenet \
      --exp corruption \
      --mode corruption_yangclip \
      --seed 22 \
      --kr 30

  The command above reads:
      mask/corruption_yangclip/tiny-imagenet/22/mask_30.npz

  python train_after_selection.py \
      --exp corruption \
      --dataset cifar100 \
      --mode corruption_learned_group \
      --kr 30,50,70 \
      --seed 22,42,96

  python train_after_selection.py \
      --exp corruption \
      --dataset tiny-imagenet \
      --mode corruption_naive_group \
      --kr 30,50,70 \
      --seed 22 \
      --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from corruption_exp import corruption_opt
from dataset.dataset import BaseDataLoader
from dataset.dataset_config import AVAILABLE_DATASETS
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.path_rules import resolve_checkpoint_path, resolve_mask_path, resolve_result_path
from utils.progress import PersistentStatusLine, create_persistent_bar, create_transient_batch_bar
from utils.seed import parse_seed_list, set_seed
from utils.training_defaults import apply_dataset_training_defaults


SUPPORTED_EXPERIMENTS = {"label_noise", "corruption"}
CORRUPTION_SUPPORTED_DATASETS = {"cifar100", "tiny-imagenet"}
EXPECTED_TRAIN_SIZES = {"cifar100": 50000, "tiny-imagenet": 100000}
DATASET_NUMERIC_ID = {"cifar100": 100, "tiny-imagenet": 200}


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
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_EXPERIMENTS),
        help="实验设置。None 为正常数据实验；label_noise 为标签噪声实验；corruption 为图像破坏实验。",
    )
    parser.add_argument(
        "--noise_root",
        type=str,
        default=str(Path("noise")),
        help="标签注噪列表根目录，仅在 --exp label_noise 时使用。",
    )
    parser.add_argument(
        "--strict_noise_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="标签噪声实验中是否严格检查注噪表合法性。",
    )
    parser.add_argument(
        "--corruption_root",
        type=str,
        default=str(Path("corruption_data")),
        help="图像破坏清单根目录，仅在 --exp corruption 时使用。",
    )
    parser.add_argument(
        "--strict_corruption_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="图像破坏实验中是否严格检查 corruption list 的数量和类型分布。",
    )
    parser.add_argument("--result_root", type=str, default="result")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoint",
        help="checkpoint 根目录；指定 --exp 时会自动追加实验名。",
    )
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


def _root_for_exp(base_root: str | Path, exp: str | None) -> Path:
    base = Path(base_root)
    if exp is None:
        return base

    # Default result/checkpoint roots become result_label_noise/checkpoint_label_noise.
    # Custom roots are handled consistently by appending the experiment name.
    return base.with_name(f"{base.name}_{exp}")


def _extract_labels(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "samples"):
        return np.asarray([label for _, label in dataset.samples])
    return np.asarray([dataset[idx][1] for idx in range(len(dataset))])


def _set_dataset_labels(dataset: torch.utils.data.Dataset, labels: np.ndarray) -> None:
    """Mutate a torchvision-style dataset so subsequent __getitem__ returns labels."""
    labels_list = [int(x) for x in labels.tolist()]
    updated = False

    if hasattr(dataset, "targets"):
        dataset.targets = labels_list
        updated = True
    if hasattr(dataset, "labels"):
        dataset.labels = labels_list
        updated = True

    # ImageFolder/DatasetFolder also stores labels inside samples/imgs.
    if hasattr(dataset, "samples"):
        dataset.samples = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.samples)]
        updated = True
    if hasattr(dataset, "imgs"):
        dataset.imgs = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.imgs)]
        updated = True

    if not updated:
        raise TypeError(
            "当前训练集对象不支持原地修改标签。请确认 BaseDataLoader 返回的是 "
            "torchvision CIFAR/ImageFolder 风格数据集。"
        )


def read_noise_list(noise_path: Path) -> np.ndarray:
    if not noise_path.exists():
        raise FileNotFoundError(f"未找到标签注噪文件: {noise_path}")
    mapping = np.loadtxt(noise_path, dtype=np.int64)
    if mapping.ndim == 1:
        mapping = mapping.reshape(1, 2)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError(f"标签注噪文件必须是两列 txt: {noise_path}, 当前 shape={mapping.shape}")
    return mapping


def apply_label_noise_to_dataset(
    dataset: torch.utils.data.Dataset,
    dataset_name: str,
    seed: int,
    noise_root: Path,
    num_classes: int,
    strict: bool = True,
) -> dict[str, object]:
    """Apply fixed label noise to the training dataset in-place.

    The noise list uses the original training-set order. Since this script loads
    train split with val_split=0.0, the dataset order should match the original
    torchvision training order.
    """
    clean_labels = _extract_labels(dataset).astype(np.int64)
    num_total = int(clean_labels.shape[0])

    noise_path = noise_root / dataset_name / f"noise_list_{seed}.txt"
    mapping = read_noise_list(noise_path)
    noisy_indices = mapping[:, 0].astype(np.int64)
    noisy_new_labels = mapping[:, 1].astype(np.int64)

    if strict:
        if len(np.unique(noisy_indices)) != len(noisy_indices):
            raise ValueError(f"标签注噪文件存在重复 sample_id: {noise_path}")
        if np.any(noisy_indices < 0) or np.any(noisy_indices >= num_total):
            raise ValueError(f"标签注噪文件存在越界 sample_id: {noise_path}")
        if np.any(noisy_new_labels < 0) or np.any(noisy_new_labels >= num_classes):
            raise ValueError(f"标签注噪文件存在越界 noisy_label: {noise_path}")
        same = noisy_new_labels == clean_labels[noisy_indices]
        if np.any(same):
            raise ValueError(
                f"标签注噪文件中有 {int(np.sum(same))} 个 noisy_label 与原始标签相同: {noise_path}"
            )

    noisy_labels = clean_labels.copy()
    noisy_labels[noisy_indices] = noisy_new_labels
    _set_dataset_labels(dataset, noisy_labels)

    is_noisy = np.zeros(num_total, dtype=bool)
    is_noisy[noisy_indices] = True

    return {
        "noise_path": str(noise_path),
        "num_noisy": int(noisy_indices.shape[0]),
        "noise_ratio": float(noisy_indices.shape[0] / num_total) if num_total else 0.0,
        "num_total": num_total,
        "noisy_indices": noisy_indices,
        "is_noisy": is_noisy,
    }



def read_corruption_list(corruption_path: Path) -> np.ndarray:
    if not corruption_path.exists():
        raise FileNotFoundError(f"未找到图像破坏清单文件: {corruption_path}")
    mapping = np.loadtxt(corruption_path, dtype=np.int64)
    if mapping.ndim == 1:
        mapping = mapping.reshape(1, 2)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError(
            f"图像破坏清单文件必须是两列 txt: {corruption_path}, 当前 shape={mapping.shape}"
        )
    return mapping


def _corruption_type_counts(type_ids: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for type_id, name in corruption_opt.CORRUPTION_ID_TO_NAME.items():
        counts[name] = int(np.sum(type_ids == int(type_id)))
    return counts


def load_corruption_info(
    dataset_name: str,
    seed: int,
    corruption_root: Path,
    num_samples: int,
    strict: bool = True,
) -> dict[str, object]:
    if dataset_name not in CORRUPTION_SUPPORTED_DATASETS:
        supported = ", ".join(sorted(CORRUPTION_SUPPORTED_DATASETS))
        raise ValueError(f"--exp corruption 仅支持数据集: {supported}; 当前 dataset={dataset_name}")

    corruption_path = corruption_root / dataset_name / f"corruption_list_{seed}.txt"
    mapping = read_corruption_list(corruption_path)
    sample_ids = mapping[:, 0].astype(np.int64)
    type_ids = mapping[:, 1].astype(np.int64)

    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError(
            f"图像破坏清单存在越界 sample_id: {corruption_path}, train_size={num_samples}"
        )
    if np.any(type_ids < 0) or np.any(type_ids >= corruption_opt.NUM_CORRUPTION_TYPES):
        raise ValueError(
            f"图像破坏清单存在非法 corruption_type: {corruption_path}, "
            f"valid=[0, {corruption_opt.NUM_CORRUPTION_TYPES})"
        )
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"图像破坏清单存在重复 sample_id: {corruption_path}")

    if strict:
        expected_train_size = EXPECTED_TRAIN_SIZES[dataset_name]
        if num_samples != expected_train_size:
            raise ValueError(
                f"严格检查要求 {dataset_name} 训练集大小为 {expected_train_size}, "
                f"当前为 {num_samples}"
            )
        expected_corrupted = expected_train_size // 5
        if sample_ids.shape[0] != expected_corrupted:
            raise ValueError(
                f"严格检查要求图像破坏样本数为训练集 20% ({expected_corrupted}), "
                f"当前为 {sample_ids.shape[0]}: {corruption_path}"
            )
        expected_per_type = expected_corrupted // corruption_opt.NUM_CORRUPTION_TYPES
        counts = _corruption_type_counts(type_ids)
        bad_counts = {name: count for name, count in counts.items() if count != expected_per_type}
        if bad_counts:
            raise ValueError(
                f"严格检查要求五种图像破坏类型各 {expected_per_type} 个, "
                f"当前 type_counts={counts}: {corruption_path}"
            )

    corruption_types = np.full(num_samples, -1, dtype=np.int16)
    corruption_types[sample_ids] = type_ids.astype(np.int16)
    is_corrupted = corruption_types >= 0

    return {
        "corruption_path": str(corruption_path),
        "num_total": int(num_samples),
        "num_corrupted": int(sample_ids.shape[0]),
        "corruption_ratio": float(sample_ids.shape[0] / num_samples) if num_samples else 0.0,
        "corrupted_indices": sample_ids,
        "corruption_types": corruption_types,
        "is_corrupted": is_corrupted,
        "type_counts": _corruption_type_counts(type_ids),
    }


class FixedCorruptionDataset(Dataset):
    """Apply deterministic image corruption before the original train transform."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        corruption_types: np.ndarray,
        dataset_name: str,
        seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption_types = np.asarray(corruption_types, dtype=np.int16)
        self.dataset_name = dataset_name
        self.seed = int(seed)
        self.transform = getattr(base_dataset, "transform", None)
        self.target_transform = getattr(base_dataset, "target_transform", None)
        if hasattr(base_dataset, "transform"):
            base_dataset.transform = None
        if hasattr(base_dataset, "target_transform"):
            base_dataset.target_transform = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        corruption_type = int(self.corruption_types[idx])
        if corruption_type >= 0:
            seed_sequence = np.random.SeedSequence(
                [
                    int(self.seed),
                    int(idx),
                    int(corruption_type),
                    DATASET_NUMERIC_ID[self.dataset_name],
                ]
            )
            rng = np.random.default_rng(seed_sequence)
            image = corruption_opt.apply_corruption(image, corruption_type, rng=rng)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __getattr__(self, name: str):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)


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

    mask = load_selection_mask(
        dataset_name,
        mode,
        keep_ratio,
        seed,
        model_name,
    )
    if mask.shape[0] != len(dataset):
        raise ValueError(
            f"mask 长度与训练集长度不一致: mask={mask.shape[0]}, train={len(dataset)}, "
            f"dataset={dataset_name}, mode={mode}, kr={keep_ratio}, seed={seed}"
        )
    mask = np.asarray(mask).astype(bool)
    return np.flatnonzero(mask)


def _selected_noise_stats(selected_indices: np.ndarray, noise_info: dict[str, object] | None) -> dict[str, object]:
    if noise_info is None:
        return {}

    is_noisy = np.asarray(noise_info["is_noisy"], dtype=bool)
    selected_is_noisy = is_noisy[selected_indices]
    num_selected = int(selected_indices.shape[0])
    num_noisy_selected = int(np.sum(selected_is_noisy))
    return {
        "num_noisy_selected": num_noisy_selected,
        "noise_ratio_in_selection": float(num_noisy_selected / num_selected) if num_selected else 0.0,
    }



def _selected_corruption_stats(
    selected_indices: np.ndarray,
    corruption_info: dict[str, object] | None,
) -> dict[str, object]:
    if corruption_info is None:
        return {}

    is_corrupted = np.asarray(corruption_info["is_corrupted"], dtype=bool)
    corruption_types = np.asarray(corruption_info["corruption_types"], dtype=np.int16)
    selected_is_corrupted = is_corrupted[selected_indices]
    selected_types = corruption_types[selected_indices][selected_is_corrupted]
    num_selected = int(selected_indices.shape[0])
    num_corrupted_selected = int(np.sum(selected_is_corrupted))
    return {
        "num_corrupted_selected": num_corrupted_selected,
        "corruption_ratio_in_selection": (
            float(num_corrupted_selected / num_selected) if num_selected else 0.0
        ),
        "selected_corruption_type_counts": _corruption_type_counts(selected_types),
    }


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    del multi_seed  # kept for backward-compatible function signature

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

    noise_info: dict[str, object] | None = None
    corruption_info: dict[str, object] | None = None
    if args.exp == "label_noise":
        noise_info = apply_label_noise_to_dataset(
            dataset=train_dataset,
            dataset_name=args.dataset,
            seed=seed,
            noise_root=Path(args.noise_root),
            num_classes=data_loader.num_classes,
            strict=args.strict_noise_check,
        )
        tqdm.write(
            f"Applied label noise: dataset={args.dataset}, seed={seed}, "
            f"num_noisy={noise_info['num_noisy']}/{noise_info['num_total']} "
            f"({noise_info['noise_ratio']:.4f}), path={noise_info['noise_path']}"
        )
    elif args.exp == "corruption":
        corruption_info = load_corruption_info(
            dataset_name=args.dataset,
            seed=seed,
            corruption_root=Path(args.corruption_root),
            num_samples=len(train_dataset),
            strict=args.strict_corruption_check,
        )
        train_dataset = FixedCorruptionDataset(
            base_dataset=train_dataset,
            corruption_types=np.asarray(corruption_info["corruption_types"], dtype=np.int16),
            dataset_name=args.dataset,
            seed=seed,
        )
        tqdm.write(
            f"Applied image corruption: dataset={args.dataset}, seed={seed}, "
            f"num_corrupted={corruption_info['num_corrupted']}/{corruption_info['num_total']}, "
            f"ratio={corruption_info['corruption_ratio']:.4f}, "
            f"path={corruption_info['corruption_path']}, "
            f"type_counts={corruption_info['type_counts']}"
        )

    model_name = args.model
    keep_ratios = parse_ratio_list(args.kr)
    model_factory = get_model(model_name)

    result_root = _root_for_exp(args.result_root, args.exp)
    checkpoint_root = _root_for_exp(args.checkpoint_root, args.exp)
    for keep_ratio in keep_ratios:
        # Result files are model-sensitive and therefore include the model name.
        result_path = resolve_result_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            keep_ratio=keep_ratio,
            root=result_root,
        )
        result_dir = result_path.parent
        checkpoint_path = resolve_checkpoint_path(
            mode=args.mode,
            dataset=args.dataset,
            model=model_name,
            seed=seed,
            keep_ratio=keep_ratio,
            root=checkpoint_root,
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
        noise_selection_stats = _selected_noise_stats(selected_indices, noise_info)
        corruption_selection_stats = _selected_corruption_stats(selected_indices, corruption_info)
        if noise_selection_stats:
            tqdm.write(
                f"Selection stats: mode={args.mode}, kr={keep_ratio}, "
                f"num_selected={len(selected_indices)}, "
                f"noisy_selected={noise_selection_stats['num_noisy_selected']}, "
                f"noise_ratio_in_selection={noise_selection_stats['noise_ratio_in_selection']:.4f}"
            )
        if corruption_selection_stats:
            tqdm.write(
                f"Selection stats: mode={args.mode}, kr={keep_ratio}, "
                f"num_selected={len(selected_indices)}, "
                f"corrupted_selected={corruption_selection_stats['num_corrupted_selected']}, "
                f"corruption_ratio_in_selection="
                f"{corruption_selection_stats['corruption_ratio_in_selection']:.4f}, "
                f"type_counts={corruption_selection_stats['selected_corruption_type_counts']}"
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
                checkpoint_payload = {
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
                    "exp": args.exp,
                }
                if noise_info is not None:
                    checkpoint_payload["noise_path"] = noise_info["noise_path"]
                if corruption_info is not None:
                    checkpoint_payload["corruption_path"] = corruption_info["corruption_path"]
                    checkpoint_payload["num_corrupted"] = corruption_info["num_corrupted"]
                torch.save(checkpoint_payload, checkpoint_path)
            epoch_bar.update(1)

        total_time = time.time() - start_time
        accuracy = float(np.mean(accuracy_samples)) if accuracy_samples else 0.0
        accuracy_std = float(np.std(accuracy_samples, ddof=0)) if accuracy_samples else 0.0
        accuracy = round(accuracy, 4)
        accuracy_std = round(accuracy_std, 4)

        result_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset": args.dataset,
            "data_root": str(Path(args.data_root)),
            "exp": args.exp,
            "result_root": str(result_root),
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
        }
        if noise_info is not None:
            metadata.update(
                {
                    "noise_root": str(Path(args.noise_root)),
                    "noise_path": noise_info["noise_path"],
                    "noise_rate": noise_info["noise_ratio"],
                    "num_noisy_total": int(noise_info["num_noisy"]),
                    "num_noisy_selected": int(noise_selection_stats["num_noisy_selected"]),
                    "noise_ratio_in_selection": float(noise_selection_stats["noise_ratio_in_selection"]),
                }
            )
        if corruption_info is not None:
            metadata.update(
                {
                    "corruption_root": str(Path(args.corruption_root)),
                    "corruption_path": corruption_info["corruption_path"],
                    "corruption_rate": corruption_info["corruption_ratio"],
                    "num_corrupted_total": int(corruption_info["num_corrupted"]),
                    "corruption_type_counts": corruption_info["type_counts"],
                    "num_corrupted_selected": int(
                        corruption_selection_stats["num_corrupted_selected"]
                    ),
                    "corruption_ratio_in_selection": float(
                        corruption_selection_stats["corruption_ratio_in_selection"]
                    ),
                    "selected_corruption_type_counts": corruption_selection_stats[
                        "selected_corruption_type_counts"
                    ],
                }
            )

        result_payload = {
            "metadata": metadata,
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
    if args.exp == "corruption" and args.dataset not in CORRUPTION_SUPPORTED_DATASETS:
        supported = ", ".join(sorted(CORRUPTION_SUPPORTED_DATASETS))
        raise ValueError(f"--exp corruption 仅支持数据集: {supported}; 当前 dataset={args.dataset}")
    args = apply_dataset_defaults(args)
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)


if __name__ == "__main__":
    main()
