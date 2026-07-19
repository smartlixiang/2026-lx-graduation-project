#!/usr/bin/env python3
"""Generate fixed image-corruption manifests for CIFAR-100 and Tiny-ImageNet.

Expected repository layout
--------------------------
2026-lx-graduation-project/
├── data/
│   ├── cifar-100-python/
│   └── tiny-imagenet-200/train/
├── corruption_exp/
│   └── corruption_opt.py
└── generate_corruption_list.py

For each dataset and seed, the script randomly selects 20% of the full training
set without replacement, divides the selected indices equally among five
corruption types, sorts indices inside each type, and writes:

    corruption_data/<dataset>/corruption_list_<seed>.txt

Each line contains two integers without a header:

    sample_id corruption_type

Corruption type IDs are defined centrally in corruption_exp/corruption_opt.py.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from corruption_exp.corruption_opt import (
    CORRUPTION_ID_TO_NAME,
    NUM_CORRUPTION_TYPES,
)


DEFAULT_DATASETS = ("cifar100", "tiny-imagenet")
DEFAULT_SEEDS = (22, 42, 96)
DEFAULT_CORRUPTION_RATIO = 0.2

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}

DATASET_ALIASES = {
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
    "tiny-imagenet": "tiny-imagenet",
    "tiny_imagenet": "tiny-imagenet",
    "tinyimagenet": "tiny-imagenet",
}


def parse_csv(value: str) -> list[str]:
    """Parse a comma-separated argument and remove empty fields."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv(value)]
    if not seeds:
        raise ValueError("At least one random seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds


def normalize_dataset_names(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw_name in values:
        key = raw_name.strip().lower()
        if key not in DATASET_ALIASES:
            raise ValueError(
                f"Unsupported dataset '{raw_name}'. "
                f"Supported datasets: {sorted(set(DATASET_ALIASES.values()))}"
            )
        name = DATASET_ALIASES[key]
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one dataset is required.")
    return normalized


def get_cifar100_train_size(data_root: Path) -> int:
    """Read the CIFAR-100 training-label count without loading image arrays."""
    train_file = data_root / "cifar-100-python" / "train"
    if not train_file.is_file():
        raise FileNotFoundError(
            "CIFAR-100 training file not found. Expected: " f"{train_file}"
        )

    with train_file.open("rb") as handle:
        entry = pickle.load(handle, encoding="latin1")

    labels = entry.get("fine_labels")
    if labels is None:
        labels = entry.get(b"fine_labels")
    if labels is None:
        raise ValueError(f"Cannot find fine_labels in {train_file}")

    num_samples = len(labels)
    if num_samples != 50_000:
        raise ValueError(
            f"Unexpected CIFAR-100 training size: {num_samples}; expected 50000."
        )
    return num_samples


def iter_imagefolder_paths(root: Path) -> Iterable[Path]:
    """Yield image paths in torchvision ImageFolder-compatible sorted order."""
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {root}")

    for class_dir in class_dirs:
        for current_root, dirnames, filenames in os.walk(class_dir):
            dirnames.sort()
            current = Path(current_root)
            for filename in sorted(filenames):
                path = current / filename
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield path


def get_tiny_imagenet_train_size(data_root: Path) -> int:
    """Count Tiny-ImageNet samples using ImageFolder-compatible ordering."""
    train_root = data_root / "tiny-imagenet-200" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            "Tiny-ImageNet training directory not found. Expected: " f"{train_root}"
        )

    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if len(class_dirs) != 200:
        raise ValueError(
            f"Unexpected Tiny-ImageNet class count: {len(class_dirs)}; expected 200."
        )

    num_samples = sum(1 for _ in iter_imagefolder_paths(train_root))
    if num_samples != 100_000:
        raise ValueError(
            f"Unexpected Tiny-ImageNet training size: {num_samples}; expected 100000."
        )
    return num_samples


def get_dataset_size(dataset: str, data_root: Path) -> int:
    if dataset == "cifar100":
        return get_cifar100_train_size(data_root)
    if dataset == "tiny-imagenet":
        return get_tiny_imagenet_train_size(data_root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_corruption_rows(
    num_samples: int,
    seed: int,
    corruption_ratio: float,
) -> np.ndarray:
    """Return [sample_id, corruption_type] rows grouped by corruption type."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not 0.0 < corruption_ratio < 1.0:
        raise ValueError(
            f"corruption_ratio must be in (0, 1), got {corruption_ratio}."
        )

    exact_num_corrupted = num_samples * corruption_ratio
    num_corrupted = int(round(exact_num_corrupted))
    if not np.isclose(exact_num_corrupted, num_corrupted, atol=1e-9):
        raise ValueError(
            "num_samples * corruption_ratio must be an integer, got "
            f"{num_samples} * {corruption_ratio} = {exact_num_corrupted}."
        )
    if num_corrupted % NUM_CORRUPTION_TYPES != 0:
        raise ValueError(
            f"{num_corrupted} corrupted samples cannot be divided equally among "
            f"{NUM_CORRUPTION_TYPES} corruption types."
        )

    per_type = num_corrupted // NUM_CORRUPTION_TYPES
    rng = np.random.default_rng(seed)
    selected = rng.permutation(num_samples)[:num_corrupted]

    blocks: list[np.ndarray] = []
    for corruption_type in range(NUM_CORRUPTION_TYPES):
        start = corruption_type * per_type
        end = start + per_type
        sample_ids = np.sort(selected[start:end]).astype(np.int64)
        type_ids = np.full(per_type, corruption_type, dtype=np.int64)
        blocks.append(np.column_stack((sample_ids, type_ids)))

    rows = np.concatenate(blocks, axis=0)
    validate_corruption_rows(rows, num_samples, corruption_ratio)
    return rows


def validate_corruption_rows(
    rows: np.ndarray,
    num_samples: int,
    corruption_ratio: float,
) -> None:
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(f"Corruption rows must have shape [N, 2], got {rows.shape}.")

    sample_ids = rows[:, 0].astype(np.int64)
    type_ids = rows[:, 1].astype(np.int64)
    expected_total = int(round(num_samples * corruption_ratio))
    expected_per_type = expected_total // NUM_CORRUPTION_TYPES

    if rows.shape[0] != expected_total:
        raise ValueError(
            f"Unexpected row count: {rows.shape[0]}; expected {expected_total}."
        )
    if np.unique(sample_ids).size != expected_total:
        raise ValueError("A training sample was assigned more than one corruption.")
    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError("The corruption list contains out-of-range sample IDs.")
    if np.any(type_ids < 0) or np.any(type_ids >= NUM_CORRUPTION_TYPES):
        raise ValueError("The corruption list contains invalid corruption type IDs.")

    for corruption_type in range(NUM_CORRUPTION_TYPES):
        positions = np.flatnonzero(type_ids == corruption_type)
        if positions.size != expected_per_type:
            raise ValueError(
                f"Corruption type {corruption_type} has {positions.size} samples; "
                f"expected {expected_per_type}."
            )

        expected_positions = np.arange(
            corruption_type * expected_per_type,
            (corruption_type + 1) * expected_per_type,
        )
        if not np.array_equal(positions, expected_positions):
            raise ValueError(
                f"Rows for corruption type {corruption_type} are not contiguous."
            )

        block_ids = sample_ids[positions]
        if block_ids.size > 1 and np.any(block_ids[1:] < block_ids[:-1]):
            raise ValueError(
                f"Sample IDs for corruption type {corruption_type} are not sorted."
            )


def write_corruption_list(
    output_path: Path,
    rows: np.ndarray,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite to replace existing lists."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, rows, fmt="%d %d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed 20% image-corruption lists for CIFAR-100 and "
            "Tiny-ImageNet."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets. Default: cifar100,tiny-imagenet",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds. Default: 22,42,96",
    )
    parser.add_argument(
        "--corruption-ratio",
        type=float,
        default=DEFAULT_CORRUPTION_RATIO,
        help="Total corrupted fraction of the full training set. Default: 0.2",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Original dataset root. Default: data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("corruption_data"),
        help="Output root for corruption lists. Default: corruption_data",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing corruption_list_<seed>.txt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = normalize_dataset_names(parse_csv(args.datasets))
    seeds = parse_seeds(args.seeds)

    print("Corruption type mapping:")
    for type_id in range(NUM_CORRUPTION_TYPES):
        print(f"  {type_id}: {CORRUPTION_ID_TO_NAME[type_id]}")

    for dataset in datasets:
        num_samples = get_dataset_size(dataset, args.data_root)
        num_corrupted = int(round(num_samples * args.corruption_ratio))
        per_type = num_corrupted // NUM_CORRUPTION_TYPES

        for seed in seeds:
            rows = build_corruption_rows(
                num_samples=num_samples,
                seed=seed,
                corruption_ratio=args.corruption_ratio,
            )
            output_path = args.output_root / dataset / f"corruption_list_{seed}.txt"
            write_corruption_list(output_path, rows, args.overwrite)
            print(
                f"[saved] dataset={dataset} seed={seed} "
                f"total={num_corrupted} per_type={per_type} path={output_path}"
            )


if __name__ == "__main__":
    main()
