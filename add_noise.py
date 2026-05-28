#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate fixed label-noise lists for CIFAR-10 and CIFAR-100.

This script does not modify the original dataset files. It only writes txt files:

    noise/cifar10/noise_list_22.txt
    noise/cifar10/noise_list_42.txt
    noise/cifar10/noise_list_96.txt
    noise/cifar100/noise_list_22.txt
    noise/cifar100/noise_list_42.txt
    noise/cifar100/noise_list_96.txt

Each txt file has two integer columns without header:

    sample_id noisy_label

The sample_id follows the training-set order of torchvision.datasets.CIFAR10/CIFAR100.
The noisy_label is guaranteed to be different from the original clean label.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS = ("cifar10", "cifar100")
SEEDS = (22, 42, 96)
NOISE_RATE = 0.20

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "noise"


def set_all_seeds(seed: int) -> None:
    """Set all available random seeds used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # This script itself does not depend on torch. Ignore torch-related issues.
        pass


def load_train_targets(dataset_name: str, data_root: Path) -> tuple[np.ndarray, int]:
    """Load clean training targets and number of classes."""
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=str(data_root),
            train=True,
            download=True,
            transform=None,
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=True,
            transform=None,
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    targets = np.asarray(dataset.targets, dtype=np.int64)
    if targets.ndim != 1:
        raise RuntimeError(f"Unexpected targets shape for {dataset_name}: {targets.shape}")

    return targets, num_classes


def generate_noise_mapping(
    clean_targets: np.ndarray,
    num_classes: int,
    noise_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a two-column array: [sample_id, noisy_label].

    The same seed controls both:
      1. which samples are selected for label corruption;
      2. what wrong label each selected sample is changed to.
    """
    if not (0.0 < noise_rate < 1.0):
        raise ValueError(f"noise_rate must be in (0, 1), got {noise_rate}")

    set_all_seeds(seed)
    rng = np.random.default_rng(seed)

    num_samples = int(clean_targets.shape[0])
    num_noisy = int(round(num_samples * noise_rate))

    if num_noisy <= 0:
        raise RuntimeError("num_noisy is zero; check noise_rate and dataset size.")

    sample_ids = rng.choice(num_samples, size=num_noisy, replace=False)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)

    old_labels = clean_targets[sample_ids]

    # Uniformly sample from num_classes - 1 wrong labels.
    # Draw values in [0, num_classes - 2], then shift by +1 when the draw
    # is greater than or equal to the original label. This guarantees:
    #     noisy_label != clean_label
    raw_new_labels = rng.integers(
        low=0,
        high=num_classes - 1,
        size=num_noisy,
        dtype=np.int64,
    )
    noisy_labels = raw_new_labels + (raw_new_labels >= old_labels).astype(np.int64)

    if np.any(noisy_labels == old_labels):
        bad = int(np.sum(noisy_labels == old_labels))
        raise RuntimeError(f"Found {bad} unchanged labels after noise generation.")

    # Sort by sample_id for readability and stable diff in git/logs.
    order = np.argsort(sample_ids, kind="mergesort")
    mapping = np.stack([sample_ids[order], noisy_labels[order]], axis=1).astype(np.int64)

    return mapping


def validate_mapping(
    mapping: np.ndarray,
    clean_targets: np.ndarray,
    num_classes: int,
    expected_noise_rate: float,
) -> None:
    """Validate shape, id range, label range, uniqueness, and no unchanged labels."""
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError(f"Noise mapping must have shape (N, 2), got {mapping.shape}")

    sample_ids = mapping[:, 0]
    noisy_labels = mapping[:, 1]

    num_samples = int(clean_targets.shape[0])
    expected_num_noisy = int(round(num_samples * expected_noise_rate))

    if mapping.shape[0] != expected_num_noisy:
        raise ValueError(
            f"Unexpected noisy sample count: got {mapping.shape[0]}, "
            f"expected {expected_num_noisy}"
        )

    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError("sample_id out of range.")

    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample_id found in noise mapping.")

    if np.any(noisy_labels < 0) or np.any(noisy_labels >= num_classes):
        raise ValueError("noisy_label out of range.")

    clean_labels = clean_targets[sample_ids]
    if np.any(noisy_labels == clean_labels):
        bad = int(np.sum(noisy_labels == clean_labels))
        raise ValueError(f"{bad} noisy labels are equal to clean labels.")


def save_mapping_txt(mapping: np.ndarray, output_path: Path) -> None:
    """Save two-column integer txt without header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, mapping, fmt="%d")


def load_existing_mapping(path: Path) -> np.ndarray:
    """Load an existing two-column txt file robustly."""
    mapping = np.loadtxt(path, dtype=np.int64)
    if mapping.ndim == 1:
        mapping = mapping.reshape(1, 2)
    return mapping


def parse_seed_list(seed_text: str | None) -> list[int]:
    if seed_text is None or not seed_text.strip():
        return list(SEEDS)
    return [int(item.strip()) for item in seed_text.split(",") if item.strip()]


def parse_dataset_list(dataset_text: str | None) -> list[str]:
    if dataset_text is None or not dataset_text.strip():
        return list(DATASETS)
    return [item.strip().lower() for item in dataset_text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed 20% symmetric label-noise txt files for CIFAR datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated dataset names. Default: cifar10,cifar100",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds. Default: 22,42,96",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=NOISE_RATE,
        help="Noise rate. Default: 0.20",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root. Default: ./data",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root. Default: ./noise",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing noise_list files. By default existing files are kept unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_names = parse_dataset_list(args.datasets)
    seeds = parse_seed_list(args.seeds)
    noise_rate = float(args.noise_rate)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    print("[Config]")
    print(f"  datasets    = {dataset_names}")
    print(f"  seeds       = {seeds}")
    print(f"  noise_rate  = {noise_rate:.4f}")
    print(f"  data_root   = {data_root}")
    print(f"  output_root = {output_root}")
    print(f"  overwrite   = {args.overwrite}")
    print()

    for dataset_name in dataset_names:
        clean_targets, num_classes = load_train_targets(dataset_name, data_root)
        num_samples = int(clean_targets.shape[0])
        expected_num_noisy = int(round(num_samples * noise_rate))

        print(f"[Dataset] {dataset_name}")
        print(f"  num_samples        = {num_samples}")
        print(f"  num_classes        = {num_classes}")
        print(f"  expected_num_noisy = {expected_num_noisy}")
        print()

        for seed in seeds:
            output_path = output_root / dataset_name / f"noise_list_{seed}.txt"

            if output_path.exists() and not args.overwrite:
                mapping = load_existing_mapping(output_path)
                validate_mapping(
                    mapping=mapping,
                    clean_targets=clean_targets,
                    num_classes=num_classes,
                    expected_noise_rate=noise_rate,
                )
                print(f"[Keep] {output_path} already exists and passed validation.")
                continue

            mapping = generate_noise_mapping(
                clean_targets=clean_targets,
                num_classes=num_classes,
                noise_rate=noise_rate,
                seed=seed,
            )
            validate_mapping(
                mapping=mapping,
                clean_targets=clean_targets,
                num_classes=num_classes,
                expected_noise_rate=noise_rate,
            )
            save_mapping_txt(mapping, output_path)

            actual_rate = mapping.shape[0] / num_samples
            print(
                f"[Save] dataset={dataset_name} | seed={seed} | "
                f"num_noisy={mapping.shape[0]} | actual_rate={actual_rate:.4f} | "
                f"path={output_path}"
            )

        print()

    print("[Done] All noise-list files are ready.")


if __name__ == "__main__":
    main()