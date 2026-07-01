#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate fixed label-noise lists for selection experiments.

The output format is intentionally the same as the existing CIFAR label-noise
files used by the project:

    noise/{dataset}/noise_list_{seed}.txt

Each txt file has no header and contains two integer columns:

    sample_id noisy_label

``sample_id`` is the index in the original training-set order.  For
Tiny-ImageNet this order is the order returned by torchvision.datasets.ImageFolder
on data/tiny-imagenet-200/train, which is also the order used by the project
dataset loader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from utils.seed import parse_seed_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed label-noise txt files.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=TINY_IMAGENET,
        choices=AVAILABLE_DATASETS,
        help="Dataset name. Default: tiny-imagenet.",
    )
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--noise-root", type=str, default=str(PROJECT_ROOT / "noise"))
    parser.add_argument(
        "--seed",
        type=str,
        default="22,42,96",
        help="Seed list, e.g. 22,42,96.",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.2,
        help="Fraction of training samples whose labels are changed. Default: 0.2.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="classwise",
        choices=("classwise", "global"),
        help=(
            "classwise: inject approximately the same ratio per class; "
            "global: sample the required number from all training samples."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing noise_list_{seed}.txt files.",
    )
    parser.add_argument(
        "--save-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save a JSON summary for each generated txt file.",
    )
    return parser.parse_args()


def build_train_dataset(dataset_name: str, data_root: Path):
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=None)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=None)
    if dataset_name == TINY_IMAGENET:
        train_root = data_root / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(
                f"Tiny-ImageNet train split not found: {train_root}. "
                "Expected data/tiny-imagenet-200/train."
            )
        return datasets.ImageFolder(root=str(train_root), transform=None)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        values = getattr(dataset, "targets")
        if len(values) == len(dataset):
            return np.asarray(values, dtype=np.int64)
    if hasattr(dataset, "labels"):
        values = getattr(dataset, "labels")
        if len(values) == len(dataset):
            return np.asarray(values, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([label for _, label in dataset.samples], dtype=np.int64)
    return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=np.int64)


def allocate_classwise_counts(labels: np.ndarray, noise_rate: float, num_total_noisy: int) -> dict[int, int]:
    """Allocate classwise noisy counts while keeping the exact global count."""
    classes = np.unique(labels)
    class_sizes = np.asarray([int(np.sum(labels == c)) for c in classes], dtype=np.int64)
    raw = class_sizes.astype(np.float64) * float(noise_rate)
    counts = np.floor(raw).astype(np.int64)
    counts = np.minimum(counts, class_sizes)

    need = int(num_total_noisy - counts.sum())
    if need > 0:
        frac = raw - counts.astype(np.float64)
        order = np.lexsort((classes.astype(np.int64), -frac))
        for pos in order:
            if need <= 0:
                break
            if counts[pos] < class_sizes[pos]:
                counts[pos] += 1
                need -= 1
    elif need < 0:
        frac = raw - counts.astype(np.float64)
        order = np.lexsort((classes.astype(np.int64), frac))
        for pos in order:
            if need >= 0:
                break
            if counts[pos] > 0:
                counts[pos] -= 1
                need += 1

    if int(counts.sum()) != int(num_total_noisy):
        raise RuntimeError("Failed to allocate classwise noisy counts.")

    return {int(cls): int(cnt) for cls, cnt in zip(classes.tolist(), counts.tolist())}


def sample_noisy_indices(labels: np.ndarray, noise_rate: float, strategy: str, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < noise_rate < 1.0):
        raise ValueError("--noise-rate must be in (0, 1).")

    num_samples = int(labels.shape[0])
    num_total_noisy = int(round(num_samples * float(noise_rate)))
    num_total_noisy = min(num_samples, max(1, num_total_noisy))

    if strategy == "global":
        return np.sort(rng.choice(num_samples, size=num_total_noisy, replace=False).astype(np.int64))

    counts = allocate_classwise_counts(labels, noise_rate, num_total_noisy)
    selected: list[int] = []
    for class_id in sorted(counts):
        count = counts[class_id]
        if count <= 0:
            continue
        class_indices = np.flatnonzero(labels == class_id)
        chosen = rng.choice(class_indices, size=count, replace=False)
        selected.extend(int(x) for x in chosen.tolist())

    return np.sort(np.asarray(selected, dtype=np.int64))


def sample_new_labels(labels: np.ndarray, sample_ids: np.ndarray, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample labels from all classes except the clean label."""
    clean = labels[sample_ids].astype(np.int64)
    raw = rng.integers(0, num_classes - 1, size=sample_ids.shape[0], dtype=np.int64)

    new_labels = raw + (raw >= clean).astype(np.int64)
    if np.any(new_labels == clean):
        raise RuntimeError("Internal error: generated noisy labels equal clean labels.")
    return new_labels.astype(np.int64)


def generate_for_seed(
    *,
    dataset_name: str,
    labels: np.ndarray,
    num_classes: int,
    seed: int,
    noise_rate: float,
    strategy: str,
    out_dir: Path,
    overwrite: bool,
    save_meta: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_list_{seed}.txt"
    meta_path = out_dir / f"noise_list_{seed}.json"

    if out_path.exists() and not overwrite:
        print(f"[skip] existing noise file: {out_path}")
        return out_path

    rng = np.random.default_rng(seed)
    sample_ids = sample_noisy_indices(labels, noise_rate, strategy, rng)
    new_labels = sample_new_labels(labels, sample_ids, num_classes, rng)

    mapping = np.stack([sample_ids, new_labels], axis=1).astype(np.int64)
    np.savetxt(out_path, mapping, fmt="%d")

    if save_meta:
        clean_selected = labels[sample_ids]
        per_class_counts = {
            int(cls): int(np.sum(clean_selected == cls))
            for cls in np.unique(labels).tolist()
        }
        meta = {
            "dataset": dataset_name,
            "seed": int(seed),
            "noise_rate": float(noise_rate),
            "strategy": strategy,
            "num_samples": int(labels.shape[0]),
            "num_classes": int(num_classes),
            "num_noisy": int(sample_ids.shape[0]),
            "actual_noise_rate": float(sample_ids.shape[0] / labels.shape[0]),
            "format": "two columns without header: sample_id noisy_label",
            "path": str(out_path),
            "per_clean_class_noisy_counts": per_class_counts,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[saved] {out_path} | dataset={dataset_name} seed={seed} "
        f"num_noisy={sample_ids.shape[0]}/{labels.shape[0]} "
        f"rate={sample_ids.shape[0] / labels.shape[0]:.4f}"
    )
    return out_path


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    data_root = Path(args.data_root)
    noise_root = Path(args.noise_root)

    dataset = build_train_dataset(dataset_name, data_root)
    labels = extract_labels(dataset)
    num_classes = len(getattr(dataset, "classes", np.unique(labels)))

    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("--seed cannot be empty.")

    print(
        f"[dataset] {dataset_name} | samples={len(dataset)} | classes={num_classes} | "
        f"labels_shape={labels.shape}"
    )
    out_dir = noise_root / dataset_name
    for seed in seeds:
        generate_for_seed(
            dataset_name=dataset_name,
            labels=labels,
            num_classes=num_classes,
            seed=int(seed),
            noise_rate=float(args.noise_rate),
            strategy=args.strategy,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
            save_meta=bool(args.save_meta),
        )


if __name__ == "__main__":
    main()
