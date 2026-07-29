#!/usr/bin/env python3
"""Generate reproducible, stratified unseen-sample index lists."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torchvision import datasets

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_DATASETS = ("cifar10", "cifar100", "tiny-imagenet")


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seed must contain integers separated by commas") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seed cannot be empty")
    return seeds


def load_train_labels(dataset: str, data_root: str | Path) -> np.ndarray:
    root = Path(data_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if dataset == "cifar10":
        ds = datasets.CIFAR10(str(root), train=True, download=True)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(str(root), train=True, download=True)
    elif dataset == "tiny-imagenet":
        train_root = root / "tiny-imagenet-200" / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_root}")
        ds = datasets.ImageFolder(str(train_root))
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return np.asarray(ds.targets, dtype=np.int64)


def largest_remainder_quotas(labels: np.ndarray, unseen_ratio: int) -> dict[int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not 0 <= unseen_ratio <= 100:
        raise ValueError("labels must be one-dimensional and ratio must be in [0, 100]")
    classes, counts = np.unique(labels, return_counts=True)
    target = round(len(labels) * unseen_ratio / 100)
    raw = counts.astype(np.float64) * unseen_ratio / 100.0
    quotas = np.floor(raw).astype(np.int64)
    remaining = target - int(quotas.sum())
    # Class id is a deterministic tie breaker.
    order = np.lexsort((classes, -(raw - quotas)))
    for pos in order[:remaining]:
        quotas[pos] += 1
    return {int(cls): int(quota) for cls, quota in zip(classes, quotas)}


def generate_unseen_indices(labels: np.ndarray, unseen_ratio: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    quotas = largest_remainder_quotas(labels, unseen_ratio)
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(labels == cls), size=n, replace=False)
              for cls, n in quotas.items() if n]
    result = np.sort(np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64))
    return result.astype(np.int64)


def load_valid_unseen_list(path: Path, labels: np.ndarray, unseen_ratio: int) -> np.ndarray | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(not line.strip() or not line.strip().lstrip("+-").isdigit() for line in lines):
            return None
        values = np.asarray([int(line) for line in lines], dtype=np.int64)
    except (OSError, ValueError):
        return None
    expected = round(len(labels) * unseen_ratio / 100)
    if (values.ndim != 1 or len(values) != expected or np.unique(values).size != len(values)
            or np.any(values < 0) or np.any(values >= len(labels))
            or not np.array_equal(values, np.sort(values))):
        return None
    quotas = largest_remainder_quotas(labels, unseen_ratio)
    actual = {cls: int(np.sum(labels[values] == cls)) for cls in quotas}
    return values if actual == quotas else None


def unseen_list_path(dataset: str, unseen_ratio: int, seed: int) -> Path:
    return PROJECT_ROOT / "unseen_data" / dataset / str(unseen_ratio) / f"unseen_list_{seed}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--unseen-ratio", required=True, type=int)
    parser.add_argument("--seed", required=True, type=parse_seeds)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--skip-saved", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.unseen_ratio <= 100:
        parser.error("--unseen-ratio must be an integer percentage in [0, 100]")
    labels = load_train_labels(args.dataset, args.data_root)
    for seed in args.seed:
        path = unseen_list_path(args.dataset, args.unseen_ratio, seed)
        if args.skip_saved and load_valid_unseen_list(path, labels, args.unseen_ratio) is not None:
            print(f"[Skip] valid unseen list: {path}")
            continue
        indices = generate_unseen_indices(labels, args.unseen_ratio, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{index}\n" for index in indices), encoding="utf-8")
        print(f"[Save] {len(indices)} unseen indices: {path}")


if __name__ == "__main__":
    main()
