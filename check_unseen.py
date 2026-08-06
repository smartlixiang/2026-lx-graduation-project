#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose the three internal terms of dynamic component A in unseen Exp. 1.

This script is intentionally specialized for the unseen-sample experiment 1 and
for comparing CIFAR-100 (the normal reference) with Tiny-ImageNet (the suspected
abnormal setting). It reads existing caches only and never modifies them.

Typical usage from the project root:

    python check_unseen_1.py
    python check_unseen_1.py --dataset tiny-imagenet --seed 22
    python check_unseen_1.py --dataset both --seed 22

Inputs:

    unseen_exp/proxy_logs/1/<dataset>/<proxy_model>/<seed>/<epochs>/fold_*.npz
    unseen_exp/dynamic_cache/1/<dataset>/<proxy_model>/<seed>/<epochs>/A.npz
    unseen_exp/static_scores/1/selection/<dataset>/<seed>/static_scores.npz

Main diagnostics:

1. Reconstruct A exactly from its three terms:
   - Boundary: mean p_y(1-p_y) in the early phase.
   - Gain: sqrt(mean_early(p_y) * mean_mid(p_y)) *
           (mean_mid(p_y) - mean_early(p_y)).
   - Stability: Var_late(CE) - Var_mid(CE).
2. Correlate each exact additive A contribution with class-standardized
   SA, Div, and DDS.
3. Locate suspicious single-epoch changes using true-label probability,
   support-weighted one-step gain, boundary information, and loss improvement.
4. Compare first/second boundary sensitivity without retraining the proxy model.
5. Compare learning-time behaviour between low/high static-score groups.

The boundary scan only changes how the fixed trajectory is partitioned. It does
not simulate a different MultiStepLR training trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
EXP_ID = 1
STATIC_NAMES = ("SA", "Div", "DDS")
STATIC_KEYS = ("sa", "div", "dds")
A_TERM_NAMES = ("Boundary", "Gain", "Stability")
SUPPORTED_DATASETS = ("cifar100", "tiny-imagenet")
EPS = 1e-12
ZSCORE_EPS = 1e-6
SAFE_ZSCORE_EPS = 1e-8


@dataclass(frozen=True)
class Target:
    dataset: str
    seed: int
    proxy_model: str
    epochs: int
    first_boundary: int
    second_boundary: int


@dataclass(frozen=True)
class CachePaths:
    proxy_dir: Path
    dynamic_dir: Path
    static_path: Path


@dataclass(frozen=True)
class NpyStreamInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool


@dataclass
class FoldTrajectory:
    fold_id: int
    train_indices: np.ndarray
    true_probability: np.ndarray  # (epochs, train samples), float32
    loss: np.ndarray              # (epochs, train samples), float32


@dataclass
class DecompositionResult:
    contributions: dict[str, np.ndarray]
    reconstructed_a: np.ndarray
    pearson_matrix: np.ndarray
    spearman_matrix: np.ndarray
    variance_shares: dict[str, float]
    saved_pearson: float
    saved_spearman: float
    saved_max_abs_error: float


@dataclass
class EpochDiagnostics:
    epochs: np.ndarray
    true_probability_spearman: np.ndarray
    boundary_spearman: np.ndarray
    one_step_gain_spearman: np.ndarray
    loss_improvement_spearman: np.ndarray
    mean_true_probability: np.ndarray
    mean_loss: np.ndarray
    mean_accuracy: np.ndarray
    final_true_probability: np.ndarray
    final_correct_rate: np.ndarray
    first_stable_correct_epoch: np.ndarray
    first_probability_05_epoch: np.ndarray
    first_probability_09_epoch: np.ndarray


@dataclass
class ScanPoint:
    boundary: int
    pearson_matrix: np.ndarray
    spearman_matrix: np.ndarray


@dataclass
class DatasetResult:
    target: Target
    static_class_z: dict[str, np.ndarray]
    saved_a: np.ndarray
    decomposition: DecompositionResult
    epoch_diagnostics: EpochDiagnostics
    first_scan: list[ScanPoint]
    second_scan: list[ScanPoint]
    output_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Boundary/Gain/Stability inside A for unseen experiment 1, "
            "with optional CIFAR-100 vs Tiny-ImageNet comparison."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("both", *SUPPORTED_DATASETS),
        default="both",
        help="Default: compare CIFAR-100 and Tiny-ImageNet.",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Exact epoch directory; valid only for a single dataset.",
    )
    parser.add_argument("--cifar100-epochs", type=int, default=None)
    parser.add_argument("--tiny-imagenet-epochs", type=int, default=None)
    parser.add_argument(
        "--boundaries",
        type=int,
        nargs=2,
        metavar=("FIRST", "SECOND"),
        default=None,
        help=(
            "Override phase boundaries for a single dataset. By default read the "
            "first two lr_milestones from proxy meta.json."
        ),
    )
    parser.add_argument(
        "--unseen-root",
        type=Path,
        default=Path("unseen_exp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("check_unseen_output"),
    )
    parser.add_argument(
        "--scan-radius",
        type=int,
        default=8,
        help="Boundary sensitivity scan radius in epochs.",
    )
    parser.add_argument(
        "--scan-step",
        type=int,
        default=2,
        help="Boundary sensitivity scan step.",
    )
    parser.add_argument(
        "--minimum-phase-length",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--top-k-epochs",
        type=int,
        default=8,
        help="Number of most suspicious one-step epochs printed.",
    )
    parser.add_argument(
        "--learn-patience",
        type=int,
        default=3,
        help="Consecutive epochs required for a sample to be stably correct.",
    )
    parser.add_argument(
        "--group-quantile",
        type=float,
        default=0.20,
        help="Fraction used for low/high static-score learning-time groups.",
    )
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.dataset == "both" and args.epochs is not None:
        parser.error("--epochs can only be used with one dataset")
    if args.dataset == "both" and args.boundaries is not None:
        parser.error("--boundaries can only be used with one dataset")
    for name in ("epochs", "cifar100_epochs", "tiny_imagenet_epochs"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.scan_radius < 0:
        parser.error("--scan-radius must be non-negative")
    if args.scan_step <= 0:
        parser.error("--scan-step must be positive")
    if args.minimum_phase_length <= 0:
        parser.error("--minimum-phase-length must be positive")
    if args.top_k_epochs <= 0:
        parser.error("--top-k-epochs must be positive")
    if args.learn_patience <= 0:
        parser.error("--learn-patience must be positive")
    if not 0.0 < args.group_quantile < 0.5:
        parser.error("--group-quantile must lie in (0, 0.5)")
    return args


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def numeric_epoch_dirs(root: Path, *, require_folds: bool) -> dict[int, Path]:
    if not root.is_dir():
        return {}
    result: dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        if require_folds and not any(path.glob("fold_*.npz")):
            continue
        result[int(path.name)] = path
    return result


def requested_epochs(args: argparse.Namespace, dataset: str) -> int | None:
    if args.dataset != "both" and args.epochs is not None:
        return int(args.epochs)
    if dataset == "cifar100" and args.cifar100_epochs is not None:
        return int(args.cifar100_epochs)
    if dataset == "tiny-imagenet" and args.tiny_imagenet_epochs is not None:
        return int(args.tiny_imagenet_epochs)
    return None


def resolve_cache_paths(
    args: argparse.Namespace,
    dataset: str,
) -> tuple[int, CachePaths]:
    unseen_root = resolve_project_path(args.unseen_root)
    proxy_seed_root = (
        unseen_root
        / "proxy_logs"
        / str(EXP_ID)
        / dataset
        / args.proxy_model
        / str(args.seed)
    )
    dynamic_seed_root = (
        unseen_root
        / "dynamic_cache"
        / str(EXP_ID)
        / dataset
        / args.proxy_model
        / str(args.seed)
    )

    exact = requested_epochs(args, dataset)
    if exact is None:
        proxy = numeric_epoch_dirs(proxy_seed_root, require_folds=True)
        dynamic = numeric_epoch_dirs(dynamic_seed_root, require_folds=False)
        common = sorted(set(proxy) & set(dynamic))
        common = [epoch for epoch in common if (dynamic[epoch] / "A.npz").is_file()]
        if not common:
            raise FileNotFoundError(
                "No shared epoch cache containing proxy folds and A.npz: "
                f"proxy={proxy_seed_root}, dynamic={dynamic_seed_root}"
            )
        epochs = common[-1]
    else:
        epochs = exact

    paths = CachePaths(
        proxy_dir=proxy_seed_root / str(epochs),
        dynamic_dir=dynamic_seed_root / str(epochs),
        static_path=(
            unseen_root
            / "static_scores"
            / str(EXP_ID)
            / "selection"
            / dataset
            / str(args.seed)
            / "static_scores.npz"
        ),
    )
    if not paths.proxy_dir.is_dir():
        raise FileNotFoundError(f"Proxy cache not found: {paths.proxy_dir}")
    if not (paths.dynamic_dir / "A.npz").is_file():
        raise FileNotFoundError(f"A cache not found: {paths.dynamic_dir / 'A.npz'}")
    if not paths.static_path.is_file():
        raise FileNotFoundError(f"Static cache not found: {paths.static_path}")
    return epochs, paths


def read_phase_boundaries(
    args: argparse.Namespace,
    dataset: str,
    epochs: int,
    proxy_dir: Path,
) -> tuple[int, int]:
    if args.dataset != "both" and args.boundaries is not None:
        first, second = map(int, args.boundaries)
    else:
        first = second = -1
        meta_path = proxy_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                milestones = [
                    int(value)
                    for value in meta.get("lr_milestones", [])
                    if 0 < int(value) < epochs
                ]
                if len(milestones) >= 2:
                    first, second = milestones[:2]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        if first < 0:
            known_defaults = {
                ("cifar100", 100): (40, 75),
                ("tiny-imagenet", 55): (25, 40),
                ("tiny-imagenet", 50): (20, 35),
            }
            if (dataset, epochs) in known_defaults:
                first, second = known_defaults[(dataset, epochs)]
            else:
                first = max(1, int(round(0.40 * epochs)))
                second = max(first + 1, int(round(0.75 * epochs)))

    if not (0 < first < second < epochs):
        raise ValueError(
            f"Invalid phase boundaries for {dataset}: first={first}, "
            f"second={second}, epochs={epochs}"
        )
    return first, second


def fold_sort_key(path: Path) -> tuple[float, str]:
    try:
        number = float(int(path.stem.split("_")[1]))
    except (IndexError, ValueError):
        number = math.inf
    return number, path.name


def _read_npy_header(stream: BinaryIO) -> NpyStreamInfo:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        reader = getattr(np.lib.format, "_read_array_header", None)
        if reader is None:
            raise ValueError("NPY 3.0 headers are unsupported by this NumPy")
        shape, fortran_order, dtype = reader(stream, version)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    return NpyStreamInfo(
        shape=tuple(int(value) for value in shape),
        dtype=np.dtype(dtype),
        fortran_order=bool(fortran_order),
    )


def _read_exact(stream: BinaryIO, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = num_bytes
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected end of NPY member; missing {remaining} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def inspect_logits_shape(npz_path: Path, array_name: str) -> tuple[int, int, int]:
    member = f"{array_name}.npy"
    with zipfile.ZipFile(npz_path, mode="r") as archive:
        if member not in archive.namelist():
            raise KeyError(f"{member} not found in {npz_path}")
        with archive.open(member, mode="r") as stream:
            info = _read_npy_header(stream)
    if len(info.shape) != 3:
        raise ValueError(f"{array_name} must be 3-D; got {info.shape}")
    return int(info.shape[0]), int(info.shape[1]), int(info.shape[2])


def iter_logits_epochs(npz_path: Path, array_name: str) -> Iterable[np.ndarray]:
    member = f"{array_name}.npy"
    with zipfile.ZipFile(npz_path, mode="r") as archive:
        if member not in archive.namelist():
            raise KeyError(f"{member} not found in {npz_path}")
        with archive.open(member, mode="r") as stream:
            info = _read_npy_header(stream)
            if len(info.shape) != 3:
                raise ValueError(f"{array_name} must be 3-D; got {info.shape}")
            if info.fortran_order:
                raise ValueError("Fortran-order logits are unsupported")
            num_epochs, num_samples, num_classes = info.shape
            values_per_epoch = num_samples * num_classes
            bytes_per_epoch = values_per_epoch * info.dtype.itemsize
            for _ in range(num_epochs):
                raw = _read_exact(stream, bytes_per_epoch)
                values = np.frombuffer(raw, dtype=info.dtype, count=values_per_epoch)
                yield values.reshape(num_samples, num_classes)


def standard_zscore(values: np.ndarray, eps: float = ZSCORE_EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("standard_zscore expects a 1-D vector")
    finite = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.float64)
    if int(finite.sum()) < 2:
        return output
    finite_values = values[finite]
    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values))
    if not np.isfinite(mean) or not np.isfinite(std) or std < eps:
        return output
    output[finite] = (finite_values - mean) / (std + eps)
    return output


def safe_standardize(values: np.ndarray, eps: float = SAFE_ZSCORE_EPS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(values)) if values.size else 0.0
    std = float(np.nanstd(values)) if values.size else 0.0
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / (std + eps)


def classwise_zscore(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros_like(values)
    for cls in np.unique(labels):
        mask = labels == cls
        output[mask] = standard_zscore(values[mask])
    return output


def rankdata(values: np.ndarray) -> np.ndarray:
    """Vectorized average ranks for ties (equivalent to scipy rankdata)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=np.float64)
    if n == 0:
        return ranks
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    counts = ends - starts
    average_ranks = 0.5 * (starts + ends - 1) + 1.0
    ranks[order] = np.repeat(average_ranks, counts)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Correlation vectors must share a shape")
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator < EPS:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def correlation_matrix(
    static_values: Mapping[str, np.ndarray],
    component_values: Mapping[str, np.ndarray],
    method: str,
) -> np.ndarray:
    if method == "pearson":
        left = static_values
        right = component_values
    elif method == "spearman":
        left = {name: rankdata(values) for name, values in static_values.items()}
        right = {name: rankdata(values) for name, values in component_values.items()}
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return np.asarray(
        [
            [pearson(left[sname], right[cname]) for cname in right]
            for sname in left
        ],
        dtype=np.float64,
    )


def load_a_and_static(
    paths: CachePaths,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    a_path = paths.dynamic_dir / "A.npz"
    with np.load(a_path, allow_pickle=False) as data:
        required = {"known_indices", "labels", "final_normalized"}
        if not required.issubset(data.files):
            raise KeyError(f"{a_path} missing {sorted(required - set(data.files))}")
        known = np.asarray(data["known_indices"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        saved_a = np.asarray(data["final_normalized"], dtype=np.float64)
    if known.ndim != 1 or labels.shape != known.shape or saved_a.shape != known.shape:
        raise ValueError("A cache arrays have inconsistent shapes")
    if np.unique(known).size != known.size:
        raise ValueError("known_indices contains duplicates")
    if not np.isfinite(saved_a).all():
        raise ValueError("Saved A contains NaN or infinity")

    with np.load(paths.static_path, allow_pickle=False) as data:
        required = set(STATIC_KEYS) | {"labels"}
        if not required.issubset(data.files):
            raise KeyError(
                f"{paths.static_path} missing {sorted(required - set(data.files))}"
            )
        full_labels = np.asarray(data["labels"], dtype=np.int64)
        full_scores = {
            name: np.asarray(data[key], dtype=np.float64)
            for name, key in zip(STATIC_NAMES, STATIC_KEYS)
        }
        n = len(full_labels)
        if any(values.shape != (n,) for values in full_scores.values()):
            raise ValueError("Static score arrays have inconsistent shapes")
        sample_indices = (
            np.asarray(data["sample_indices"], dtype=np.int64)
            if "sample_indices" in data.files
            else np.arange(n, dtype=np.int64)
        )

    if sample_indices.shape != (len(full_labels),):
        raise ValueError("static sample_indices shape mismatch")
    order = np.argsort(sample_indices)
    sorted_indices = sample_indices[order]
    positions = np.searchsorted(sorted_indices, known)
    if (
        np.any(positions >= len(sorted_indices))
        or not np.array_equal(sorted_indices[positions], known)
    ):
        raise ValueError("Static cache does not cover every known index")
    aligned_positions = order[positions]
    aligned_labels = full_labels[aligned_positions]
    if not np.array_equal(aligned_labels, labels):
        raise ValueError("Static labels do not match A-cache labels")
    static_raw = {
        name: values[aligned_positions] for name, values in full_scores.items()
    }
    static_class_z = {
        name: classwise_zscore(static_raw[name], labels) for name in STATIC_NAMES
    }
    return known, labels, saved_a, static_raw, static_class_z


def select_labels(indices: np.ndarray, known_labels: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("Fold indices must be non-negative 1-D integers")
    if indices.size and int(indices.max()) >= len(known_labels):
        raise ValueError(
            "Fold indices are not local to the known subset; this diagnostic "
            "expects unseen-exp proxy caches generated by the current protocol"
        )
    return known_labels[indices]


def true_probability_and_loss(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("Logits/labels shape mismatch")
    row_max = np.max(values, axis=1)
    shifted = values - row_max[:, None]
    log_sum_exp = row_max + np.log(np.exp(shifted).sum(axis=1))
    true_logits = values[np.arange(values.shape[0]), labels]
    loss = log_sum_exp - true_logits
    probability = np.exp(np.clip(true_logits - log_sum_exp, -80.0, 0.0))
    correct = np.argmax(values, axis=1) == labels
    return (
        probability.astype(np.float32),
        loss.astype(np.float32),
        correct.astype(np.uint8),
    )


def load_fold_trajectories(
    proxy_dir: Path,
    labels: np.ndarray,
    epochs: int,
) -> tuple[list[FoldTrajectory], np.ndarray, np.ndarray, np.ndarray]:
    fold_paths = sorted(proxy_dir.glob("fold_*.npz"), key=fold_sort_key)
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files in {proxy_dir}")

    num_samples = len(labels)
    probability_sum = np.zeros((epochs, num_samples), dtype=np.float32)
    loss_sum = np.zeros((epochs, num_samples), dtype=np.float32)
    correct_sum = np.zeros((epochs, num_samples), dtype=np.uint8)
    train_count = np.zeros(num_samples, dtype=np.uint8)
    folds: list[FoldTrajectory] = []

    for fold_id, fold_path in enumerate(fold_paths, start=1):
        print(f"[Proxy] reading train trajectory: {fold_path}")
        with np.load(fold_path, allow_pickle=False) as data:
            if "train_indices" not in data.files:
                raise KeyError(f"{fold_path} missing train_indices")
            train_indices = np.asarray(data["train_indices"], dtype=np.int64)
        train_labels = select_labels(train_indices, labels)
        shape = inspect_logits_shape(fold_path, "train_logits")
        if shape[0] != epochs:
            raise ValueError(
                f"Epoch mismatch in {fold_path}: cache={shape[0]}, expected={epochs}"
            )
        if shape[1] != len(train_indices):
            raise ValueError(f"Sample mismatch in {fold_path}")

        probability = np.empty((epochs, len(train_indices)), dtype=np.float32)
        loss = np.empty_like(probability)
        correct = np.empty((epochs, len(train_indices)), dtype=np.uint8)
        for epoch_index, logits in enumerate(iter_logits_epochs(fold_path, "train_logits")):
            p, l, c = true_probability_and_loss(logits, train_labels)
            probability[epoch_index] = p
            loss[epoch_index] = l
            correct[epoch_index] = c

        probability_sum[:, train_indices] += probability
        loss_sum[:, train_indices] += loss
        correct_sum[:, train_indices] += correct
        train_count[train_indices] += 1
        folds.append(
            FoldTrajectory(
                fold_id=fold_id,
                train_indices=train_indices,
                true_probability=probability,
                loss=loss,
            )
        )

    if np.any(train_count == 0):
        missing = np.flatnonzero(train_count == 0)
        raise ValueError(f"Samples absent from all train folds: {missing[:10]}")
    divisor = train_count.astype(np.float32)[None, :]
    return (
        folds,
        probability_sum / divisor,
        loss_sum / divisor,
        correct_sum.astype(np.float32) / divisor,
    )


def fold_raw_terms(
    fold: FoldTrajectory,
    first: int,
    second: int,
) -> dict[str, np.ndarray]:
    epochs = fold.true_probability.shape[0]
    if not (0 < first < second < epochs):
        raise ValueError(f"Invalid boundaries {(first, second, epochs)}")
    p = fold.true_probability.astype(np.float64, copy=False)
    loss = fold.loss.astype(np.float64, copy=False)
    early = slice(0, first)
    middle = slice(first, second)
    late = slice(second, epochs)

    early_mean = np.mean(p[early], axis=0)
    middle_mean = np.mean(p[middle], axis=0)
    boundary = np.mean(p[early] * (1.0 - p[early]), axis=0)
    support = np.sqrt(np.clip(early_mean * middle_mean, 0.0, 1.0))
    gain = support * (middle_mean - early_mean)
    stability = np.var(loss[late], axis=0) - np.var(loss[middle], axis=0)
    return {
        "Boundary": boundary,
        "Gain": gain,
        "Stability": stability,
    }


def reconstruct_a(
    folds: Sequence[FoldTrajectory],
    num_samples: int,
    first: int,
    second: int,
    static_class_z: Mapping[str, np.ndarray],
    saved_a: np.ndarray,
) -> DecompositionResult:
    contribution_sum = {
        name: np.zeros(num_samples, dtype=np.float64) for name in A_TERM_NAMES
    }
    count = np.zeros(num_samples, dtype=np.int64)

    for fold in folds:
        raw_terms = fold_raw_terms(fold, first, second)
        standardized = {
            name: safe_standardize(raw_terms[name]) for name in A_TERM_NAMES
        }
        combined = sum(standardized[name] for name in A_TERM_NAMES)
        combined_mean = float(np.mean(combined))
        combined_std = float(np.std(combined))
        denominator = combined_std + ZSCORE_EPS
        if not np.isfinite(combined_std) or combined_std < ZSCORE_EPS:
            fold_contributions = {
                name: np.zeros_like(combined) for name in A_TERM_NAMES
            }
        else:
            fold_contributions = {
                name: (standardized[name] - float(np.mean(standardized[name]))) / denominator
                for name in A_TERM_NAMES
            }
            reconstructed_fold = sum(fold_contributions.values())
            expected_fold = (combined - combined_mean) / denominator
            if np.max(np.abs(reconstructed_fold - expected_fold)) > 1e-5:
                raise RuntimeError("Internal fold decomposition is not additive")

        indices = fold.train_indices
        for name in A_TERM_NAMES:
            contribution_sum[name][indices] += fold_contributions[name]
        count[indices] += 1

    if np.any(count == 0):
        raise ValueError("Some samples were not covered by A train-fold aggregation")
    aggregated_terms = {
        name: contribution_sum[name] / count for name in A_TERM_NAMES
    }
    aggregated_a = sum(aggregated_terms.values())
    final_mean = float(np.mean(aggregated_a))
    final_std = float(np.std(aggregated_a))
    denominator = final_std + ZSCORE_EPS
    if not np.isfinite(final_std) or final_std < ZSCORE_EPS:
        contributions = {
            name: np.zeros(num_samples, dtype=np.float64) for name in A_TERM_NAMES
        }
        reconstructed_a = np.zeros(num_samples, dtype=np.float64)
    else:
        contributions = {
            name: (aggregated_terms[name] - float(np.mean(aggregated_terms[name]))) / denominator
            for name in A_TERM_NAMES
        }
        reconstructed_a = (aggregated_a - final_mean) / denominator

    component_values = {
        **contributions,
        "A(rebuilt)": reconstructed_a,
        "A(saved)": saved_a,
    }
    pearson_matrix = correlation_matrix(static_class_z, component_values, "pearson")
    spearman_matrix = correlation_matrix(static_class_z, component_values, "spearman")

    variance = float(np.var(reconstructed_a))
    variance_shares: dict[str, float] = {}
    for name in A_TERM_NAMES:
        covariance = float(
            np.mean(
                (contributions[name] - np.mean(contributions[name]))
                * (reconstructed_a - np.mean(reconstructed_a))
            )
        )
        variance_shares[name] = 0.0 if variance < EPS else covariance / variance

    return DecompositionResult(
        contributions=contributions,
        reconstructed_a=reconstructed_a,
        pearson_matrix=pearson_matrix,
        spearman_matrix=spearman_matrix,
        variance_shares=variance_shares,
        saved_pearson=pearson(reconstructed_a, saved_a),
        saved_spearman=spearman(reconstructed_a, saved_a),
        saved_max_abs_error=float(np.max(np.abs(reconstructed_a - saved_a))),
    )


def static_correlation_over_epochs(
    static_class_z: Mapping[str, np.ndarray],
    epoch_values: np.ndarray,
) -> np.ndarray:
    if epoch_values.ndim != 2:
        raise ValueError("epoch_values must have shape (epochs, samples)")
    static_ranks = {name: rankdata(values) for name, values in static_class_z.items()}
    result = np.empty((epoch_values.shape[0], len(STATIC_NAMES)), dtype=np.float64)
    for epoch in range(epoch_values.shape[0]):
        value_ranks = rankdata(epoch_values[epoch])
        for index, name in enumerate(STATIC_NAMES):
            result[epoch, index] = pearson(static_ranks[name], value_ranks)
    return result


def first_sustained_true(mask: np.ndarray, patience: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    epochs, samples = mask.shape
    result = np.full(samples, epochs + 1, dtype=np.int64)
    if patience > epochs:
        return result
    running = np.zeros(samples, dtype=np.int64)
    for epoch in range(epochs):
        running = np.where(mask[epoch], running + 1, 0)
        newly = (running >= patience) & (result == epochs + 1)
        result[newly] = epoch - patience + 2  # 1-indexed first epoch in the run
    return result


def build_epoch_diagnostics(
    static_class_z: Mapping[str, np.ndarray],
    average_probability: np.ndarray,
    average_loss: np.ndarray,
    average_correct: np.ndarray,
    learn_patience: int,
) -> EpochDiagnostics:
    epochs = average_probability.shape[0]
    if average_loss.shape != average_probability.shape or average_correct.shape != average_probability.shape:
        raise ValueError("Aggregated trajectory shapes do not match")

    boundary = average_probability * (1.0 - average_probability)
    one_step_gain = np.zeros_like(average_probability)
    loss_improvement = np.zeros_like(average_loss)
    if epochs > 1:
        previous = average_probability[:-1]
        current = average_probability[1:]
        support = np.sqrt(np.clip(previous * current, 0.0, 1.0))
        one_step_gain[1:] = support * (current - previous)
        loss_improvement[1:] = average_loss[:-1] - average_loss[1:]

    majority_correct = average_correct >= 0.5
    first_correct = first_sustained_true(majority_correct, learn_patience)
    first_p05 = first_sustained_true(average_probability >= 0.5, learn_patience)
    first_p09 = first_sustained_true(average_probability >= 0.9, learn_patience)

    return EpochDiagnostics(
        epochs=np.arange(1, epochs + 1, dtype=np.int64),
        true_probability_spearman=static_correlation_over_epochs(
            static_class_z, average_probability
        ),
        boundary_spearman=static_correlation_over_epochs(static_class_z, boundary),
        one_step_gain_spearman=static_correlation_over_epochs(
            static_class_z, one_step_gain
        ),
        loss_improvement_spearman=static_correlation_over_epochs(
            static_class_z, loss_improvement
        ),
        mean_true_probability=np.mean(average_probability, axis=1),
        mean_loss=np.mean(average_loss, axis=1),
        mean_accuracy=np.mean(average_correct, axis=1),
        final_true_probability=average_probability[-1].copy(),
        final_correct_rate=average_correct[-1].copy(),
        first_stable_correct_epoch=first_correct,
        first_probability_05_epoch=first_p05,
        first_probability_09_epoch=first_p09,
    )


def boundary_candidates(
    center: int,
    low: int,
    high: int,
    radius: int,
    step: int,
) -> list[int]:
    start = max(low, center - radius)
    stop = min(high, center + radius)
    values = list(range(start, stop + 1, step))
    if center not in values:
        values.append(center)
    return sorted(set(values))


def scan_boundaries(
    folds: Sequence[FoldTrajectory],
    num_samples: int,
    first: int,
    second: int,
    static_class_z: Mapping[str, np.ndarray],
    saved_a: np.ndarray,
    radius: int,
    step: int,
    minimum_phase: int,
) -> tuple[list[ScanPoint], list[ScanPoint]]:
    epochs = folds[0].true_probability.shape[0]
    first_values = boundary_candidates(
        first,
        minimum_phase,
        second - minimum_phase,
        radius,
        step,
    )
    second_values = boundary_candidates(
        second,
        first + minimum_phase,
        epochs - minimum_phase,
        radius,
        step,
    )

    first_scan: list[ScanPoint] = []
    for candidate in first_values:
        result = reconstruct_a(
            folds,
            num_samples,
            candidate,
            second,
            static_class_z,
            saved_a,
        )
        first_scan.append(
            ScanPoint(candidate, result.pearson_matrix, result.spearman_matrix)
        )

    second_scan: list[ScanPoint] = []
    for candidate in second_values:
        result = reconstruct_a(
            folds,
            num_samples,
            first,
            candidate,
            static_class_z,
            saved_a,
        )
        second_scan.append(
            ScanPoint(candidate, result.pearson_matrix, result.spearman_matrix)
        )
    return first_scan, second_scan


def print_matrix(
    title: str,
    matrix: np.ndarray,
    rows: Sequence[str],
    columns: Sequence[str],
) -> None:
    print(f"\n{title}")
    print(" " * 14 + "".join(f"{column:>14s}" for column in columns))
    for row_name, row in zip(rows, matrix):
        print(f"{row_name:<14s}" + "".join(f"{value:14.5f}" for value in row))


def print_decomposition_summary(
    target: Target,
    result: DecompositionResult,
) -> None:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    print_matrix(
        "[Pearson] static vs exact A contributions",
        result.pearson_matrix,
        STATIC_NAMES,
        columns,
    )
    print_matrix(
        "[Spearman] static vs exact A contributions",
        result.spearman_matrix,
        STATIC_NAMES,
        columns,
    )
    print("\n[A reconstruction check]")
    print(
        f"  boundaries=[{target.first_boundary}, {target.second_boundary}, {target.epochs}] | "
        f"Pearson={result.saved_pearson:.8f}, "
        f"Spearman={result.saved_spearman:.8f}, "
        f"max_abs_error={result.saved_max_abs_error:.8g}"
    )
    if result.saved_pearson < 0.999:
        print(
            "  WARNING: reconstructed A does not match saved A. The dynamic cache may "
            "have been generated with different phase boundaries or different A code."
        )

    print("\n[Exact contribution to variance of reconstructed A]")
    for name in A_TERM_NAMES:
        values = result.contributions[name]
        print(
            f"  {name:<10s}: covariance_share={result.variance_shares[name]:+.5f}, "
            f"std={np.std(values):.5f}, corr_with_A={pearson(values, result.reconstructed_a):+.5f}"
        )
    print(
        "  covariance shares sum to "
        f"{sum(result.variance_shares.values()):.5f} (approximately 1 by additivity)."
    )


def print_epoch_localization(
    target: Target,
    diagnostics: EpochDiagnostics,
    top_k: int,
) -> None:
    sa = STATIC_NAMES.index("SA")
    div = STATIC_NAMES.index("Div")
    dds = STATIC_NAMES.index("DDS")
    gain = diagnostics.one_step_gain_spearman
    preference = gain[:, sa] - np.maximum(gain[:, div], gain[:, dds])
    preference[0] = -np.inf
    order = np.argsort(preference)[::-1]

    print("\n[Most SA-favouring one-step absorption gains]")
    print("epoch      SA       Div       DDS    SA-gap   mean-p    mean-loss    accuracy")
    printed = 0
    for index in order:
        if not np.isfinite(preference[index]):
            continue
        print(
            f"{index + 1:5d}  "
            f"{gain[index, sa]:+8.5f} {gain[index, div]:+9.5f} "
            f"{gain[index, dds]:+9.5f} {preference[index]:+9.5f} "
            f"{diagnostics.mean_true_probability[index]:8.5f} "
            f"{diagnostics.mean_loss[index]:10.5f} "
            f"{diagnostics.mean_accuracy[index]:9.5f}"
        )
        printed += 1
        if printed >= top_k:
            break

    print("\n[Milestone-local diagnostics]")
    for boundary in (target.first_boundary, target.second_boundary):
        index = boundary  # transition from epoch boundary to boundary+1, zero-based row=boundary
        if index >= target.epochs:
            continue
        before = max(1, index - 2)
        after = min(target.epochs - 1, index + 2)
        print(
            f"  milestone={boundary}: step(epoch {boundary}->{boundary + 1}) "
            f"gain Spearman SA={gain[index, sa]:+.5f}, "
            f"Div={gain[index, div]:+.5f}, DDS={gain[index, dds]:+.5f}; "
            f"mean-p {diagnostics.mean_true_probability[index - 1]:.5f}"
            f"->{diagnostics.mean_true_probability[index]:.5f}; "
            f"accuracy {diagnostics.mean_accuracy[index - 1]:.5f}"
            f"->{diagnostics.mean_accuracy[index]:.5f}"
        )
        before_sa = float(np.mean(gain[before:index + 1, sa]))
        after_sa = float(np.mean(gain[index + 1:after + 1, sa])) if after > index else 0.0
        before_div = float(np.mean(gain[before:index + 1, div]))
        after_div = float(np.mean(gain[index + 1:after + 1, div])) if after > index else 0.0
        print(
            f"    nearby mean one-step correlation: "
            f"SA {before_sa:+.5f}->{after_sa:+.5f}, "
            f"Div {before_div:+.5f}->{after_div:+.5f}"
        )


def epoch_with_never_as_end(values: np.ndarray, epochs: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > epochs, epochs + 1, values)


def print_learning_time_diagnostics(
    target: Target,
    static_class_z: Mapping[str, np.ndarray],
    diagnostics: EpochDiagnostics,
    quantile: float,
) -> None:
    first = epoch_with_never_as_end(
        diagnostics.first_stable_correct_epoch, target.epochs
    )
    print("\n[Static score vs first stably-correct epoch]")
    for name in STATIC_NAMES:
        print(
            f"  {name:<4s}: Pearson={pearson(static_class_z[name], first):+.5f}, "
            f"Spearman={spearman(static_class_z[name], first):+.5f}"
        )

    print(f"\n[Low/high {quantile:.0%} groups: stable learning time]")
    print(
        "metric group      n   median-first  never%   final-mean-p  final-accuracy"
    )
    final_p = diagnostics.final_true_probability
    final_correct_rate = diagnostics.final_correct_rate
    for name in STATIC_NAMES:
        values = static_class_z[name]
        low_threshold = float(np.quantile(values, quantile))
        high_threshold = float(np.quantile(values, 1.0 - quantile))
        for group_name, mask in (
            ("low", values <= low_threshold),
            ("high", values >= high_threshold),
        ):
            group_first = diagnostics.first_stable_correct_epoch[mask]
            learned = group_first <= target.epochs
            median = (
                float(np.median(group_first[learned])) if np.any(learned) else float("nan")
            )
            never = float(np.mean(~learned))
            print(
                f"{name:<6s} {group_name:<5s} {int(mask.sum()):6d} "
                f"{median:12.3f} {100.0 * never:7.3f}% "
                f"{float(np.mean(final_p[mask])):13.5f} "
                f"{float(np.mean(final_correct_rate[mask])):14.5f}"
            )


def scan_table(
    title: str,
    points: Sequence[ScanPoint],
    actual_boundary: int,
) -> None:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    a_index = columns.index("A(rebuilt)")
    gain_index = columns.index("Gain")
    stability_index = columns.index("Stability")
    print(f"\n{title}")
    print("boundary actual  A-SA     A-Div    A-DDS   Gain-SA  Stability-SA")
    for point in points:
        matrix = point.spearman_matrix
        print(
            f"{point.boundary:8d} {'*' if point.boundary == actual_boundary else ' ':>6s} "
            f"{matrix[0, a_index]:+8.5f} {matrix[1, a_index]:+9.5f} "
            f"{matrix[2, a_index]:+9.5f} {matrix[0, gain_index]:+9.5f} "
            f"{matrix[0, stability_index]:+13.5f}"
        )


def annotate_heatmap(
    axis,
    matrix: np.ndarray,
    rows: Sequence[str],
    columns: Sequence[str],
    title: str,
):
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(columns)), labels=columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(rows)), labels=rows)
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    return image


def save_component_figure(
    path: Path,
    target: Target,
    result: DecompositionResult,
) -> None:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    image = annotate_heatmap(
        axes[0], result.pearson_matrix, STATIC_NAMES, columns, "Pearson"
    )
    annotate_heatmap(
        axes[1], result.spearman_matrix, STATIC_NAMES, columns, "Spearman"
    )
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Correlation")
    figure.suptitle(
        f"A decomposition - Exp.1 {target.dataset}, seed {target.seed}\n"
        f"phases: 1-{target.first_boundary}, "
        f"{target.first_boundary + 1}-{target.second_boundary}, "
        f"{target.second_boundary + 1}-{target.epochs}",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.07, right=0.93, bottom=0.18, top=0.82, wspace=0.28)
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def save_epoch_localization_figure(
    path: Path,
    target: Target,
    diagnostics: EpochDiagnostics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    panels = (
        (diagnostics.true_probability_spearman, "True-class probability"),
        (diagnostics.boundary_spearman, "Boundary p(1-p)"),
        (diagnostics.one_step_gain_spearman, "One-step supported gain"),
        (diagnostics.loss_improvement_spearman, "One-step loss improvement"),
    )
    for axis, (matrix, title) in zip(axes.ravel(), panels):
        for index, name in enumerate(STATIC_NAMES):
            axis.plot(diagnostics.epochs, matrix[:, index], label=name, linewidth=1.7)
        axis.axhline(0.0, linewidth=1)
        for boundary in (target.first_boundary, target.second_boundary):
            axis.axvline(boundary, linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel("Spearman correlation")
        axis.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].legend()
    figure.suptitle(
        f"Epoch-level localization - Exp.1 {target.dataset}, seed {target.seed}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def scan_series(
    points: Sequence[ScanPoint],
    component_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    component_index = columns.index(component_name)
    x = np.asarray([point.boundary for point in points], dtype=np.int64)
    y = np.asarray(
        [point.spearman_matrix[:, component_index] for point in points],
        dtype=np.float64,
    )
    return x, y


def save_boundary_scan_figure(
    path: Path,
    target: Target,
    first_scan: Sequence[ScanPoint],
    second_scan: Sequence[ScanPoint],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    configurations = (
        (axes[0, 0], first_scan, "A(rebuilt)", "Scan first boundary: A"),
        (axes[0, 1], first_scan, "Gain", "Scan first boundary: Gain"),
        (axes[1, 0], second_scan, "A(rebuilt)", "Scan second boundary: A"),
        (axes[1, 1], second_scan, "Stability", "Scan second boundary: Stability"),
    )
    for axis, points, component, title in configurations:
        x, matrix = scan_series(points, component)
        for index, name in enumerate(STATIC_NAMES):
            axis.plot(x, matrix[:, index], marker="o", label=name)
        actual = target.first_boundary if points is first_scan else target.second_boundary
        axis.axvline(actual, linestyle="--", linewidth=1.2)
        axis.axhline(0.0, linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Candidate boundary")
        axis.set_ylabel("Spearman correlation")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle(
        f"Fixed-trajectory phase sensitivity - Exp.1 {target.dataset}, seed {target.seed}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def save_learning_figure(
    path: Path,
    target: Target,
    static_class_z: Mapping[str, np.ndarray],
    diagnostics: EpochDiagnostics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    learning_epoch = epoch_with_never_as_end(
        diagnostics.first_stable_correct_epoch, target.epochs
    )
    bins = np.linspace(-3.0, 3.0, 13)
    for axis, name in zip(axes, STATIC_NAMES):
        values = static_class_z[name]
        centers: list[float] = []
        medians: list[float] = []
        never_rates: list[float] = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (values >= left) & (values < right)
            if int(mask.sum()) < 5:
                continue
            centers.append(0.5 * (left + right))
            medians.append(float(np.median(learning_epoch[mask])))
            never_rates.append(
                float(np.mean(diagnostics.first_stable_correct_epoch[mask] > target.epochs))
            )
        axis.plot(centers, medians, marker="o", label="Median first stable epoch")
        secondary = axis.twinx()
        secondary.plot(centers, never_rates, marker="s", linestyle="--", label="Never learned")
        axis.set_title(name)
        axis.set_xlabel("Class-standardized score")
        axis.grid(True, alpha=0.3)
        secondary.set_ylim(-0.02, 1.02)
        if name == "DDS":
            secondary.set_ylabel("Never-learned fraction")
    axes[0].set_ylabel("First stable-correct epoch (end+1 = never)")
    figure.suptitle(
        f"Learning time vs static scores - Exp.1 {target.dataset}, seed {target.seed}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def process_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    epochs, paths = resolve_cache_paths(args, dataset)
    first, second = read_phase_boundaries(args, dataset, epochs, paths.proxy_dir)
    target = Target(
        dataset=dataset,
        seed=int(args.seed),
        proxy_model=str(args.proxy_model),
        epochs=epochs,
        first_boundary=first,
        second_boundary=second,
    )

    print("\n" + "=" * 88)
    print(
        f"[Target] exp=1, dataset={dataset}, seed={target.seed}, "
        f"proxy_model={target.proxy_model}, epochs={epochs}, "
        f"phases=[1-{first}], [{first + 1}-{second}], [{second + 1}-{epochs}]"
    )
    print(f"[Cache] proxy={paths.proxy_dir}")
    print(f"[Cache] A={paths.dynamic_dir / 'A.npz'}")
    print(f"[Cache] static={paths.static_path}")

    _, labels, saved_a, _, static_class_z = load_a_and_static(paths)
    folds, average_probability, average_loss, average_correct = load_fold_trajectories(
        paths.proxy_dir,
        labels,
        epochs,
    )

    decomposition = reconstruct_a(
        folds,
        len(labels),
        first,
        second,
        static_class_z,
        saved_a,
    )
    print_decomposition_summary(target, decomposition)

    epoch_diagnostics = build_epoch_diagnostics(
        static_class_z,
        average_probability,
        average_loss,
        average_correct,
        args.learn_patience,
    )
    print_epoch_localization(target, epoch_diagnostics, args.top_k_epochs)
    print_learning_time_diagnostics(
        target,
        static_class_z,
        epoch_diagnostics,
        args.group_quantile,
    )

    first_scan, second_scan = scan_boundaries(
        folds,
        len(labels),
        first,
        second,
        static_class_z,
        saved_a,
        args.scan_radius,
        args.scan_step,
        args.minimum_phase_length,
    )
    scan_table(
        "[Fixed-trajectory scan] first boundary (second fixed)",
        first_scan,
        first,
    )
    scan_table(
        "[Fixed-trajectory scan] second boundary (first fixed)",
        second_scan,
        second,
    )

    output_dir = resolve_project_path(args.output_dir)
    stem = f"exp1_{dataset.replace('-', '_')}_seed{target.seed}_A"
    component_path = output_dir / f"{stem}_component_correlations.png"
    epoch_path = output_dir / f"{stem}_epoch_localization.png"
    scan_path = output_dir / f"{stem}_boundary_scan.png"
    learning_path = output_dir / f"{stem}_learning_time.png"
    save_component_figure(component_path, target, decomposition)
    save_epoch_localization_figure(epoch_path, target, epoch_diagnostics)
    save_boundary_scan_figure(scan_path, target, first_scan, second_scan)
    save_learning_figure(learning_path, target, static_class_z, epoch_diagnostics)

    output_paths = [component_path, epoch_path, scan_path, learning_path]
    print("\n[Saved figures]")
    for path in output_paths:
        print(f"  {path}")

    return DatasetResult(
        target=target,
        static_class_z=static_class_z,
        saved_a=saved_a,
        decomposition=decomposition,
        epoch_diagnostics=epoch_diagnostics,
        first_scan=first_scan,
        second_scan=second_scan,
        output_paths=output_paths,
    )


def save_comparison_figure(
    path: Path,
    cifar: DatasetResult,
    tiny: DatasetResult,
) -> None:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    image = annotate_heatmap(
        axes[0],
        cifar.decomposition.spearman_matrix,
        STATIC_NAMES,
        columns,
        f"CIFAR-100 ({cifar.target.epochs} epochs)",
    )
    annotate_heatmap(
        axes[1],
        tiny.decomposition.spearman_matrix,
        STATIC_NAMES,
        columns,
        f"Tiny-ImageNet ({tiny.target.epochs} epochs)",
    )
    delta = tiny.decomposition.spearman_matrix - cifar.decomposition.spearman_matrix
    annotate_heatmap(
        axes[2],
        delta,
        STATIC_NAMES,
        columns,
        "Tiny minus CIFAR",
    )
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Correlation")
    figure.suptitle(
        f"A decomposition comparison - unseen Exp.1, seed {cifar.target.seed}",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.19, top=0.83, wspace=0.32)
    figure.savefig(path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def print_cross_dataset_comparison(cifar: DatasetResult, tiny: DatasetResult) -> None:
    columns = (*A_TERM_NAMES, "A(rebuilt)", "A(saved)")
    delta = tiny.decomposition.spearman_matrix - cifar.decomposition.spearman_matrix
    print("\n" + "=" * 88)
    print("[Cross-dataset comparison: Tiny-ImageNet minus CIFAR-100 Spearman]")
    print_matrix("[Delta]", delta, STATIC_NAMES, columns)

    candidates: list[tuple[float, str, str, float, float]] = []
    for row, static_name in enumerate(STATIC_NAMES):
        for column, component_name in enumerate(columns[:-1]):
            change = float(delta[row, column])
            candidates.append(
                (
                    abs(change),
                    static_name,
                    component_name,
                    float(cifar.decomposition.spearman_matrix[row, column]),
                    float(tiny.decomposition.spearman_matrix[row, column]),
                )
            )
    candidates.sort(reverse=True)
    print("\n[Largest normal-to-abnormal changes]")
    for _, static_name, component_name, cifar_value, tiny_value in candidates[:8]:
        print(
            f"  {static_name:>3s} vs {component_name:<11s}: "
            f"CIFAR={cifar_value:+.5f}, Tiny={tiny_value:+.5f}, "
            f"delta={tiny_value - cifar_value:+.5f}"
        )

    a_column = columns.index("A(rebuilt)")
    tiny_a = tiny.decomposition.spearman_matrix[:, a_column]
    cifar_a = cifar.decomposition.spearman_matrix[:, a_column]
    print("\n[Compact interpretation aid]")
    print(
        "  A correlation shift: "
        + ", ".join(
            f"{name} {cifar_value:+.5f}->{tiny_value:+.5f}"
            for name, cifar_value, tiny_value in zip(STATIC_NAMES, cifar_a, tiny_a)
        )
    )
    term_delta = delta[:, : len(A_TERM_NAMES)]
    row, column = np.unravel_index(np.argmax(np.abs(term_delta)), term_delta.shape)
    print(
        f"  Largest internal-term shift: {STATIC_NAMES[row]} vs "
        f"{A_TERM_NAMES[column]}, delta={term_delta[row, column]:+.5f}."
    )


def main() -> int:
    args = parse_args()
    datasets = (
        list(SUPPORTED_DATASETS) if args.dataset == "both" else [args.dataset]
    )
    results: dict[str, DatasetResult] = {}
    for dataset in datasets:
        results[dataset] = process_dataset(args, dataset)

    if len(results) == 2:
        cifar = results["cifar100"]
        tiny = results["tiny-imagenet"]
        print_cross_dataset_comparison(cifar, tiny)
        comparison_path = (
            resolve_project_path(args.output_dir)
            / f"exp1_seed{args.seed}_A_component_comparison.png"
        )
        save_comparison_figure(comparison_path, cifar, tiny)
        print(f"\n[Saved comparison figure]\n  {comparison_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[Error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)