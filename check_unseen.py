#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose proxy dynamics and static/dynamic score alignment for one unseen experiment.

Run from the project root, for example:

    python check_unseen.py --exp 1 --dataset tiny-imagenet
    python check_unseen.py --exp 3 --dataset cifar100 --seed 22 --epochs 100

The script reads one experiment/dataset/seed combination from:

    unseen_exp/proxy_logs/<exp>/<dataset>/<proxy_model>/<seed>/<epochs>/
    unseen_exp/dynamic_cache/<exp>/<dataset>/<proxy_model>/<seed>/<epochs>/
    unseen_exp/static_scores/<exp>/selection/<dataset>/<seed>/static_scores.npz

It prints diagnostics to the terminal and saves PNG figures only. It does not
modify any cache file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_NAMES = ("SA", "Div", "DDS")
STATIC_KEYS = ("sa", "div", "dds")
DYNAMIC_NAMES = ("A", "C", "T")
SUPPORTED_DATASETS = ("cifar100", "tiny-imagenet")
EPS = 1e-12


@dataclass(frozen=True)
class Target:
    exp: int
    dataset: str
    seed: int
    proxy_model: str
    epochs: int


@dataclass(frozen=True)
class CachePaths:
    proxy_dir: Path
    dynamic_dir: Path
    static_path: Path
    weights_path: Path


@dataclass(frozen=True)
class NpyStreamInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool


@dataclass
class ProxyMetrics:
    epochs: np.ndarray
    train_loss: np.ndarray
    val_loss: np.ndarray
    val_accuracy: np.ndarray
    num_folds: int


@dataclass
class DiagnosticData:
    known_indices: np.ndarray
    labels: np.ndarray
    static_raw: dict[str, np.ndarray]
    static_class_z: dict[str, np.ndarray]
    dynamic: dict[str, np.ndarray]
    pseudo_target: np.ndarray
    is_corrupted: np.ndarray | None


@dataclass(frozen=True)
class FitResult:
    weights: np.ndarray
    r2: float
    prediction_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose one unseen-sample experiment by plotting proxy dynamics "
            "and analysing SA/Div/DDS against A/C/T."
        )
    )
    parser.add_argument("--exp", type=int, required=True, choices=(1, 3))
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Experiment 3 supports only cifar100.",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Exact numeric cache directory. If omitted, use the largest epoch "
            "directory present in both proxy_logs and dynamic_cache."
        ),
    )
    parser.add_argument(
        "--unseen-root",
        type=Path,
        default=Path("unseen_exp"),
        help="Unseen-experiment root relative to the project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("check_unseen_output"),
        help="Directory in which PNG figures are saved.",
    )
    parser.add_argument(
        "--normalization",
        choices=("relative", "none"),
        default="relative",
        help="Normalization used only for the proxy-dynamics figure.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window used only for the proxy-dynamics figure.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of strongest cross-component correlations printed separately.",
    )
    args = parser.parse_args()
    if args.exp == 3 and args.dataset != "cifar100":
        parser.error("Experiment 3 supports only --dataset cifar100")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def numeric_cache_dirs(root: Path, *, require_folds: bool = False) -> dict[int, Path]:
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


def resolve_target(args: argparse.Namespace) -> tuple[Target, CachePaths]:
    unseen_root = resolve_project_path(args.unseen_root)
    proxy_seed_root = (
        unseen_root
        / "proxy_logs"
        / str(args.exp)
        / args.dataset
        / args.proxy_model
        / str(args.seed)
    )
    dynamic_seed_root = (
        unseen_root
        / "dynamic_cache"
        / str(args.exp)
        / args.dataset
        / args.proxy_model
        / str(args.seed)
    )

    if args.epochs is None:
        proxy_epochs = numeric_cache_dirs(proxy_seed_root, require_folds=True)
        dynamic_epochs = numeric_cache_dirs(dynamic_seed_root)
        common = sorted(set(proxy_epochs) & set(dynamic_epochs))
        if not common:
            raise FileNotFoundError(
                "No common numeric epoch directory exists in both proxy and dynamic "
                f"caches. proxy={proxy_seed_root}, dynamic={dynamic_seed_root}"
            )
        epochs = common[-1]
    else:
        epochs = int(args.epochs)

    target = Target(
        exp=int(args.exp),
        dataset=str(args.dataset),
        seed=int(args.seed),
        proxy_model=str(args.proxy_model),
        epochs=epochs,
    )
    paths = CachePaths(
        proxy_dir=proxy_seed_root / str(epochs),
        dynamic_dir=dynamic_seed_root / str(epochs),
        static_path=(
            unseen_root
            / "static_scores"
            / str(args.exp)
            / "selection"
            / args.dataset
            / str(args.seed)
            / "static_scores.npz"
        ),
        weights_path=unseen_root / "weights" / str(args.exp) / "scoring_weights.json",
    )

    if not paths.proxy_dir.is_dir():
        raise FileNotFoundError(f"Proxy cache not found: {paths.proxy_dir}")
    if not paths.dynamic_dir.is_dir():
        raise FileNotFoundError(f"Dynamic cache not found: {paths.dynamic_dir}")
    if not paths.static_path.is_file():
        raise FileNotFoundError(f"Static-score cache not found: {paths.static_path}")
    return target, paths


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
            raise ValueError("NPY 3.0 headers are unsupported by this NumPy version")
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
    while remaining > 0:
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
        raise ValueError(f"{array_name} must be 3-D, got {info.shape}")
    return int(info.shape[0]), int(info.shape[1]), int(info.shape[2])


def iter_logits_epochs(npz_path: Path, array_name: str) -> Iterable[np.ndarray]:
    member = f"{array_name}.npy"
    with zipfile.ZipFile(npz_path, mode="r") as archive:
        if member not in archive.namelist():
            raise KeyError(f"{member} not found in {npz_path}")
        with archive.open(member, mode="r") as stream:
            info = _read_npy_header(stream)
            if len(info.shape) != 3:
                raise ValueError(f"{array_name} must be 3-D, got {info.shape}")
            if info.fortran_order:
                raise ValueError(
                    f"Fortran-order logits are unsupported: {npz_path}:{array_name}"
                )
            num_epochs, num_samples, num_classes = info.shape
            values_per_epoch = num_samples * num_classes
            bytes_per_epoch = values_per_epoch * info.dtype.itemsize
            for _ in range(num_epochs):
                raw = _read_exact(stream, bytes_per_epoch)
                values = np.frombuffer(raw, dtype=info.dtype, count=values_per_epoch)
                yield values.reshape(num_samples, num_classes)


def cross_entropy_sum(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError(
            f"Labels/logits mismatch: labels={labels.shape}, logits={values.shape}"
        )
    if np.any(labels < 0) or np.any(labels >= values.shape[1]):
        raise ValueError("Label out of range for logits")
    row_max = np.max(values, axis=1)
    shifted = values - row_max[:, None]
    log_sum_exp = row_max + np.log(np.exp(shifted).sum(axis=1))
    result = log_sum_exp - values[np.arange(values.shape[0]), labels]
    if not np.isfinite(result).all():
        raise ValueError("Cross entropy contains NaN or infinity")
    return float(result.sum())


def load_known_indices_from_dynamic(dynamic_dir: Path) -> np.ndarray:
    for name in (*DYNAMIC_NAMES, "pseudo_labels"):
        path = dynamic_dir / f"{name}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if "known_indices" in data.files:
                known = np.asarray(data["known_indices"], dtype=np.int64)
                if known.ndim == 1 and np.unique(known).size == known.size:
                    return known
    raise KeyError(f"No valid known_indices found under {dynamic_dir}")


def load_static_cache(
    static_path: Path,
    dynamic_known: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(static_path, allow_pickle=False) as data:
        required = set(STATIC_KEYS) | {"labels"}
        if not required.issubset(data.files):
            missing = sorted(required - set(data.files))
            raise KeyError(f"{static_path} missing arrays: {missing}")

        labels = np.asarray(data["labels"], dtype=np.int64)
        scores = {
            name: np.asarray(data[key], dtype=np.float64)
            for name, key in zip(STATIC_NAMES, STATIC_KEYS)
        }
        lengths = {len(values) for values in scores.values()} | {len(labels)}
        if len(lengths) != 1:
            raise ValueError(f"Static arrays have inconsistent lengths: {lengths}")
        n_static = len(labels)

        if "sample_indices" in data.files:
            sample_indices = np.asarray(data["sample_indices"], dtype=np.int64)
        else:
            sample_indices = np.arange(n_static, dtype=np.int64)
        if sample_indices.shape != (n_static,):
            raise ValueError("static sample_indices shape mismatch")
        if np.unique(sample_indices).size != sample_indices.size:
            raise ValueError("static sample_indices contains duplicates")

        order = np.argsort(sample_indices)
        sorted_indices = sample_indices[order]
        positions = np.searchsorted(sorted_indices, dynamic_known)
        if (
            np.any(positions >= len(sorted_indices))
            or not np.array_equal(sorted_indices[positions], dynamic_known)
        ):
            raise ValueError(
                "Static cache does not cover every dynamic known index. "
                f"static={static_path}"
            )
        static_positions = order[positions]
        aligned_scores = {name: values[static_positions] for name, values in scores.items()}
        aligned_labels = labels[static_positions]

        is_corrupted: np.ndarray | None = None
        if "is_corrupted" in data.files:
            full_corrupted = np.asarray(data["is_corrupted"], dtype=bool)
            if full_corrupted.shape == (n_static,):
                is_corrupted = full_corrupted[static_positions]

    for name, values in aligned_scores.items():
        validate_vector(name, values, len(dynamic_known))
    return aligned_scores, aligned_labels, static_positions, is_corrupted


def load_dynamic_components(
    dynamic_dir: Path,
    expected_known: np.ndarray,
    expected_labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray | None]:
    components: dict[str, np.ndarray] = {}
    dynamic_corrupted: np.ndarray | None = None
    for name in DYNAMIC_NAMES:
        path = dynamic_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Dynamic component cache not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"final_normalized", "known_indices", "labels"}
            if not required.issubset(data.files):
                missing = sorted(required - set(data.files))
                raise KeyError(f"{path} missing arrays: {missing}")
            known = np.asarray(data["known_indices"], dtype=np.int64)
            labels = np.asarray(data["labels"], dtype=np.int64)
            values = np.asarray(data["final_normalized"], dtype=np.float64)
            if not np.array_equal(known, expected_known):
                raise ValueError(f"known_indices mismatch in {path}")
            if not np.array_equal(labels, expected_labels):
                raise ValueError(f"labels mismatch in {path}")
            validate_vector(name, values, len(expected_known))
            components[name] = values
            if "is_corrupted" in data.files:
                candidate = np.asarray(data["is_corrupted"], dtype=bool)
                if candidate.shape == (len(expected_known),):
                    dynamic_corrupted = candidate
                elif candidate.ndim == 1 and len(candidate) > int(expected_known.max()):
                    dynamic_corrupted = candidate[expected_known]

    pseudo_path = dynamic_dir / "pseudo_labels.npz"
    pseudo: np.ndarray | None = None
    if pseudo_path.is_file():
        with np.load(pseudo_path, allow_pickle=False) as data:
            if "dynamic_target" in data.files:
                candidate = np.asarray(data["dynamic_target"], dtype=np.float64)
                if candidate.shape == (len(expected_known),) and np.isfinite(candidate).all():
                    pseudo = candidate
    if pseudo is None:
        pseudo = standard_zscore(
            sum(components[name] for name in DYNAMIC_NAMES) / len(DYNAMIC_NAMES)
        )
    return components, pseudo, dynamic_corrupted


def validate_vector(name: str, values: np.ndarray, expected_length: int) -> None:
    if values.shape != (expected_length,):
        raise ValueError(
            f"{name} shape mismatch: got {values.shape}, expected {(expected_length,)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")


def standard_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if not np.isfinite(std) or std < EPS:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def classwise_zscore(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros_like(values)
    for cls in np.unique(labels):
        mask = labels == cls
        output[mask] = standard_zscore(values[mask])
    return output


def load_diagnostic_data(paths: CachePaths) -> DiagnosticData:
    known = load_known_indices_from_dynamic(paths.dynamic_dir)
    static_raw, labels, _, static_corrupted = load_static_cache(paths.static_path, known)
    dynamic, pseudo, dynamic_corrupted = load_dynamic_components(
        paths.dynamic_dir,
        known,
        labels,
    )
    static_class_z = {
        name: classwise_zscore(static_raw[name], labels) for name in STATIC_NAMES
    }
    is_corrupted = dynamic_corrupted if dynamic_corrupted is not None else static_corrupted
    return DiagnosticData(
        known_indices=known,
        labels=labels,
        static_raw=static_raw,
        static_class_z=static_class_z,
        dynamic=dynamic,
        pseudo_target=pseudo,
        is_corrupted=is_corrupted,
    )


def select_fold_labels(
    indices: np.ndarray,
    known_labels: np.ndarray,
    full_static_labels: np.ndarray | None = None,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("Fold indices must be non-negative one-dimensional integers")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    max_index = int(indices.max())
    if max_index < len(known_labels):
        return known_labels[indices]
    if full_static_labels is not None and max_index < len(full_static_labels):
        return full_static_labels[indices]
    raise ValueError(
        f"Fold index {max_index} exceeds known-label length {len(known_labels)}"
    )


def compute_proxy_metrics(proxy_dir: Path, known_labels: np.ndarray) -> ProxyMetrics:
    fold_paths = sorted(proxy_dir.glob("fold_*.npz"), key=fold_sort_key)
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {proxy_dir}")

    train_total: np.ndarray | None = None
    val_total: np.ndarray | None = None
    correct_total: np.ndarray | None = None
    train_count = 0
    val_count = 0

    for fold_path in fold_paths:
        print(f"[Proxy] reading {fold_path}")
        with np.load(fold_path, allow_pickle=False) as data:
            required = {"train_indices", "val_indices"}
            if not required.issubset(data.files):
                missing = sorted(required - set(data.files))
                raise KeyError(f"{fold_path} missing arrays: {missing}")
            train_indices = np.asarray(data["train_indices"], dtype=np.int64)
            val_indices = np.asarray(data["val_indices"], dtype=np.int64)

        train_labels = select_fold_labels(train_indices, known_labels)
        val_labels = select_fold_labels(val_indices, known_labels)
        train_shape = inspect_logits_shape(fold_path, "train_logits")
        val_shape = inspect_logits_shape(fold_path, "val_logits")
        if train_shape[0] != val_shape[0]:
            raise ValueError(f"Epoch count mismatch in {fold_path}")
        if train_shape[1] != len(train_labels) or val_shape[1] != len(val_labels):
            raise ValueError(f"Sample count mismatch in {fold_path}")
        if train_shape[2] != val_shape[2]:
            raise ValueError(f"Class count mismatch in {fold_path}")

        fold_train_loss = np.empty(train_shape[0], dtype=np.float64)
        for epoch, logits in enumerate(iter_logits_epochs(fold_path, "train_logits")):
            fold_train_loss[epoch] = cross_entropy_sum(logits, train_labels)

        fold_val_loss = np.empty(val_shape[0], dtype=np.float64)
        fold_correct = np.empty(val_shape[0], dtype=np.float64)
        for epoch, logits in enumerate(iter_logits_epochs(fold_path, "val_logits")):
            fold_val_loss[epoch] = cross_entropy_sum(logits, val_labels)
            fold_correct[epoch] = float(
                np.count_nonzero(np.argmax(logits, axis=1) == val_labels)
            )

        if train_total is None:
            train_total = np.zeros_like(fold_train_loss)
            val_total = np.zeros_like(fold_val_loss)
            correct_total = np.zeros_like(fold_correct)
        elif len(fold_train_loss) != len(train_total):
            raise ValueError(f"Fold epoch count mismatch in {proxy_dir}")

        train_total += fold_train_loss
        val_total += fold_val_loss
        correct_total += fold_correct
        train_count += len(train_labels)
        val_count += len(val_labels)

    assert train_total is not None
    assert val_total is not None
    assert correct_total is not None
    return ProxyMetrics(
        epochs=np.arange(1, len(train_total) + 1, dtype=np.int64),
        train_loss=train_total / train_count,
        val_loss=val_total / val_count,
        val_accuracy=correct_total / val_count,
        num_folds=len(fold_paths),
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Correlation vectors must have the same shape")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator < EPS:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def correlation_matrix(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    method: str,
) -> np.ndarray:
    function = pearson if method == "pearson" else spearman
    return np.asarray(
        [[function(left[lname], right[rname]) for rname in right] for lname in left],
        dtype=np.float64,
    )


def pairwise_matrix(values: Mapping[str, np.ndarray], method: str) -> np.ndarray:
    return correlation_matrix(values, values, method)


def positive_fit(features: np.ndarray, target: np.ndarray) -> FitResult:
    """Fit a centered non-negative linear model for diagnostic purposes.

    SciPy NNLS is used when available. A projected-gradient fallback keeps the
    script usable in minimal environments. This is not a replacement for stage
    5's softplus-ratio regression; it is an interpretable diagnostic showing
    which static component can positively explain each dynamic component.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y)

    try:
        from scipy.optimize import nnls

        coefficients, _ = nnls(x, y)
    except Exception:
        gram = x.T @ x
        lipschitz = float(np.linalg.norm(gram, ord=2))
        step = 1.0 / max(lipschitz, EPS)
        coefficients = np.zeros(x.shape[1], dtype=np.float64)
        for _ in range(20_000):
            gradient = x.T @ (x @ coefficients - y)
            updated = np.maximum(0.0, coefficients - step * gradient)
            if np.max(np.abs(updated - coefficients)) < 1e-10:
                coefficients = updated
                break
            coefficients = updated

    prediction = x @ coefficients
    residual = float(np.sum((y - prediction) ** 2))
    total = float(np.sum(y**2))
    r2 = 0.0 if total < EPS else 1.0 - residual / total
    coefficient_sum = float(np.sum(coefficients))
    weights = (
        coefficients / coefficient_sum
        if coefficient_sum > EPS
        else np.zeros_like(coefficients)
    )
    return FitResult(
        weights=weights,
        r2=float(r2),
        prediction_std=float(np.std(prediction)),
    )


def vector_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    skewness = 0.0 if std < EPS else float(np.mean((centered / std) ** 3))
    kurtosis = 0.0 if std < EPS else float(np.mean((centered / std) ** 4) - 3.0)
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "skew": skewness,
        "excess_kurtosis": kurtosis,
        "outlier_fraction": float(np.mean(np.abs(standard_zscore(values)) > 3.0)),
        "unique_ratio": float(np.unique(values).size / max(1, values.size)),
    }


def load_saved_weights(path: Path, target: Target) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload[target.dataset][str(target.seed)]
        return {name: float(entry[key]) for name, key in zip(STATIC_NAMES, STATIC_KEYS)}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def print_matrix(title: str, matrix: np.ndarray, rows: tuple[str, ...], columns: tuple[str, ...]) -> None:
    width = 11
    print(f"\n{title}")
    print("".ljust(width) + "".join(name.rjust(width) for name in columns))
    for row_name, row in zip(rows, matrix):
        print(row_name.ljust(width) + "".join(f"{value:>{width}.5f}" for value in row))


def print_component_summaries(data: DiagnosticData) -> None:
    print("\n[Component distribution summaries]")
    print(
        f"{'name':<8}{'mean':>11}{'std':>11}{'q01':>11}{'median':>11}"
        f"{'q99':>11}{'skew':>11}{'kurt':>11}{'outlier%':>11}"
    )
    all_values = {
        **{name: data.static_class_z[name] for name in STATIC_NAMES},
        **data.dynamic,
        "Target": data.pseudo_target,
    }
    for name, values in all_values.items():
        stats = vector_summary(values)
        print(
            f"{name:<8}{stats['mean']:>11.5f}{stats['std']:>11.5f}"
            f"{stats['q01']:>11.5f}{stats['median']:>11.5f}{stats['q99']:>11.5f}"
            f"{stats['skew']:>11.5f}{stats['excess_kurtosis']:>11.5f}"
            f"{100.0 * stats['outlier_fraction']:>10.3f}%"
        )


def print_positive_fits(data: DiagnosticData) -> dict[str, FitResult]:
    x = np.column_stack([data.static_class_z[name] for name in STATIC_NAMES])
    targets = {**data.dynamic, "Target": data.pseudo_target}
    results = {name: positive_fit(x, values) for name, values in targets.items()}
    print("\n[Positive diagnostic fits: static components -> each target]")
    print(f"{'target':<10}{'SA':>11}{'Div':>11}{'DDS':>11}{'R2':>11}{'pred_std':>12}")
    for name in (*DYNAMIC_NAMES, "Target"):
        result = results[name]
        print(
            f"{name:<10}{result.weights[0]:>11.5f}{result.weights[1]:>11.5f}"
            f"{result.weights[2]:>11.5f}{result.r2:>11.5f}"
            f"{result.prediction_std:>12.5f}"
        )
    return results


def print_strongest_correlations(
    pearson_cross: np.ndarray,
    spearman_cross: np.ndarray,
    top_k: int,
) -> None:
    rows: list[tuple[float, str, str, float, float]] = []
    for i, static_name in enumerate(STATIC_NAMES):
        for j, dynamic_name in enumerate(DYNAMIC_NAMES):
            rows.append(
                (
                    abs(float(spearman_cross[i, j])),
                    static_name,
                    dynamic_name,
                    float(pearson_cross[i, j]),
                    float(spearman_cross[i, j]),
                )
            )
    rows.sort(reverse=True)
    print(f"\n[Top {min(top_k, len(rows))} static/dynamic relationships by |Spearman|]")
    for _, static_name, dynamic_name, p_value, s_value in rows[:top_k]:
        print(
            f"  {static_name:>3} vs {dynamic_name}: "
            f"Pearson={p_value:+.5f}, Spearman={s_value:+.5f}"
        )


def print_corruption_effects(data: DiagnosticData) -> None:
    if data.is_corrupted is None:
        return
    corrupted = np.asarray(data.is_corrupted, dtype=bool)
    if corrupted.shape != (len(data.labels),) or not np.any(corrupted) or np.all(corrupted):
        return
    print("\n[Experiment-3 clean/corrupted component effects]")
    print(f"{'name':<9}{'clean_mean':>13}{'corr_mean':>13}{'Cohen_d':>12}")
    values_map = {
        **{name: data.static_class_z[name] for name in STATIC_NAMES},
        **data.dynamic,
        "Target": data.pseudo_target,
    }
    for name, values in values_map.items():
        clean_values = values[~corrupted]
        corrupt_values = values[corrupted]
        pooled = math.sqrt(
            0.5 * (float(np.var(clean_values)) + float(np.var(corrupt_values)))
        )
        effect = 0.0 if pooled < EPS else (
            float(np.mean(corrupt_values)) - float(np.mean(clean_values))
        ) / pooled
        print(
            f"{name:<9}{float(np.mean(clean_values)):>13.5f}"
            f"{float(np.mean(corrupt_values)):>13.5f}{effect:>12.5f}"
        )


def diagnose_components(
    data: DiagnosticData,
    spearman_cross: np.ndarray,
    dynamic_pairwise: np.ndarray,
    fits: Mapping[str, FitResult],
) -> None:
    print("\n[Heuristic diagnosis]")
    contributions: list[tuple[float, str, float, float, float]] = []
    for dynamic_index, dynamic_name in enumerate(DYNAMIC_NAMES):
        sa_abs = abs(float(spearman_cross[0, dynamic_index]))
        other_abs = max(
            abs(float(spearman_cross[1, dynamic_index])),
            abs(float(spearman_cross[2, dynamic_index])),
        )
        sa_gap = sa_abs - other_abs
        fit = fits[dynamic_name]
        sa_fit_weight = float(fit.weights[0])
        score = max(0.0, sa_gap) * max(0.0, fit.r2) * sa_fit_weight
        contributions.append((score, dynamic_name, sa_gap, sa_fit_weight, fit.r2))

    contributions.sort(reverse=True)
    for score, name, gap, sa_weight, r2 in contributions:
        print(
            f"  {name}: SA-gap={gap:+.5f}, positive-fit-SA={sa_weight:.5f}, "
            f"fit-R2={r2:.5f}, suspicion-score={score:.6f}"
        )

    target_correlations = np.asarray(
        [spearman(data.static_class_z[name], data.pseudo_target) for name in STATIC_NAMES]
    )
    abs_target = np.abs(target_correlations)
    order = np.argsort(abs_target)[::-1]
    leading = STATIC_NAMES[int(order[0])]
    gap = float(abs_target[order[0]] - abs_target[order[1]])
    print(
        "  Pseudo-target correlations: "
        + ", ".join(
            f"{name}={value:+.5f}" for name, value in zip(STATIC_NAMES, target_correlations)
        )
    )

    max_dynamic_redundancy = 0.0
    redundant_pair: tuple[str, str] | None = None
    for i in range(len(DYNAMIC_NAMES)):
        for j in range(i + 1, len(DYNAMIC_NAMES)):
            value = abs(float(dynamic_pairwise[i, j]))
            if value > max_dynamic_redundancy:
                max_dynamic_redundancy = value
                redundant_pair = (DYNAMIC_NAMES[i], DYNAMIC_NAMES[j])

    if max(abs_target) < 0.08:
        print(
            "  Conclusion: all static components are weakly related to the pseudo target; "
            "a highly concentrated learned weight is likely unstable rather than evidence "
            "that one static metric is genuinely dominant."
        )
    elif leading == "SA" and gap >= 0.08:
        likely_dynamic = contributions[0][1]
        print(
            f"  Conclusion: the pseudo target structurally favours SA. The dynamic "
            f"component most likely contributing to that preference is {likely_dynamic}."
        )
    else:
        print(
            f"  Conclusion: the pseudo target is not uniquely dominated by SA; its "
            f"strongest static association is {leading}, with an absolute-correlation "
            f"gap of {gap:.5f}."
        )

    if redundant_pair is not None and max_dynamic_redundancy >= 0.90:
        print(
            f"  Warning: {redundant_pair[0]} and {redundant_pair[1]} are highly redundant "
            f"(|Spearman|={max_dynamic_redundancy:.5f}); averaging them gives that signal "
            "double influence in the pseudo target."
        )

    weak_components = []
    for index, name in enumerate(DYNAMIC_NAMES):
        if float(np.max(np.abs(spearman_cross[:, index]))) < 0.05:
            weak_components.append(name)
    if weak_components:
        print(
            "  Warning: dynamic components with almost no monotonic relationship to any "
            f"static component: {', '.join(weak_components)}."
        )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    if window > len(values):
        raise ValueError(f"smooth window {window} exceeds epoch count {len(values)}")
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return result


def prepare_proxy_curves(
    metrics: ProxyMetrics,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if normalization == "none":
        return (
            metrics.train_loss.copy(),
            metrics.val_loss.copy(),
            metrics.val_accuracy.copy(),
            "Value",
        )
    loss_low = float(min(np.min(metrics.train_loss), np.min(metrics.val_loss)))
    loss_high = float(max(np.max(metrics.train_loss), np.max(metrics.val_loss)))
    train_curve = minmax(metrics.train_loss, loss_low, loss_high)
    val_curve = minmax(metrics.val_loss, loss_low, loss_high)
    accuracy_curve = minmax(
        metrics.val_accuracy,
        float(np.min(metrics.val_accuracy)),
        float(np.max(metrics.val_accuracy)),
    )
    return train_curve, val_curve, accuracy_curve, "Relative value"


def minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if high - low < EPS:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def save_proxy_plot(
    path: Path,
    target: Target,
    metrics: ProxyMetrics,
    normalization: str,
    smooth_window: int,
) -> None:
    train_curve, val_curve, accuracy_curve, y_label = prepare_proxy_curves(
        metrics,
        normalization,
    )
    train_curve = moving_average(train_curve, smooth_window)
    val_curve = moving_average(val_curve, smooth_window)
    accuracy_curve = moving_average(accuracy_curve, smooth_window)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(metrics.epochs, train_curve, label="Train mean loss", linewidth=2)
    axis.plot(metrics.epochs, val_curve, label="Validation mean loss", linewidth=2)
    axis.plot(metrics.epochs, accuracy_curve, label="Validation accuracy", linewidth=2)
    axis.set_xlabel("Training epoch")
    axis.set_ylabel(y_label)
    axis.set_title(
        f"Experiment {target.exp} - {target.dataset} proxy dynamics\n"
        f"Seed: {target.seed}, epochs: {target.epochs}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    if normalization == "relative":
        axis.set_ylim(-0.05, 1.05)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def annotate_heatmap(axis, matrix: np.ndarray, rows: tuple[str, ...], columns: tuple[str, ...], title: str) -> None:
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axis.set_xticks(np.arange(len(columns)), labels=columns)
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
            )
    return image


def save_correlation_plot(
    path: Path,
    target: Target,
    pearson_cross: np.ndarray,
    spearman_cross: np.ndarray,
    static_pairwise: np.ndarray,
    dynamic_pairwise: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    image = annotate_heatmap(
        axes[0, 0],
        pearson_cross,
        STATIC_NAMES,
        DYNAMIC_NAMES,
        "Static vs dynamic: Pearson",
    )
    annotate_heatmap(
        axes[0, 1],
        spearman_cross,
        STATIC_NAMES,
        DYNAMIC_NAMES,
        "Static vs dynamic: Spearman",
    )
    annotate_heatmap(
        axes[1, 0],
        static_pairwise,
        STATIC_NAMES,
        STATIC_NAMES,
        "Static pairwise: Spearman",
    )
    annotate_heatmap(
        axes[1, 1],
        dynamic_pairwise,
        DYNAMIC_NAMES,
        DYNAMIC_NAMES,
        "Dynamic pairwise: Spearman",
    )
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Correlation")
    figure.suptitle(
        f"Experiment {target.exp} - {target.dataset} - seed {target.seed}",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.91, wspace=0.28, hspace=0.28)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_distribution_plot(path: Path, target: Target, data: DiagnosticData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = (*STATIC_NAMES, *DYNAMIC_NAMES, "Target")
    values = [
        *(data.static_class_z[name] for name in STATIC_NAMES),
        *(data.dynamic[name] for name in DYNAMIC_NAMES),
        data.pseudo_target,
    ]
    figure, axis = plt.subplots(figsize=(11, 6))
    try:
        axis.boxplot(values, tick_labels=names, showfliers=False)
    except TypeError:
        # Compatibility with Matplotlib versions before ``tick_labels``.
        axis.boxplot(values, labels=names, showfliers=False)
    axis.axhline(0.0, linewidth=1)
    axis.set_ylabel("Standardized value")
    axis.set_title(
        f"Component distributions - experiment {target.exp}, "
        f"{target.dataset}, seed {target.seed}"
    )
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    target, paths = resolve_target(args)
    output_dir = resolve_project_path(args.output_dir)

    print(
        f"[Target] exp={target.exp}, dataset={target.dataset}, seed={target.seed}, "
        f"proxy_model={target.proxy_model}, epochs={target.epochs}"
    )
    print(f"[Cache] proxy={paths.proxy_dir}")
    print(f"[Cache] dynamic={paths.dynamic_dir}")
    print(f"[Cache] static={paths.static_path}")

    data = load_diagnostic_data(paths)
    metrics = compute_proxy_metrics(paths.proxy_dir, data.labels)

    static_primary = {name: data.static_class_z[name] for name in STATIC_NAMES}
    pearson_cross = correlation_matrix(static_primary, data.dynamic, "pearson")
    spearman_cross = correlation_matrix(static_primary, data.dynamic, "spearman")
    static_pairwise = pairwise_matrix(static_primary, "spearman")
    dynamic_pairwise = pairwise_matrix(data.dynamic, "spearman")

    print_matrix(
        "[Pearson] class-standardized static vs dynamic",
        pearson_cross,
        STATIC_NAMES,
        DYNAMIC_NAMES,
    )
    print_matrix(
        "[Spearman] class-standardized static vs dynamic",
        spearman_cross,
        STATIC_NAMES,
        DYNAMIC_NAMES,
    )
    print_matrix(
        "[Spearman] static-component redundancy",
        static_pairwise,
        STATIC_NAMES,
        STATIC_NAMES,
    )
    print_matrix(
        "[Spearman] dynamic-component redundancy",
        dynamic_pairwise,
        DYNAMIC_NAMES,
        DYNAMIC_NAMES,
    )
    print_strongest_correlations(pearson_cross, spearman_cross, args.top_k)
    print_component_summaries(data)
    fits = print_positive_fits(data)

    saved_weights = load_saved_weights(paths.weights_path, target)
    if saved_weights is not None:
        print(
            "\n[Saved stage-5 weights] "
            + ", ".join(f"{name}={saved_weights[name]:.5f}" for name in STATIC_NAMES)
        )

    print_corruption_effects(data)
    diagnose_components(data, spearman_cross, dynamic_pairwise, fits)

    stem = f"exp{target.exp}_{target.dataset.replace('-', '_')}_seed{target.seed}"
    proxy_path = output_dir / f"{stem}_proxy_dynamics.png"
    correlation_path = output_dir / f"{stem}_component_correlations.png"
    distribution_path = output_dir / f"{stem}_component_distributions.png"

    save_proxy_plot(
        proxy_path,
        target,
        metrics,
        args.normalization,
        args.smooth_window,
    )
    save_correlation_plot(
        correlation_path,
        target,
        pearson_cross,
        spearman_cross,
        static_pairwise,
        dynamic_pairwise,
    )
    save_distribution_plot(distribution_path, target, data)

    print("\n[Saved figures]")
    print(f"  {proxy_path}")
    print(f"  {correlation_path}")
    print(f"  {distribution_path}")
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