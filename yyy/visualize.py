"""Visualize proxy-model cross-validation dynamics.

This script is intended to be placed at ``yyy/visualize.py`` in the project
root. It uses the existing proxy logs and never retrains a proxy model.

Default behavior:
- random seed is fixed to 22;
- dataset is CIFAR-100;
- proxy model is ResNet-18;
- the largest available proxy-log run is loaded;
- all available epochs are used unless ``--max-epoch`` is specified;
- four min-max-normalized trends are plotted in one figure:
  train accuracy, validation accuracy, validation loss mean, and validation
  loss variance.

The raw per-epoch values are also saved as a CSV file next to the figure.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np


SEED = 22
DATASET_ALIASES = {
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "tiny-imagenet": "tiny-imagenet",
    "tiny_imagenet": "tiny-imagenet",
    "tinyimagenet": "tiny-imagenet",
}

# yyy/visualize.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.proxy_log_utils import load_dataset_labels  # noqa: E402
from utils.training_defaults import get_default_training_config  # noqa: E402


def parse_dataset(value: str) -> str:
    """Normalize user-facing dataset names to repository dataset identifiers."""
    key = value.strip().lower()
    if key not in DATASET_ALIASES:
        choices = "CIFAR100 or Tiny-ImageNet"
        raise argparse.ArgumentTypeError(f"Unsupported dataset {value!r}; choose {choices}.")
    return DATASET_ALIASES[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot existing proxy-CV training dynamics without retraining the proxy model."
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        default="cifar100",
        metavar="DATASET",
        help="Dataset: CIFAR100 or Tiny-ImageNet. Default: CIFAR100.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help=(
            "Use only epochs 1..MAX_EPOCH from an existing log run. "
            "The source run may contain more epochs. Default: use all available epochs."
        ),
    )
    parser.add_argument(
        "--proxy-model",
        type=str,
        default="resnet18",
        help="Proxy-model directory name. Default: resnet18.",
    )
    parser.add_argument(
        "--proxy-log-root",
        type=Path,
        default=PROJECT_ROOT / "weights" / "proxy_logs",
        help="Root of clean proxy logs. Default: weights/proxy_logs.",
    )
    parser.add_argument(
        "--proxy-log-dir",
        type=Path,
        default=None,
        help=(
            "Directly specify a directory containing fold_*.npz. "
            "When set, --proxy-log-root is ignored."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Dataset root used to recover labels. Default: data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: yyy/outputs/<dataset>/seed_22/.",
    )
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=4096,
        help="Samples processed at once for each epoch. Default: 4096.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure resolution. Default: 180.",
    )
    args = parser.parse_args()

    if args.max_epoch is not None and args.max_epoch <= 0:
        parser.error("--max-epoch must be positive.")
    if args.sample_chunk_size <= 0:
        parser.error("--sample-chunk-size must be positive.")
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    return args


def contains_fold_logs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("fold_*.npz"))


def resolve_proxy_log_dir(args: argparse.Namespace) -> Path:
    """Choose an existing source run; cutoff and source-run length are separate."""
    if args.proxy_log_dir is not None:
        direct = args.proxy_log_dir.expanduser().resolve()
        if not contains_fold_logs(direct):
            raise FileNotFoundError(f"No fold_*.npz files found in --proxy-log-dir: {direct}")
        return direct

    seed_dir = (
        args.proxy_log_root.expanduser().resolve()
        / args.dataset
        / args.proxy_model
        / str(SEED)
    )
    if not seed_dir.is_dir():
        raise FileNotFoundError(
            "Proxy-log seed directory does not exist: "
            f"{seed_dir}\nExpected structure: "
            "<proxy-log-root>/<dataset>/<proxy-model>/22/<source-epochs>/fold_*.npz"
        )

    epoch_dirs = [
        path
        for path in seed_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and contains_fold_logs(path)
    ]
    if not epoch_dirs:
        raise FileNotFoundError(f"No numeric epoch directory with fold_*.npz under: {seed_dir}")

    epoch_dirs.sort(key=lambda path: int(path.name))
    if args.max_epoch is None:
        return epoch_dirs[-1]

    # Prefer the shortest existing run that still contains the requested prefix.
    adequate = [path for path in epoch_dirs if int(path.name) >= args.max_epoch]
    if adequate:
        return adequate[0]

    available = ", ".join(path.name for path in epoch_dirs)
    raise ValueError(
        f"Requested --max-epoch={args.max_epoch}, but available source runs are: {available}."
    )


def load_run_metadata(log_dir: Path) -> dict[str, object]:
    meta_path = log_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def list_fold_paths(log_dir: Path) -> list[Path]:
    fold_paths = sorted(log_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in: {log_dir}")
    return fold_paths


def load_indices(fold_path: Path, key: str) -> np.ndarray:
    with np.load(fold_path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{fold_path} does not contain {key!r}.")
        indices = np.asarray(data[key], dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{key} in {fold_path} must be one-dimensional.")
    return indices


@contextmanager
def open_npz_array_as_memmap(
    fold_path: Path,
    key: str,
    temp_parent: Path,
) -> Iterator[np.ndarray]:
    """Extract one .npy member from an NPZ and open it read-only via mmap.

    ``np.load(npz)[key]`` materializes the complete logits tensor in memory.
    CIFAR-100 and Tiny-ImageNet logs can be several GB per fold, so this helper
    temporarily extracts only the requested member and processes it epoch by
    epoch through a memory map.
    """
    member_name = f"{key}.npy"
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{fold_path.stem}_{key}_", dir=temp_parent))
    extracted_path = temp_dir / member_name
    array: np.ndarray | None = None

    try:
        with zipfile.ZipFile(fold_path, mode="r") as archive:
            if member_name not in archive.namelist():
                raise KeyError(f"{fold_path} does not contain {member_name!r}.")
            with archive.open(member_name, mode="r") as source, extracted_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)

        array = np.load(extracted_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(
                f"{key} in {fold_path} must have shape (epochs, samples, classes); "
                f"got {array.shape}."
            )
        yield array
    finally:
        if array is not None:
            del array
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def cross_entropy_per_sample(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Numerically stable cross-entropy for one epoch."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D logits for one epoch; got {values.shape}.")
    if labels.shape != (values.shape[0],):
        raise ValueError("Labels do not match the number of samples in logits.")

    row_max = np.max(values, axis=1)
    shifted = values - row_max[:, None]
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + row_max
    true_logits = values[np.arange(values.shape[0]), labels]
    losses = logsumexp - true_logits
    return np.nan_to_num(losses, nan=0.0, posinf=1.0e6, neginf=0.0)


def read_npz_array_shape(fold_path: Path, key: str) -> tuple[int, ...]:
    """Read an NPY member's shape from its header without extracting the array."""
    member_name = f"{key}.npy"
    with zipfile.ZipFile(fold_path, mode="r") as archive:
        if member_name not in archive.namelist():
            raise KeyError(f"{fold_path} does not contain {member_name!r}.")
        with archive.open(member_name, mode="r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            elif version == (3, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {fold_path}:{member_name}")
    return tuple(int(value) for value in shape)


def inspect_available_epochs(fold_paths: list[Path]) -> int:
    """Validate epoch counts using only the small NPY headers inside each NPZ."""
    epoch_counts: list[int] = []
    for fold_path in fold_paths:
        for key in ("train_logits", "val_logits"):
            shape = read_npz_array_shape(fold_path, key)
            if len(shape) != 3:
                raise ValueError(
                    f"{key} in {fold_path} must have shape (epochs, samples, classes); got {shape}."
                )
            epoch_counts.append(shape[0])
    unique_counts = sorted(set(epoch_counts))
    if len(unique_counts) != 1:
        raise ValueError(f"Inconsistent epoch counts across fold logs: {unique_counts}")
    return unique_counts[0]


def iter_sample_slices(num_samples: int, chunk_size: int) -> Iterator[slice]:
    for start in range(0, num_samples, chunk_size):
        yield slice(start, min(start + chunk_size, num_samples))


def collect_metrics(
    fold_paths: list[Path],
    labels_all: np.ndarray,
    max_epoch: int,
    temp_parent: Path,
    sample_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Aggregate fold/sample-weighted metrics for epochs 1..max_epoch."""
    train_correct = np.zeros(max_epoch, dtype=np.float64)
    train_count = np.zeros(max_epoch, dtype=np.int64)
    val_correct = np.zeros(max_epoch, dtype=np.float64)
    val_count = np.zeros(max_epoch, dtype=np.int64)
    val_loss_sum = np.zeros(max_epoch, dtype=np.float64)
    val_loss_sq_sum = np.zeros(max_epoch, dtype=np.float64)

    num_samples = int(labels_all.shape[0])

    for fold_number, fold_path in enumerate(fold_paths, start=1):
        print(f"Processing fold {fold_number}/{len(fold_paths)}: {fold_path.name}")
        train_indices = load_indices(fold_path, "train_indices")
        val_indices = load_indices(fold_path, "val_indices")

        for name, indices in (("train_indices", train_indices), ("val_indices", val_indices)):
            if np.any(indices < 0) or np.any(indices >= num_samples):
                raise ValueError(f"{name} out of range in {fold_path}.")

        y_train = labels_all[train_indices]
        y_val = labels_all[val_indices]

        with open_npz_array_as_memmap(fold_path, "train_logits", temp_parent) as logits:
            if logits.shape[0] < max_epoch or logits.shape[1] != train_indices.size:
                raise ValueError(
                    f"train_logits shape mismatch in {fold_path}: {logits.shape}, "
                    f"train_indices={train_indices.size}, requested_epochs={max_epoch}."
                )
            for epoch_idx in range(max_epoch):
                for sample_slice in iter_sample_slices(y_train.size, sample_chunk_size):
                    epoch_logits = np.asarray(logits[epoch_idx, sample_slice, :])
                    labels_chunk = y_train[sample_slice]
                    predictions = np.argmax(epoch_logits, axis=1)
                    train_correct[epoch_idx] += float(
                        np.count_nonzero(predictions == labels_chunk)
                    )
                    train_count[epoch_idx] += labels_chunk.size

        with open_npz_array_as_memmap(fold_path, "val_logits", temp_parent) as logits:
            if logits.shape[0] < max_epoch or logits.shape[1] != val_indices.size:
                raise ValueError(
                    f"val_logits shape mismatch in {fold_path}: {logits.shape}, "
                    f"val_indices={val_indices.size}, requested_epochs={max_epoch}."
                )
            for epoch_idx in range(max_epoch):
                for sample_slice in iter_sample_slices(y_val.size, sample_chunk_size):
                    epoch_logits = np.asarray(logits[epoch_idx, sample_slice, :])
                    labels_chunk = y_val[sample_slice]
                    predictions = np.argmax(epoch_logits, axis=1)
                    losses = cross_entropy_per_sample(epoch_logits, labels_chunk)

                    val_correct[epoch_idx] += float(
                        np.count_nonzero(predictions == labels_chunk)
                    )
                    val_count[epoch_idx] += labels_chunk.size
                    val_loss_sum[epoch_idx] += float(np.sum(losses, dtype=np.float64))
                    val_loss_sq_sum[epoch_idx] += float(
                        np.sum(losses * losses, dtype=np.float64)
                    )

    if np.any(train_count <= 0) or np.any(val_count <= 0):
        raise RuntimeError("At least one epoch has no aggregated train or validation samples.")

    train_accuracy = train_correct / train_count
    val_accuracy = val_correct / val_count
    val_loss_mean = val_loss_sum / val_count
    val_loss_variance = val_loss_sq_sum / val_count - val_loss_mean**2
    val_loss_variance = np.maximum(val_loss_variance, 0.0)

    return {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "val_loss_mean": val_loss_mean,
        "val_loss_variance": val_loss_variance,
    }


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError("Metric contains NaN or infinity.")
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper - lower < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lower) / (upper - lower)


def save_csv(
    output_path: Path,
    epochs: np.ndarray,
    metrics: dict[str, np.ndarray],
    normalized: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "epoch",
        "train_accuracy",
        "val_accuracy",
        "val_loss_mean",
        "val_loss_variance",
        "train_accuracy_normalized",
        "val_accuracy_normalized",
        "val_loss_mean_normalized",
        "val_loss_variance_normalized",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, epoch in enumerate(epochs):
            writer.writerow(
                {
                    "epoch": int(epoch),
                    "train_accuracy": float(metrics["train_accuracy"][idx]),
                    "val_accuracy": float(metrics["val_accuracy"][idx]),
                    "val_loss_mean": float(metrics["val_loss_mean"][idx]),
                    "val_loss_variance": float(metrics["val_loss_variance"][idx]),
                    "train_accuracy_normalized": float(normalized["train_accuracy"][idx]),
                    "val_accuracy_normalized": float(normalized["val_accuracy"][idx]),
                    "val_loss_mean_normalized": float(normalized["val_loss_mean"][idx]),
                    "val_loss_variance_normalized": float(normalized["val_loss_variance"][idx]),
                }
            )


def resolve_lr_milestones(dataset: str, metadata: dict[str, object]) -> list[int]:
    raw = metadata.get("lr_milestones")
    if isinstance(raw, list):
        try:
            return sorted({int(value) for value in raw})
        except (TypeError, ValueError):
            pass
    defaults = get_default_training_config(dataset)
    return [int(value) for value in defaults["lr_milestones"]]


def plot_metrics(
    output_path: Path,
    dataset: str,
    source_log_dir: Path,
    epochs: np.ndarray,
    normalized: dict[str, np.ndarray],
    lr_milestones: list[int],
    dpi: int,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))

    # Matplotlib's default color cycle gives each metric a different color.
    axis.plot(epochs, normalized["train_accuracy"], linewidth=2.0, label="Train accuracy")
    axis.plot(epochs, normalized["val_accuracy"], linewidth=2.0, label="Validation accuracy")
    axis.plot(epochs, normalized["val_loss_mean"], linewidth=2.0, label="Validation loss mean")
    axis.plot(
        epochs,
        normalized["val_loss_variance"],
        linewidth=2.0,
        label="Validation loss variance",
    )

    for milestone in lr_milestones:
        if 1 <= milestone <= int(epochs[-1]):
            axis.axvline(milestone, linestyle="--", linewidth=1.0, alpha=0.55)
            axis.text(
                milestone,
                1.015,
                f"LR decay {milestone}",
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )

    display_name = "CIFAR-100" if dataset == "cifar100" else "Tiny-ImageNet"
    axis.set_title(
        f"Proxy CV dynamics: {display_name}, seed={SEED}\n"
        f"source={source_log_dir.name} epochs, plotted=1-{int(epochs[-1])}"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Min-max normalized value")
    axis.set_xlim(int(epochs[0]), int(epochs[-1]))
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
    axis.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_dir = resolve_proxy_log_dir(args)
    fold_paths = list_fold_paths(log_dir)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "yyy" / "outputs" / args.dataset / f"seed_{SEED}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_parent = output_dir / ".tmp_npz_extract"
    temp_parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Dataset: {args.dataset}")
        print(f"Fixed seed: {SEED}")
        print(f"Proxy log: {log_dir}")
        print(f"Fold count: {len(fold_paths)}")

        available_epochs = inspect_available_epochs(fold_paths)
        max_epoch = available_epochs if args.max_epoch is None else int(args.max_epoch)
        if max_epoch > available_epochs:
            raise ValueError(
                f"Requested {max_epoch} epochs, but fold logs contain only {available_epochs}."
            )
        print(f"Available epochs: {available_epochs}; plotted prefix: 1-{max_epoch}")

        labels_all = load_dataset_labels(args.dataset, str(args.data_root.expanduser().resolve()))
        labels_all = np.asarray(labels_all, dtype=np.int64)

        metrics = collect_metrics(
            fold_paths=fold_paths,
            labels_all=labels_all,
            max_epoch=max_epoch,
            temp_parent=temp_parent,
            sample_chunk_size=args.sample_chunk_size,
        )
        normalized = {name: minmax_normalize(values) for name, values in metrics.items()}
        epochs = np.arange(1, max_epoch + 1, dtype=np.int64)

        stem = f"proxy_cv_dynamics_{args.dataset}_seed_{SEED}_epoch_{max_epoch}"
        figure_path = output_dir / f"{stem}.png"
        csv_path = output_dir / f"{stem}.csv"

        metadata = load_run_metadata(log_dir)
        lr_milestones = resolve_lr_milestones(args.dataset, metadata)
        plot_metrics(
            output_path=figure_path,
            dataset=args.dataset,
            source_log_dir=log_dir,
            epochs=epochs,
            normalized=normalized,
            lr_milestones=lr_milestones,
            dpi=args.dpi,
        )
        save_csv(csv_path, epochs, metrics, normalized)

        print(f"Figure saved to: {figure_path}")
        print(f"Raw and normalized metrics saved to: {csv_path}")
        print(
            "Final plotted epoch: "
            f"train_acc={metrics['train_accuracy'][-1]:.6f}, "
            f"val_acc={metrics['val_accuracy'][-1]:.6f}, "
            f"val_loss_mean={metrics['val_loss_mean'][-1]:.6f}, "
            f"val_loss_var={metrics['val_loss_variance'][-1]:.6f}"
        )
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


if __name__ == "__main__":
    main()