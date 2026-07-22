#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check correlations between dynamic signals and static Semantic Alignment.

The script reuses ``yyy/try_2.py`` for:

- CIFAR-100 160-epoch fixed windows: 1-60 / 61-120 / 121-160;
- Tiny-ImageNet 80-epoch fixed windows: 1-30 / 31-60 / 61-80;
- normal / label_noise / corruption experiment paths;
- dynamic caches under ``yyy/dynamic_cache``;
- existing project static-score caches;
- ``--keep`` semantics for original 200/90-epoch dynamic caches.

Reported dynamic signals
------------------------
- A_boundary: standardized early boundary-information sub-signal of A;
- A_gain: standardized early-to-middle absorption-gain sub-signal of A;
- A_stability: standardized late-stage stability sub-signal of A;
- A, C, T;
- noise_gate.

The three A sub-signals are not present in the repository's A cache. They are
computed from existing proxy logs only when their dedicated cache is absent and
saved under:

    yyy/a_subcomponents_cache/<strategy>/<experiment>/<dataset>/<epochs>/
        A_subcomponents.npz

When ``A`` is listed in ``--keep``, the A sub-signals are computed with the
original full-run definition and the repository's original 0.3/0.4/0.3 window
resolver over 200 epochs (CIFAR-100) or 90 epochs (Tiny-ImageNet). Otherwise,
they use try_2.py's fixed 160/80 windows.

Correlation targets
-------------------
The primary target is ``SA_class_z``, the class-wise standardized SA feature
actually passed to weight regression. Correlations with raw SA are also saved
for reference. Both Pearson and Spearman coefficients are reported.

The script does not fit weights and does not calculate masks.

Examples
--------
Run all datasets and all experiments using try_2 fixed-window caches:

    python yyy/checkSA.py

Run only CIFAR-100 corruption:

    python yyy/checkSA.py --dataset cifar100 --experiment corruption

Read original full-run A, C, T and noise-gate caches:

    python yyy/checkSA.py --keep A,C,T,noise_gate

Use original A only; the other dynamic signals still use try_2 fixed windows:

    python yyy/checkSA.py --keep A

Force only the dedicated A-subcomponent cache to be recomputed:

    python yyy/checkSA.py --force-a-subcomponents
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
YYY_ROOT = THIS_FILE.parent
PROJECT_ROOT = YYY_ROOT.parent

for path in (YYY_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import try_2 as trial
except ImportError as exc:
    raise ImportError("yyy/checkSA.py requires yyy/try_2.py.") from exc

from weights.dynamic_utils import (  # noqa: E402
    safe_standardize,
    standard_zscore_dynamic,
)


A_SUBCOMPONENT_CACHE_VERSION = "checkSA_a_subcomponents_v1"
A_SUBCOMPONENT_NAMES = ("A_boundary", "A_gain", "A_stability")
DYNAMIC_NAMES = ("A", "C", "T", "noise_gate")
ALL_SIGNAL_NAMES = (*A_SUBCOMPONENT_NAMES, *DYNAMIC_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute correlations of A sub-signals and A/C/T/noise_gate "
            "with static Semantic Alignment."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *trial.SUPPORTED_DATASETS),
        help="Dataset to check. Default: CIFAR-100 and Tiny-ImageNet.",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=("all", "special", *trial.EXPERIMENTS),
        help="Experiment variant. 'special' means label_noise + corruption.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--proxy-model", default=trial.PROXY_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument("--dds-k", type=int, default=5)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.8)
    parser.add_argument("--debug-prompts", action="store_true")

    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--normal-gate-low", type=float, default=0.1)
    parser.add_argument("--normal-gate-high", type=float, default=0.9)
    parser.add_argument("--special-gate-low", type=float, default=0.2)
    parser.add_argument("--special-gate-high", type=float, default=0.95)

    parser.add_argument(
        "--keep",
        default="",
        help=(
            "Comma-separated dynamic metrics read from original 200/90-epoch "
            "experiment caches. Supported: A,C,T,noise_gate."
        ),
    )
    parser.add_argument(
        "--force-dynamic",
        action="store_true",
        help="Force recomputation of non-kept try_2 dynamic caches.",
    )
    parser.add_argument(
        "--force-a-subcomponents",
        action="store_true",
        help="Ignore and recompute the dedicated A-subcomponent cache.",
    )
    parser.add_argument(
        "--epoch-chunk-size",
        type=int,
        default=4,
        help=(
            "Number of epochs processed together while deriving true-class "
            "probabilities for A sub-signals. Smaller values reduce temporary memory."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=YYY_ROOT / "checkSA_results",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.keep_items = trial.parse_keep_items(args.keep)
    if args.epoch_chunk_size <= 0:
        raise ValueError("--epoch-chunk-size must be positive.")
    if args.learn_window <= 0:
        raise ValueError("--learn-window must be positive.")
    if not 0 < args.learn_min_correct <= args.learn_window:
        raise ValueError("--learn-min-correct must be in [1, learn-window].")
    for low, high, name in (
        (args.normal_gate_low, args.normal_gate_high, "normal"),
        (args.special_gate_low, args.special_gate_high, "special"),
    ):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"{name} gate thresholds must satisfy 0 <= low < high <= 1."
            )


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Rank input must be a finite one-dimensional array.")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    xx = x[finite]
    yy = y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    return pearson_corr(average_ranks(x[finite]), average_ranks(y[finite]))


def special_signature(ctx: trial.ExperimentContext) -> str:
    if ctx.experiment == "corruption" and ctx.corruption_info is not None:
        value = getattr(ctx.corruption_info, "list_hash", None)
        return str(value) if value is not None else ""
    if ctx.experiment == "label_noise":
        return hashlib.sha1(
            np.asarray(ctx.labels, dtype=np.int64).tobytes()
        ).hexdigest()
    return ""


def a_strategy(ctx: trial.ExperimentContext, keep_items: frozenset[str]) -> tuple[str, int]:
    if "A" in keep_items:
        return "original_full_run", int(trial.SOURCE_PROXY_EPOCHS[ctx.dataset])
    return "fixed_lr_windows_v2", int(trial.DYNAMIC_EPOCHS[ctx.dataset])


def resolve_a_windows(
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy == "original_full_run":
        return trial.resolve_epoch_windows(int(epochs))
    if strategy == "fixed_lr_windows_v2":
        return trial.resolve_fixed_epoch_windows(ctx.dataset, int(epochs))
    raise ValueError(f"Unknown A-subcomponent strategy: {strategy}")


def a_subcomponent_cache_path(
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
) -> Path:
    return (
        YYY_ROOT
        / "a_subcomponents_cache"
        / strategy
        / ctx.experiment
        / ctx.dataset
        / str(int(epochs))
        / "A_subcomponents.npz"
    )


def load_a_subcomponent_cache(
    path: Path,
    *,
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
    source_log_dir: Path | None,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    early, middle, late = windows
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "cache_version",
                "experiment",
                "dataset",
                "seed",
                "strategy",
                "epochs",
                "source_log_dir",
                "special_signature",
                "labels",
                "early_idx",
                "middle_idx",
                "late_idx",
                "A_boundary",
                "A_gain",
                "A_stability",
                "A_recomputed",
            }
            if not required.issubset(set(data.files)):
                return None
            scalar_checks = (
                (str(np.asarray(data["cache_version"]).item()), A_SUBCOMPONENT_CACHE_VERSION),
                (str(np.asarray(data["experiment"]).item()), ctx.experiment),
                (str(np.asarray(data["dataset"]).item()), ctx.dataset),
                (int(np.asarray(data["seed"]).item()), int(trial.SEED)),
                (str(np.asarray(data["strategy"]).item()), strategy),
                (int(np.asarray(data["epochs"]).item()), int(epochs)),
                (
                    str(np.asarray(data["special_signature"]).item()),
                    special_signature(ctx),
                ),
            )
            if any(actual != expected for actual, expected in scalar_checks):
                return None
            if source_log_dir is not None and (
                str(np.asarray(data["source_log_dir"]).item())
                != str(source_log_dir.resolve())
            ):
                return None
            if not np.array_equal(
                np.asarray(data["labels"], dtype=np.int64), ctx.labels
            ):
                return None
            if not np.array_equal(np.asarray(data["early_idx"], dtype=np.int64), early):
                return None
            if not np.array_equal(np.asarray(data["middle_idx"], dtype=np.int64), middle):
                return None
            if not np.array_equal(np.asarray(data["late_idx"], dtype=np.int64), late):
                return None

            result = {
                name: np.asarray(data[name], dtype=np.float64)
                for name in (*A_SUBCOMPONENT_NAMES, "A_recomputed")
            }
    except Exception:
        return None

    for name, values in result.items():
        if values.shape != (ctx.labels.size,) or not np.all(np.isfinite(values)):
            return None
    return result


def save_a_subcomponent_cache(
    path: Path,
    *,
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
    source_log_dir: Path,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    result: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    early, middle, late = windows
    np.savez_compressed(
        path,
        cache_version=np.array(A_SUBCOMPONENT_CACHE_VERSION, dtype=np.str_),
        experiment=np.array(ctx.experiment, dtype=np.str_),
        dataset=np.array(ctx.dataset, dtype=np.str_),
        seed=np.array(int(trial.SEED), dtype=np.int64),
        strategy=np.array(strategy, dtype=np.str_),
        epochs=np.array(int(epochs), dtype=np.int64),
        source_log_dir=np.array(str(source_log_dir.resolve()), dtype=np.str_),
        special_signature=np.array(special_signature(ctx), dtype=np.str_),
        labels=np.asarray(ctx.labels, dtype=np.int64),
        early_idx=np.asarray(early, dtype=np.int64),
        middle_idx=np.asarray(middle, dtype=np.int64),
        late_idx=np.asarray(late, dtype=np.int64),
        A_boundary=np.asarray(result["A_boundary"], dtype=np.float32),
        A_gain=np.asarray(result["A_gain"], dtype=np.float32),
        A_stability=np.asarray(result["A_stability"], dtype=np.float32),
        A_recomputed=np.asarray(result["A_recomputed"], dtype=np.float32),
    )


def _validate_contiguous_window(indices: np.ndarray, name: str) -> tuple[int, int]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} window must be a non-empty vector.")
    expected = np.arange(indices[0], indices[-1] + 1, dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError(f"{name} window must be contiguous.")
    return int(indices[0]), int(indices[-1] + 1)


def _true_probability_moments(
    logits: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    epoch_chunk_size: int,
    include_boundary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute mean, variance and optional mean p(1-p) without full softmax storage."""
    start, stop = _validate_contiguous_window(indices, "epoch")
    num_samples = int(labels.size)
    sum_r = np.zeros(num_samples, dtype=np.float64)
    sum_r2 = np.zeros(num_samples, dtype=np.float64)
    sum_boundary = (
        np.zeros(num_samples, dtype=np.float64) if include_boundary else None
    )
    count = 0
    label_index = labels.reshape(1, -1, 1)

    for chunk_start in range(start, stop, int(epoch_chunk_size)):
        chunk_stop = min(stop, chunk_start + int(epoch_chunk_size))
        chunk = np.asarray(logits[chunk_start:chunk_stop], dtype=np.float64)
        if chunk.ndim != 3 or chunk.shape[1] != num_samples:
            raise ValueError("Unexpected train-logit chunk shape.")

        safe = np.nan_to_num(chunk, nan=0.0, posinf=50.0, neginf=-50.0)
        shifted = safe - np.max(safe, axis=2, keepdims=True)
        shifted = np.clip(shifted, -50.0, 50.0)
        exp_shifted = np.exp(shifted)
        denominator = np.sum(exp_shifted, axis=2)
        denominator = np.where(denominator > 1e-12, denominator, 1.0)
        true_exp = np.take_along_axis(
            exp_shifted, label_index, axis=2
        ).squeeze(2)
        probabilities = np.nan_to_num(
            true_exp / denominator,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype(np.float32)

        sum_r += np.sum(probabilities, axis=0, dtype=np.float64)
        sum_r2 += np.sum(
            probabilities.astype(np.float64) ** 2,
            axis=0,
            dtype=np.float64,
        )
        if sum_boundary is not None:
            sum_boundary += np.sum(
                probabilities.astype(np.float64)
                * (1.0 - probabilities.astype(np.float64)),
                axis=0,
                dtype=np.float64,
            )
        count += int(probabilities.shape[0])
        del chunk, safe, shifted, exp_shifted, denominator, true_exp, probabilities

    if count <= 0:
        raise ValueError("Epoch window produced no observations.")
    mean = sum_r / float(count)
    variance = np.maximum(sum_r2 / float(count) - mean * mean, 0.0)
    boundary = (
        sum_boundary / float(count) if sum_boundary is not None else None
    )
    return mean, variance, boundary


def compute_a_subcomponents(
    *,
    ctx: trial.ExperimentContext,
    source_log_dir: Path,
    epochs: int,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    epoch_chunk_size: int,
) -> dict[str, np.ndarray]:
    fold_paths = sorted(
        source_log_dir.glob("fold_*.npz"), key=trial.fold_sort_key
    )
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {source_log_dir}")

    num_samples = int(ctx.labels.size)
    sums = {
        name: np.zeros(num_samples, dtype=np.float64)
        for name in (*A_SUBCOMPONENT_NAMES, "A_recomputed")
    }
    counts = np.zeros(num_samples, dtype=np.int64)
    early, middle, late = windows

    for path in tqdm(
        fold_paths,
        desc=f"A subcomponents {ctx.experiment}/{ctx.dataset}",
        unit="fold",
    ):
        with np.load(path, allow_pickle=False) as data:
            required = {"train_indices", "train_logits"}
            if not required.issubset(set(data.files)):
                raise ValueError(f"{path} missing keys: {sorted(required - set(data.files))}")
            train_idx = np.asarray(data["train_indices"], dtype=np.int64)
            train_logits = data["train_logits"]
            if train_logits.ndim != 3:
                raise ValueError(f"train_logits in {path} must be three-dimensional.")
            if int(train_logits.shape[0]) < int(epochs):
                raise ValueError(
                    f"{path} has {train_logits.shape[0]} epochs, below required {epochs}."
                )
            if int(train_logits.shape[1]) != int(train_idx.size):
                raise ValueError(f"train_logits sample dimension mismatch: {path}")
            if np.any(train_idx < 0) or np.any(train_idx >= num_samples):
                raise ValueError(f"train_indices out of range: {path}")

            labels_train = np.asarray(ctx.labels[train_idx], dtype=np.int64)
            early_mean, _, boundary = _true_probability_moments(
                train_logits,
                labels_train,
                early,
                epoch_chunk_size=epoch_chunk_size,
                include_boundary=True,
            )
            middle_mean, _, _ = _true_probability_moments(
                train_logits,
                labels_train,
                middle,
                epoch_chunk_size=epoch_chunk_size,
                include_boundary=False,
            )
            _, late_variance, _ = _true_probability_moments(
                train_logits,
                labels_train,
                late,
                epoch_chunk_size=epoch_chunk_size,
                include_boundary=False,
            )

        if boundary is None:
            raise RuntimeError("Boundary information was not computed.")

        boundary_z = safe_standardize(boundary)
        gain_z = safe_standardize(middle_mean - early_mean)
        stability_z = safe_standardize(-late_variance)
        a_fold = standard_zscore_dynamic(
            boundary_z + gain_z + stability_z
        )

        fold_values = {
            "A_boundary": boundary_z,
            "A_gain": gain_z,
            "A_stability": stability_z,
            "A_recomputed": a_fold,
        }
        for name, values in fold_values.items():
            sums[name][train_idx] += np.asarray(values, dtype=np.float64)
        counts[train_idx] += 1

        del (
            train_logits,
            train_idx,
            labels_train,
            early_mean,
            middle_mean,
            late_variance,
            boundary,
            boundary_z,
            gain_z,
            stability_z,
            a_fold,
        )
        gc.collect()

    if np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0)
        raise ValueError(
            f"Some samples never appeared in training folds: {missing[:10].tolist()}"
        )

    result: dict[str, np.ndarray] = {}
    for name, total in sums.items():
        aggregated = total / counts
        normalized = standard_zscore_dynamic(aggregated.astype(np.float32))
        if normalized.shape != (num_samples,) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"Invalid derived {name} array.")
        result[name] = np.asarray(normalized, dtype=np.float64)
    return result


def load_or_compute_a_subcomponents(
    ctx: trial.ExperimentContext,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], str, int, Path]:
    strategy, epochs = a_strategy(ctx, args.keep_items)
    windows = resolve_a_windows(ctx, strategy, epochs)
    cache_path = a_subcomponent_cache_path(ctx, strategy, epochs)

    # Dedicated A-subcomponent caches are self-validating. Read them before
    # requiring the original proxy logs, so a completed analysis remains usable
    # even when the large log directory is temporarily unavailable.
    result = None
    if not args.force_a_subcomponents:
        result = load_a_subcomponent_cache(
            cache_path,
            ctx=ctx,
            strategy=strategy,
            epochs=epochs,
            source_log_dir=None,
            windows=windows,
        )
    if result is not None:
        tqdm.write(f"[A-subcomponents] cache HIT: {cache_path}")
        return result, strategy, epochs, cache_path

    source_log_dir = trial.find_proxy_log_dir(
        ctx.experiment,
        ctx.dataset,
        args.proxy_model,
        epochs,
    )
    tqdm.write(f"[A-subcomponents] cache MISS: {cache_path}")
    result = compute_a_subcomponents(
        ctx=ctx,
        source_log_dir=source_log_dir,
        epochs=epochs,
        windows=windows,
        epoch_chunk_size=int(args.epoch_chunk_size),
    )
    save_a_subcomponent_cache(
        cache_path,
        ctx=ctx,
        strategy=strategy,
        epochs=epochs,
        source_log_dir=source_log_dir,
        windows=windows,
        result=result,
    )
    tqdm.write(f"[A-subcomponents] saved: {cache_path}")
    return result, strategy, epochs, cache_path


def signal_source(name: str, args: argparse.Namespace, ctx: trial.ExperimentContext) -> str:
    if name in A_SUBCOMPONENT_NAMES:
        return "original_full_run" if "A" in args.keep_items else "fixed_windows"
    if name in {"A", "C", "T"}:
        return "original_full_run" if name in args.keep_items else "fixed_windows"
    if name == "noise_gate":
        return (
            "original_full_run"
            if "noise_gate" in args.keep_items
            else "fixed_windows"
        )
    raise ValueError(f"Unknown signal: {name}")


def make_rows(
    *,
    ctx: trial.ExperimentContext,
    signals: dict[str, np.ndarray],
    sa_raw: np.ndarray,
    sa_class_z: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ALL_SIGNAL_NAMES:
        values = np.asarray(signals[name], dtype=np.float64)
        if values.shape != sa_raw.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Signal {name} is invalid: shape={values.shape}")
        rows.append(
            {
                "experiment": ctx.experiment,
                "dataset": ctx.dataset,
                "seed": int(trial.SEED),
                "signal": name,
                "source_definition": signal_source(name, args, ctx),
                "pearson_sa_class_z": pearson_corr(values, sa_class_z),
                "spearman_sa_class_z": spearman_corr(values, sa_class_z),
                "pearson_sa_raw": pearson_corr(values, sa_raw),
                "spearman_sa_raw": spearman_corr(values, sa_raw),
                "signal_mean": float(np.mean(values)),
                "signal_std": float(np.std(values)),
                "n": int(values.size),
            }
        )
    return rows


def keep_tag(keep_items: frozenset[str]) -> str:
    ordered = [
        name
        for name in (*trial.COMPONENT_NAMES, "noise_gate")
        if name in keep_items
    ]
    return "keep_none" if not ordered else "keep_" + "-".join(ordered)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No correlation rows to save.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_table(
    rows: list[dict[str, Any]],
    *,
    ctx: trial.ExperimentContext,
    a_cache_path: Path,
    a_match_pearson: float,
    a_match_spearman: float,
) -> None:
    print()
    print(
        f"[{ctx.experiment}/{ctx.dataset}] dynamic signals vs static SA "
        f"(seed={trial.SEED})"
    )
    print(f"A subcomponent cache: {a_cache_path}")
    print(
        "A recomputation check: "
        f"Pearson={a_match_pearson:.6f}, Spearman={a_match_spearman:.6f}"
    )
    print(
        f"{'signal':<16}{'source':<20}"
        f"{'Pearson(class-z)':>20}{'Spearman(class-z)':>21}"
        f"{'Pearson(raw)':>16}{'Spearman(raw)':>17}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['signal']:<16}"
            f"{row['source_definition']:<20}"
            f"{row['pearson_sa_class_z']:>20.6f}"
            f"{row['spearman_sa_class_z']:>21.6f}"
            f"{row['pearson_sa_raw']:>16.6f}"
            f"{row['spearman_sa_raw']:>17.6f}"
        )


def run_task(
    experiment: trial.ExperimentName,
    dataset: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx = trial.build_context(experiment, dataset)
    device = (
        torch.device(args.device)
        if args.device
        else trial.CONFIG.global_device
    )

    components, gate_data, _ = trial.load_or_compute_dynamic_supervision(ctx, args)
    a_subcomponents, a_strategy_name, a_epochs, a_cache_path = (
        load_or_compute_a_subcomponents(ctx, args)
    )

    static = trial.build_static_bundle(ctx, args, device)
    sa_raw = np.asarray(static.scores["sa"], dtype=np.float64)
    sa_class_z = np.asarray(
        trial.standard_zscore_by_class(static.scores["sa"], static.labels),
        dtype=np.float64,
    )

    signals = {
        "A_boundary": a_subcomponents["A_boundary"],
        "A_gain": a_subcomponents["A_gain"],
        "A_stability": a_subcomponents["A_stability"],
        "A": np.asarray(components["A"].final_normalized, dtype=np.float64),
        "C": np.asarray(components["C"].final_normalized, dtype=np.float64),
        "T": np.asarray(components["T"].final_normalized, dtype=np.float64),
        "noise_gate": np.asarray(gate_data["gate"], dtype=np.float64),
    }

    rows = make_rows(
        ctx=ctx,
        signals=signals,
        sa_raw=sa_raw,
        sa_class_z=sa_class_z,
        args=args,
    )
    a_match_pearson = pearson_corr(
        a_subcomponents["A_recomputed"], signals["A"]
    )
    a_match_spearman = spearman_corr(
        a_subcomponents["A_recomputed"], signals["A"]
    )
    print_table(
        rows,
        ctx=ctx,
        a_cache_path=a_cache_path,
        a_match_pearson=a_match_pearson,
        a_match_spearman=a_match_spearman,
    )

    metadata = {
        "experiment": experiment,
        "dataset": dataset,
        "seed": int(trial.SEED),
        "keep_original_metrics": sorted(args.keep_items),
        "a_subcomponent_strategy": a_strategy_name,
        "a_subcomponent_epochs": int(a_epochs),
        "a_subcomponent_cache": str(a_cache_path),
        "a_recomputed_vs_cached_pearson": a_match_pearson,
        "a_recomputed_vs_cached_spearman": a_match_spearman,
        "primary_sa_target": "class-wise standardized SA used by weight regression",
    }

    # Release CLIP/adapters before the next task.
    del static, components, gate_data, a_subcomponents, signals
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, metadata


def main() -> None:
    args = parse_args()
    validate_args(args)

    datasets = trial.selected_datasets(args.dataset)
    experiments = trial.selected_experiments(args.experiment)
    tasks = [(experiment, dataset) for dataset in datasets for experiment in experiments]

    print("=" * 110)
    print("checkSA: dynamic signals vs static Semantic Alignment")
    print(f"datasets={datasets}")
    print(f"experiments={experiments}")
    print(f"keep original metrics={sorted(args.keep_items)}")
    print("Primary correlation target: class-wise standardized SA used in regression.")
    print("No weights or masks will be computed.")
    print("=" * 110)

    all_rows: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    for experiment, dataset in tqdm(tasks, desc="checkSA tasks", unit="task"):
        rows, metadata = run_task(experiment, dataset, args)
        all_rows.extend(rows)
        all_metadata.append(metadata)

        out_dir = (
            args.output_root
            / experiment
            / dataset
            / keep_tag(args.keep_items)
        )
        write_csv(out_dir / "correlations.csv", rows)
        (out_dir / "correlations.json").write_text(
            json.dumps(
                json_safe({"metadata": metadata, "results": rows}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[saved] {out_dir / 'correlations.csv'}")
        print(f"[saved] {out_dir / 'correlations.json'}")

    combined_dir = args.output_root / "combined" / keep_tag(args.keep_items)
    write_csv(combined_dir / "correlations.csv", all_rows)
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "correlations.json").write_text(
        json.dumps(
            json_safe({"metadata": all_metadata, "results": all_rows}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print(f"Combined CSV: {combined_dir / 'correlations.csv'}")
    print(f"Combined JSON: {combined_dir / 'correlations.json'}")
    print(f"A-subcomponent caches: {YYY_ROOT / 'a_subcomponents_cache'}")
    print("=" * 110)


if __name__ == "__main__":
    main()