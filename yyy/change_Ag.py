#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the revised A component and experiment-specific group selection.

This script supersedes ``yyy/change_Ag.py`` and ``yyy/change_Ag_1.py`` without
modifying the project's standard group implementation.

Dynamic target
--------------
A uses the two-end support weighted gain

    G_i = sqrt(e_i * m_i) * (m_i - e_i),

where e_i and m_i are the early- and middle-window mean true-class
probabilities. Its stability subcomponent is replaced with

    S_i = Var_late(loss_i) - Var_middle(loss_i).

Boundary, gain and stability are standardized separately inside each fold,
then summed and standardized exactly as in the original A pipeline.

The noise gate is completely disabled: it is neither loaded nor computed, and
no dynamic component is gated. The regression target is simply

    y_i = (A_i + C_i + T_i) / 3.

Group masks
-----------
- label_noise: use the project's mature standard ``select_group_mask`` with
  its explicit weighted distribution-correction term.
- corruption: do not use that explicit distribution term. SA, dynamic Div and
  DDS form a weighted candidate pool. Inside the pool, enumerate candidate
  pairs and sample one pair with softmax probability determined by how much
  adding the pair reduces the distance between the selected-subset class
  center and the full-class center. If one slot remains, apply the same rule to
  individual candidates.

Masks and learned weights are printed but not saved. For corruption masks, the
script prints how many samples of each corruption operation are retained.

Default settings
----------------
- datasets: CIFAR-100 and Tiny-ImageNet;
- experiments: normal, label_noise and corruption;
- seed: 22, mask keep ratio: 50;
- fixed dynamic windows: CIFAR-100 160 epochs, Tiny-ImageNet 80 epochs;
- regression tolerance: 1e-6;
- corruption candidate-pool size:
  min(remaining, max(10, ceil(sqrt(remaining))));
- corruption group initialization: 2 uniformly sampled samples per class.

Examples
--------

    python yyy/change_Ag_2.py

    python yyy/change_Ag_2.py --dataset cifar100 --experiment corruption

    python yyy/change_Ag_2.py --keep A,C,T

    python yyy/change_Ag_2.py --force-new-a --force-dynamic
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import sys
from itertools import combinations
from math import ceil, sqrt
from pathlib import Path
from typing import Any

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
    import checkSA as check
    import try_2 as trial
except ImportError as exc:
    raise ImportError(
        "yyy/change_Ag_2.py requires yyy/checkSA.py and yyy/try_2.py."
    ) from exc

from utils.score_utils import standard_zscore  # noqa: E402
from weights.dynamic_utils import (  # noqa: E402
    DynamicComponentResult,
    safe_standardize,
    standard_zscore_dynamic,
)


NEW_A_CACHE_VERSION = "balanced_gain_loss_var_delta_no_gate_v2"
NEW_GAIN_NAME = "A_gain_balanced"
NEW_STABILITY_NAME = "A_stability_loss_var_delta"
NEW_A_NAME = "A"
SIGNAL_NAMES = (
    "A_boundary",
    NEW_GAIN_NAME,
    NEW_STABILITY_NAME,
    NEW_A_NAME,
    "C",
    "T",
)
DYNAMIC_COMPONENTS_USED = ("C", "T")
MASK_KEEP_RATIO = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate balanced-gain/loss-variance A without noise gating, "
            "then compare standard label-noise group with corruption-specific "
            "candidate-pair distribution repair."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *trial.SUPPORTED_DATASETS),
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=("all", "special", *trial.EXPERIMENTS),
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

    parser.add_argument("--ratio-lambda", type=float, default=1e-3)
    parser.add_argument("--regression-learning-rate", type=float, default=4e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument(
        "--regression-tol",
        type=float,
        default=1e-6,
        help="Regression convergence tolerance. Default: 1e-6.",
    )

    parser.add_argument(
        "--keep",
        default="",
        help=(
            "Comma-separated full-run definitions among A,C,T. "
            "A means computing the revised A from 200/90 epochs. "
            "noise_gate is intentionally unsupported."
        ),
    )
    parser.add_argument(
        "--force-dynamic",
        action="store_true",
        help="Force recomputation of non-kept fixed-window C/T caches.",
    )
    parser.add_argument(
        "--force-new-a",
        action="store_true",
        help="Ignore and recompute the revised-A cache.",
    )
    parser.add_argument(
        "--epoch-chunk-size",
        type=int,
        default=4,
        help="Epochs processed together when deriving probabilities/losses.",
    )

    # Standard label-noise group settings. Defaults are resolved by try_2:
    # candidate pool 1, initialization 2, and the standard risk factor.
    parser.add_argument(
        "--group-candidate-pool-size",
        type=int,
        default=None,
        help="Override the standard label-noise group candidate-pool size.",
    )
    parser.add_argument(
        "--group-init-count",
        type=int,
        default=None,
        help="Override the standard label-noise group initialization count.",
    )

    # Corruption-only experimental group settings.
    parser.add_argument(
        "--corruption-candidate-pool-min",
        type=int,
        default=10,
        help=(
            "Corruption candidate-pool lower bound. Actual size is "
            "min(remaining, max(value, ceil(sqrt(remaining))))."
        ),
    )
    parser.add_argument(
        "--corruption-group-init-count",
        type=int,
        default=2,
        help="Uniform random initialization count per class for corruption group.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=YYY_ROOT / "change_Ag_2_results",
        help="Output directory for correlation CSV/JSON only.",
    )
    return parser.parse_args()


def parse_keep_items(text: str) -> frozenset[str]:
    aliases = {
        "a": "A",
        "c": "C",
        "t": "T",
    }
    items: set[str] = set()
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        normalized = token.lower().replace("-", "_")
        if normalized in {"noise_gate", "gate", "noisegate"}:
            raise ValueError(
                "noise_gate has been abandoned in change_Ag_2.py and cannot "
                "be used in --keep. Supported values: A,C,T."
            )
        if normalized not in aliases:
            raise ValueError(
                f"Unsupported --keep item: {token!r}. Supported values: A,C,T."
            )
        items.add(aliases[normalized])
    return frozenset(items)


def validate_args(args: argparse.Namespace) -> None:
    args.keep_items = parse_keep_items(args.keep)
    if args.epoch_chunk_size <= 0:
        raise ValueError("--epoch-chunk-size must be positive.")
    if args.ratio_lambda < 0:
        raise ValueError("--ratio-lambda must be non-negative.")
    if args.regression_learning_rate <= 0:
        raise ValueError("--regression-learning-rate must be positive.")
    if args.regression_max_iter <= 0:
        raise ValueError("--regression-max-iter must be positive.")
    if args.regression_tol <= 0:
        raise ValueError("--regression-tol must be positive.")
    if args.group_candidate_pool_size is not None and args.group_candidate_pool_size <= 0:
        raise ValueError("--group-candidate-pool-size must be positive.")
    if args.group_init_count is not None and args.group_init_count < 0:
        raise ValueError("--group-init-count must be non-negative.")
    if args.corruption_candidate_pool_min <= 0:
        raise ValueError("--corruption-candidate-pool-min must be positive.")
    if args.corruption_group_init_count <= 0:
        raise ValueError("--corruption-group-init-count must be positive.")


def new_a_cache_path(
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
) -> Path:
    return (
        YYY_ROOT
        / "a_balanced_gain_loss_stability_cache_v2"
        / strategy
        / ctx.experiment
        / ctx.dataset
        / str(int(epochs))
        / "A_balanced_gain_loss_var_delta.npz"
    )


def _cache_special_signature(ctx: trial.ExperimentContext) -> str:
    return str(check.special_signature(ctx))


def load_new_a_cache(
    path: Path,
    *,
    ctx: trial.ExperimentContext,
    strategy: str,
    epochs: int,
    source_log_dir: Path,
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
                NEW_GAIN_NAME,
                NEW_STABILITY_NAME,
                "A_new",
            }
            if not required.issubset(set(data.files)):
                return None
            checks = (
                (str(np.asarray(data["cache_version"]).item()), NEW_A_CACHE_VERSION),
                (str(np.asarray(data["experiment"]).item()), ctx.experiment),
                (str(np.asarray(data["dataset"]).item()), ctx.dataset),
                (int(np.asarray(data["seed"]).item()), int(trial.SEED)),
                (str(np.asarray(data["strategy"]).item()), strategy),
                (int(np.asarray(data["epochs"]).item()), int(epochs)),
                (
                    str(np.asarray(data["source_log_dir"]).item()),
                    str(source_log_dir.resolve()),
                ),
                (
                    str(np.asarray(data["special_signature"]).item()),
                    _cache_special_signature(ctx),
                ),
            )
            if any(actual != expected for actual, expected in checks):
                return None
            if not np.array_equal(
                np.asarray(data["labels"], dtype=np.int64), ctx.labels
            ):
                return None
            if not np.array_equal(
                np.asarray(data["early_idx"], dtype=np.int64), early
            ):
                return None
            if not np.array_equal(
                np.asarray(data["middle_idx"], dtype=np.int64), middle
            ):
                return None
            if not np.array_equal(
                np.asarray(data["late_idx"], dtype=np.int64), late
            ):
                return None

            result = {
                "A_boundary": np.asarray(data["A_boundary"], dtype=np.float64),
                NEW_GAIN_NAME: np.asarray(data[NEW_GAIN_NAME], dtype=np.float64),
                NEW_STABILITY_NAME: np.asarray(
                    data[NEW_STABILITY_NAME], dtype=np.float64
                ),
                "A_new": np.asarray(data["A_new"], dtype=np.float64),
            }
    except Exception:
        return None

    for values in result.values():
        if values.shape != (ctx.labels.size,) or not np.all(np.isfinite(values)):
            return None
    return result


def save_new_a_cache(
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
        cache_version=np.array(NEW_A_CACHE_VERSION, dtype=np.str_),
        experiment=np.array(ctx.experiment, dtype=np.str_),
        dataset=np.array(ctx.dataset, dtype=np.str_),
        seed=np.array(int(trial.SEED), dtype=np.int64),
        strategy=np.array(strategy, dtype=np.str_),
        epochs=np.array(int(epochs), dtype=np.int64),
        source_log_dir=np.array(str(source_log_dir.resolve()), dtype=np.str_),
        special_signature=np.array(_cache_special_signature(ctx), dtype=np.str_),
        labels=np.asarray(ctx.labels, dtype=np.int64),
        early_idx=np.asarray(early, dtype=np.int64),
        middle_idx=np.asarray(middle, dtype=np.int64),
        late_idx=np.asarray(late, dtype=np.int64),
        A_boundary=np.asarray(result["A_boundary"], dtype=np.float32),
        A_gain_balanced=np.asarray(result[NEW_GAIN_NAME], dtype=np.float32),
        A_stability_loss_var_delta=np.asarray(
            result[NEW_STABILITY_NAME], dtype=np.float32
        ),
        A_new=np.asarray(result["A_new"], dtype=np.float32),
    )


def _validate_window(indices: np.ndarray, name: str) -> tuple[int, int]:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} window must be a non-empty vector.")
    expected = np.arange(values[0], values[-1] + 1, dtype=np.int64)
    if not np.array_equal(values, expected):
        raise ValueError(f"{name} window must be contiguous.")
    return int(values[0]), int(values[-1] + 1)


def _true_class_loss_variance(
    logits: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    epoch_chunk_size: int,
) -> np.ndarray:
    """Return per-sample temporal variance of true-class cross-entropy loss."""
    start, stop = _validate_window(indices, "loss")
    labels = np.asarray(labels, dtype=np.int64)
    num_samples = int(labels.size)
    label_positions = labels.reshape(1, -1, 1)
    sum_loss = np.zeros(num_samples, dtype=np.float64)
    sum_loss2 = np.zeros(num_samples, dtype=np.float64)
    count = 0

    for chunk_start in range(start, stop, int(epoch_chunk_size)):
        chunk_stop = min(stop, chunk_start + int(epoch_chunk_size))
        chunk = np.asarray(logits[chunk_start:chunk_stop], dtype=np.float64)
        if chunk.ndim != 3 or chunk.shape[1] != num_samples:
            raise ValueError("Unexpected train-logit chunk shape.")
        safe = np.nan_to_num(chunk, nan=0.0, posinf=50.0, neginf=-50.0)
        max_logits = np.max(safe, axis=2, keepdims=True)
        shifted = np.clip(safe - max_logits, -50.0, 50.0)
        logsumexp = (
            np.log(np.sum(np.exp(shifted), axis=2))
            + max_logits.squeeze(2)
        )
        true_logits = np.take_along_axis(
            safe, label_positions, axis=2
        ).squeeze(2)
        losses = np.nan_to_num(
            logsumexp - true_logits,
            nan=0.0,
            posinf=50.0,
            neginf=0.0,
        )
        sum_loss += np.sum(losses, axis=0, dtype=np.float64)
        sum_loss2 += np.sum(losses * losses, axis=0, dtype=np.float64)
        count += int(losses.shape[0])
        del chunk, safe, max_logits, shifted, logsumexp, true_logits, losses

    if count <= 0:
        raise ValueError("Loss window produced no observations.")
    mean = sum_loss / float(count)
    return np.maximum(sum_loss2 / float(count) - mean * mean, 0.0)


def compute_new_a(
    *,
    ctx: trial.ExperimentContext,
    source_log_dir: Path,
    epochs: int,
    windows: tuple[np.ndarray, np.ndarray, np.ndarray],
    epoch_chunk_size: int,
) -> dict[str, np.ndarray]:
    """Compute balanced gain, loss-variance stability and revised A."""
    fold_paths = sorted(
        source_log_dir.glob("fold_*.npz"), key=trial.fold_sort_key
    )
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {source_log_dir}")

    num_samples = int(ctx.labels.size)
    names = ("A_boundary", NEW_GAIN_NAME, NEW_STABILITY_NAME, "A_new")
    sums = {
        name: np.zeros(num_samples, dtype=np.float64)
        for name in names
    }
    counts = np.zeros(num_samples, dtype=np.int64)
    early, middle, late = windows

    for path in tqdm(
        fold_paths,
        desc=f"revised A {ctx.experiment}/{ctx.dataset}",
        unit="fold",
    ):
        with np.load(path, allow_pickle=False) as data:
            required = {"train_indices", "train_logits"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} missing keys: {sorted(missing)}")
            train_idx = np.asarray(data["train_indices"], dtype=np.int64)
            train_logits = data["train_logits"]
            if train_logits.ndim != 3:
                raise ValueError(f"train_logits in {path} must be 3D.")
            if int(train_logits.shape[0]) < int(epochs):
                raise ValueError(
                    f"{path} has {train_logits.shape[0]} epochs; "
                    f"{epochs} are required."
                )
            if int(train_logits.shape[1]) != int(train_idx.size):
                raise ValueError(f"train-logit sample dimension mismatch: {path}")
            if np.any(train_idx < 0) or np.any(train_idx >= num_samples):
                raise ValueError(f"train_indices out of range: {path}")

            labels_train = np.asarray(ctx.labels[train_idx], dtype=np.int64)
            early_mean, _, boundary = check._true_probability_moments(
                train_logits,
                labels_train,
                early,
                epoch_chunk_size=int(epoch_chunk_size),
                include_boundary=True,
            )
            middle_mean, _, _ = check._true_probability_moments(
                train_logits,
                labels_train,
                middle,
                epoch_chunk_size=int(epoch_chunk_size),
                include_boundary=False,
            )
            middle_loss_var = _true_class_loss_variance(
                train_logits,
                labels_train,
                middle,
                epoch_chunk_size=int(epoch_chunk_size),
            )
            late_loss_var = _true_class_loss_variance(
                train_logits,
                labels_train,
                late,
                epoch_chunk_size=int(epoch_chunk_size),
            )

        if boundary is None:
            raise RuntimeError("Boundary information was not computed.")

        support = np.sqrt(
            np.clip(
                np.asarray(early_mean, dtype=np.float64)
                * np.asarray(middle_mean, dtype=np.float64),
                0.0,
                1.0,
            )
        )
        balanced_gain = support * (
            np.asarray(middle_mean, dtype=np.float64)
            - np.asarray(early_mean, dtype=np.float64)
        )
        # Exact requested definition: late loss variance minus middle loss
        # variance, with no sign reversal before standardization.
        loss_var_delta = (
            np.asarray(late_loss_var, dtype=np.float64)
            - np.asarray(middle_loss_var, dtype=np.float64)
        )

        boundary_z = safe_standardize(boundary)
        gain_z = safe_standardize(balanced_gain)
        stability_z = safe_standardize(loss_var_delta)
        a_fold = standard_zscore_dynamic(boundary_z + gain_z + stability_z)

        fold_values = {
            "A_boundary": boundary_z,
            NEW_GAIN_NAME: gain_z,
            NEW_STABILITY_NAME: stability_z,
            "A_new": a_fold,
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
            middle_loss_var,
            late_loss_var,
            boundary,
            support,
            balanced_gain,
            loss_var_delta,
            boundary_z,
            gain_z,
            stability_z,
            a_fold,
        )
        gc.collect()

    if np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0)
        raise ValueError(
            "Some samples never appeared in training folds: "
            f"{missing[:10].tolist()}"
        )

    result: dict[str, np.ndarray] = {}
    for name, total in sums.items():
        aggregated = total / counts
        normalized = standard_zscore_dynamic(aggregated.astype(np.float32))
        if normalized.shape != (num_samples,) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"Invalid computed signal: {name}")
        result[name] = np.asarray(normalized, dtype=np.float64)
    return result


def load_or_compute_new_a(
    ctx: trial.ExperimentContext,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], str, int, Path]:
    strategy, epochs = check.a_strategy(ctx, args.keep_items)
    windows = check.resolve_a_windows(ctx, strategy, epochs)
    source_log_dir = trial.find_proxy_log_dir(
        ctx.experiment,
        ctx.dataset,
        args.proxy_model,
        epochs,
    )
    cache_path = new_a_cache_path(ctx, strategy, epochs)

    result = None
    if not args.force_new_a:
        result = load_new_a_cache(
            cache_path,
            ctx=ctx,
            strategy=strategy,
            epochs=epochs,
            source_log_dir=source_log_dir,
            windows=windows,
        )
    if result is not None:
        tqdm.write(f"[revised-A] cache HIT: {cache_path}")
        return result, strategy, epochs, cache_path

    tqdm.write(f"[revised-A] cache MISS: {cache_path}")
    result = compute_new_a(
        ctx=ctx,
        source_log_dir=source_log_dir,
        epochs=epochs,
        windows=windows,
        epoch_chunk_size=int(args.epoch_chunk_size),
    )
    save_new_a_cache(
        cache_path,
        ctx=ctx,
        strategy=strategy,
        epochs=epochs,
        source_log_dir=source_log_dir,
        windows=windows,
        result=result,
    )
    tqdm.write(f"[revised-A] saved: {cache_path}")
    return result, strategy, epochs, cache_path


def load_or_compute_ct_components(
    ctx: trial.ExperimentContext,
    args: argparse.Namespace,
) -> dict[str, DynamicComponentResult]:
    """Load/compute C and T only; noise_gate and original A are never touched."""
    cutoff = int(trial.DYNAMIC_EPOCHS[ctx.dataset])
    kept = frozenset(args.keep_items)
    source_log_dir: Path | None = None
    missing: list[str] = []
    results: dict[str, DynamicComponentResult] = {}

    for name in DYNAMIC_COMPONENTS_USED:
        if name in kept:
            results[name] = trial.load_original_component_cache(
                ctx, args.proxy_model, name
            )
            continue
        if source_log_dir is None:
            source_log_dir = trial.find_proxy_log_dir(
                ctx.experiment, ctx.dataset, args.proxy_model, cutoff
            )
        cache_path = trial.component_cache_path(ctx, cutoff, name)
        result = None
        if not args.force_dynamic:
            result = trial.load_component_cache(
                cache_path,
                ctx=ctx,
                cutoff=cutoff,
                source_log_dir=source_log_dir,
            )
        if result is None:
            missing.append(name)
            tqdm.write(f"[dynamic] cache MISS: {cache_path}")
        else:
            results[name] = result
            tqdm.write(f"[dynamic] cache HIT: {cache_path}")

    if missing:
        if source_log_dir is None:
            raise RuntimeError("Missing C/T components without a proxy-log path.")
        patch_context = (
            trial.noise_mod.patched_training_label_noise(
                ctx.dataset, trial.SEED, verbose_once=False
            )
            if ctx.experiment == "label_noise"
            else contextlib.nullcontext()
        )
        with patch_context:
            folds = trial.load_truncated_cv_logs(
                source_log_dir,
                labels_all=ctx.labels,
                cutoff=cutoff,
            )
        computers = trial.dynamic_component_computers()
        with trial.patched_fixed_epoch_windows(ctx.dataset):
            for name in tqdm(
                missing,
                desc=f"{ctx.experiment}/{ctx.dataset} C/T",
                unit="component",
            ):
                result = computers[name].compute(
                    folds=folds, labels_all=ctx.labels
                )
                results[name] = result
                cache_path = trial.component_cache_path(ctx, cutoff, name)
                trial.save_component_cache(
                    cache_path,
                    ctx=ctx,
                    cutoff=cutoff,
                    source_log_dir=source_log_dir,
                    component_name=name,
                    result=result,
                )
                tqdm.write(f"[dynamic] saved: {cache_path}")
        del folds
        gc.collect()

    if set(results) != set(DYNAMIC_COMPONENTS_USED):
        raise RuntimeError(f"Incomplete C/T components: {sorted(results)}")
    return results


def build_ungated_target(
    new_a: np.ndarray,
    components: dict[str, DynamicComponentResult],
) -> np.ndarray:
    values = np.stack(
        [
            np.asarray(new_a, dtype=np.float64),
            np.asarray(components["C"].final_normalized, dtype=np.float64),
            np.asarray(components["T"].final_normalized, dtype=np.float64),
        ],
        axis=0,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("A/C/T contain non-finite values.")
    return np.mean(values, axis=0, dtype=np.float64)


def signal_source(name: str, args: argparse.Namespace) -> str:
    if name in {"A_boundary", NEW_GAIN_NAME, NEW_STABILITY_NAME, NEW_A_NAME}:
        return (
            "revised_full_run"
            if "A" in args.keep_items
            else "revised_fixed_windows"
        )
    if name in {"C", "T"}:
        return "original_full_run" if name in args.keep_items else "fixed_windows"
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
    for name in SIGNAL_NAMES:
        values = np.asarray(signals[name], dtype=np.float64)
        if values.shape != sa_raw.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid signal {name}: shape={values.shape}")
        rows.append(
            {
                "experiment": ctx.experiment,
                "dataset": ctx.dataset,
                "seed": int(trial.SEED),
                "signal": name,
                "source_definition": signal_source(name, args),
                "pearson_sa_class_z": check.pearson_corr(values, sa_class_z),
                "spearman_sa_class_z": check.spearman_corr(values, sa_class_z),
                "pearson_sa_raw": check.pearson_corr(values, sa_raw),
                "spearman_sa_raw": check.spearman_corr(values, sa_raw),
                "signal_mean": float(np.mean(values)),
                "signal_std": float(np.std(values)),
                "n": int(values.size),
            }
        )
    return rows


def print_correlation_table(
    rows: list[dict[str, Any]],
    *,
    ctx: trial.ExperimentContext,
    cache_path: Path,
) -> None:
    print()
    print(
        f"[{ctx.experiment}/{ctx.dataset}] revised ungated dynamic signals "
        f"vs static SA (seed={trial.SEED})"
    )
    print(f"revised A cache: {cache_path}")
    print(
        f"{'signal':<30}{'source':<26}"
        f"{'Pearson(class-z)':>20}{'Spearman(class-z)':>21}"
        f"{'Pearson(raw)':>16}{'Spearman(raw)':>17}"
    )
    print("-" * 130)
    for row in rows:
        print(
            f"{row['signal']:<30}"
            f"{row['source_definition']:<26}"
            f"{row['pearson_sa_class_z']:>20.6f}"
            f"{row['spearman_sa_class_z']:>21.6f}"
            f"{row['pearson_sa_raw']:>16.6f}"
            f"{row['spearman_sa_raw']:>17.6f}"
        )


def _allocate_class_budgets(
    labels: np.ndarray,
    num_classes: int,
    keep_ratio: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    num_samples = int(labels.size)
    target = min(
        num_samples,
        max(1, int(round(num_samples * float(keep_ratio) / 100.0))),
    )
    sizes = np.asarray(
        [np.sum(labels == class_id) for class_id in range(num_classes)],
        dtype=np.int64,
    )
    raw = sizes.astype(np.float64) * float(keep_ratio) / 100.0
    budgets = np.minimum(np.floor(raw).astype(np.int64), sizes)
    need = int(target - budgets.sum())
    if need > 0:
        fractions = raw - budgets.astype(np.float64)
        order = np.lexsort(
            (np.arange(num_classes, dtype=np.int64), -fractions)
        )
        for class_id in order:
            if need <= 0:
                break
            if budgets[class_id] >= sizes[class_id]:
                continue
            budgets[class_id] += 1
            need -= 1
    if need != 0:
        raise RuntimeError("Failed to allocate class budgets.")
    return budgets


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("softmax input must not be empty.")
    finite = np.isfinite(values)
    if not np.all(finite):
        fill = float(np.min(values[finite])) if np.any(finite) else 0.0
        values = np.where(finite, values, fill)
    std = float(values.std())
    normalized = (
        (values - float(values.mean())) / std
        if std > 1e-12
        else np.zeros_like(values)
    )
    normalized -= float(np.max(normalized))
    exp_values = np.exp(normalized)
    total = float(exp_values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(values.size, 1.0 / values.size, dtype=np.float64)
    return exp_values / total


def _corruption_pool_size(remaining: int, minimum: int) -> int:
    if remaining <= 0:
        return 0
    return min(
        int(remaining),
        max(int(minimum), int(ceil(sqrt(float(remaining))))),
    )


def _choose_repair_pair_or_single(
    *,
    pool_indices: np.ndarray,
    pool_features: np.ndarray,
    current_sum: np.ndarray,
    current_count: int,
    full_mean: np.ndarray,
    slots_to_fill: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    pool_features = np.asarray(pool_features, dtype=np.float32)
    current_sum = np.asarray(current_sum, dtype=np.float32)
    full_mean = np.asarray(full_mean, dtype=np.float32)
    if pool_indices.size == 0 or current_count <= 0:
        raise ValueError("Invalid corruption candidate-pool state.")

    old_distance = float(
        np.linalg.norm(current_sum / float(current_count) - full_mean)
    )
    if slots_to_fill == 1 or pool_indices.size == 1:
        new_means = (
            current_sum[None, :] + pool_features
        ) / float(current_count + 1)
        new_distances = np.linalg.norm(
            new_means - full_mean[None, :], axis=1
        )
        repair = old_distance - new_distances
        position = int(
            rng.choice(
                np.arange(pool_indices.size, dtype=np.int64),
                p=_stable_softmax(repair),
            )
        )
        return pool_indices[[position]], {
            "repair": float(repair[position]),
            "new_distance": float(new_distances[position]),
            "option_count": float(pool_indices.size),
        }

    pair_positions = np.asarray(
        list(combinations(range(pool_indices.size), 2)),
        dtype=np.int64,
    )
    pair_features = (
        pool_features[pair_positions[:, 0]]
        + pool_features[pair_positions[:, 1]]
    )
    new_means = (
        current_sum[None, :] + pair_features
    ) / float(current_count + 2)
    new_distances = np.linalg.norm(
        new_means - full_mean[None, :], axis=1
    )
    repair = old_distance - new_distances
    pair_id = int(
        rng.choice(
            np.arange(pair_positions.shape[0], dtype=np.int64),
            p=_stable_softmax(repair),
        )
    )
    chosen_positions = pair_positions[pair_id]
    return pool_indices[chosen_positions].astype(np.int64), {
        "repair": float(repair[pair_id]),
        "new_distance": float(new_distances[pair_id]),
        "option_count": float(pair_positions.shape[0]),
    }


def compute_corruption_repair_group_mask(
    *,
    ctx: trial.ExperimentContext,
    static: trial.StaticBundle,
    weights: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    """Corruption-only group: weighted candidate pool, repair-probability pair."""
    if ctx.experiment != "corruption":
        raise ValueError("This selector is restricted to corruption experiments.")

    labels = np.asarray(static.labels, dtype=np.int64)
    sa_raw = np.asarray(static.scores["sa"], dtype=np.float32)
    dds_raw = np.asarray(static.scores["dds"], dtype=np.float32)
    num_classes = len(static.class_names)
    num_samples = int(labels.size)
    budgets = _allocate_class_budgets(
        labels, num_classes, MASK_KEEP_RATIO
    )
    class_indices_list = [
        np.flatnonzero(labels == class_id).astype(np.int64)
        for class_id in range(num_classes)
    ]

    tqdm.write("[corruption-group] encoding Div features once...")
    div_features_t, encoded_labels = static.div_metric._encode_images(
        static.div_loader, static.image_adapter
    )
    div_features = (
        div_features_t.detach().cpu().numpy()
        if isinstance(div_features_t, torch.Tensor)
        else np.asarray(div_features_t)
    ).astype(np.float32)
    if div_features.shape[0] != num_samples:
        raise RuntimeError("Div feature count does not match corruption dataset.")
    if encoded_labels is not None:
        encoded_labels_np = (
            encoded_labels.detach().cpu().numpy()
            if isinstance(encoded_labels, torch.Tensor)
            else np.asarray(encoded_labels)
        ).astype(np.int64)
        if encoded_labels_np.shape == labels.shape and not np.array_equal(
            encoded_labels_np, labels
        ):
            raise RuntimeError("Div feature labels do not match corruption labels.")

    with trial.patched_group_mean_cache(ctx):
        mean_path = trial.mask_mod._mean_stats_cache_path(
            dataset_name=ctx.dataset,
            clip_model=args.clip_model,
            adapter_image_path=str(static.adapter_image_path),
        )
        full_class_mean, _ = trial.mask_mod._get_or_compute_group_mean_stats(
            cache_path=mean_path,
            image_features=div_features,
            labels=labels,
            num_classes=num_classes,
        )
    full_class_mean = np.asarray(full_class_mean, dtype=np.float32)

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    class_counts = np.zeros(num_classes, dtype=np.int64)
    class_sums = np.zeros(
        (num_classes, div_features.shape[1]), dtype=np.float32
    )
    class_rngs: list[np.random.Generator] = []
    requested_init = max(1, int(args.corruption_group_init_count))

    for class_id, class_indices in enumerate(class_indices_list):
        rng = np.random.default_rng(
            int(trial.SEED)
            + 9_176 * int(MASK_KEEP_RATIO)
            + 104_729 * int(class_id)
        )
        class_rngs.append(rng)
        budget = int(budgets[class_id])
        if budget <= 0 or class_indices.size == 0:
            continue
        init_count = min(requested_init, budget, int(class_indices.size))
        initial = rng.choice(
            class_indices, size=init_count, replace=False
        ).astype(np.int64)
        selected_mask[initial] = 1
        class_counts[class_id] = init_count
        class_sums[class_id] = np.sum(
            div_features[initial], axis=0, dtype=np.float32
        )

    target_size = int(budgets.sum())
    pbar = tqdm(
        total=target_size,
        initial=int(selected_mask.sum()),
        desc="[corruption-group] candidate-pair repair",
        unit="sample",
    )
    repair_history: list[float] = []
    pool_history: list[int] = []
    option_history: list[float] = []
    pair_steps = 0
    single_steps = 0

    for class_id, class_indices in enumerate(class_indices_list):
        budget = int(budgets[class_id])
        rng = class_rngs[class_id]
        while class_counts[class_id] < budget:
            remaining_slots = int(budget - class_counts[class_id])
            unselected = class_indices[selected_mask[class_indices] == 0]
            reference = class_indices[selected_mask[class_indices] > 0]
            current_count = int(class_counts[class_id])
            if unselected.size == 0 or reference.size == 0 or current_count <= 0:
                raise RuntimeError(
                    f"Invalid corruption group state for class={class_id}."
                )

            dynamic_k = max(3, int(ceil(0.05 * current_count)))
            dynamic_div = (
                static.div_metric._knn_mean_distance_to_reference(
                    query_features=torch.as_tensor(
                        div_features[unselected],
                        dtype=torch.float32,
                        device=device,
                    ),
                    reference_features=torch.as_tensor(
                        div_features[reference],
                        dtype=torch.float32,
                        device=device,
                    ),
                    k=float(dynamic_k),
                    query_indices=torch.as_tensor(
                        unselected, dtype=torch.long, device=device
                    ),
                    reference_indices=torch.as_tensor(
                        reference, dtype=torch.long, device=device
                    ),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            candidate_scores = (
                float(weights["sa"]) * standard_zscore(sa_raw[unselected])
                + float(weights["dds"]) * standard_zscore(dds_raw[unselected])
                + float(weights["div"]) * standard_zscore(dynamic_div)
            ).astype(np.float32)

            pool_n = _corruption_pool_size(
                int(unselected.size),
                int(args.corruption_candidate_pool_min),
            )
            ranking = np.argsort(-candidate_scores, kind="mergesort")
            pool_indices = unselected[ranking[:pool_n]]
            chosen, repair_stats = _choose_repair_pair_or_single(
                pool_indices=pool_indices,
                pool_features=div_features[pool_indices],
                current_sum=class_sums[class_id],
                current_count=current_count,
                full_mean=full_class_mean[class_id],
                slots_to_fill=min(2, remaining_slots),
                rng=rng,
            )

            selected_mask[chosen] = 1
            class_counts[class_id] += int(chosen.size)
            class_sums[class_id] += np.sum(
                div_features[chosen], axis=0, dtype=np.float32
            )
            repair_history.append(float(repair_stats["repair"]))
            pool_history.append(int(pool_n))
            option_history.append(float(repair_stats["option_count"]))
            pair_steps += int(chosen.size == 2)
            single_steps += int(chosen.size == 1)
            pbar.update(int(chosen.size))
            pbar.set_postfix(
                class_id=int(class_id),
                pool=int(pool_n),
                remain=int(budget - class_counts[class_id]),
            )
    pbar.close()

    if int(selected_mask.sum()) != target_size:
        raise RuntimeError(
            f"Corruption group selected {int(selected_mask.sum())}; "
            f"expected {target_size}."
        )
    for class_id, class_indices in enumerate(class_indices_list):
        if int(selected_mask[class_indices].sum()) != int(budgets[class_id]):
            raise RuntimeError(
                f"Corruption group class budget mismatch: class={class_id}."
            )

    shifts: list[float] = []
    for class_id in range(num_classes):
        if class_counts[class_id] <= 0:
            continue
        subset_mean = class_sums[class_id] / float(class_counts[class_id])
        shifts.append(
            float(np.linalg.norm(subset_mean - full_class_mean[class_id]))
        )

    stats: dict[str, object] = {
        "solver": "corruption_candidate_pool_pair_repair_probability",
        "weights": {key: float(value) for key, value in weights.items()},
        "candidate_pool_rule": (
            "min(remaining,max(minimum,ceil(sqrt(remaining))))"
        ),
        "candidate_pool_min": int(args.corruption_candidate_pool_min),
        "init_count": int(args.corruption_group_init_count),
        "pair_steps": int(pair_steps),
        "single_steps": int(single_steps),
        "mean_selected_repair": (
            float(np.mean(repair_history)) if repair_history else 0.0
        ),
        "mean_candidate_pool_size": (
            float(np.mean(pool_history)) if pool_history else 0.0
        ),
        "mean_option_count": (
            float(np.mean(option_history)) if option_history else 0.0
        ),
        "distribution_shift": (
            float(np.mean(shifts)) if shifts else 0.0
        ),
        "selected_count": int(selected_mask.sum()),
        "class_budgets": {
            int(class_id): int(value)
            for class_id, value in enumerate(budgets.tolist())
        },
    }
    return selected_mask.astype(np.uint8), stats


def print_corruption_operation_retention(
    ctx: trial.ExperimentContext,
    mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    if ctx.experiment != "corruption" or ctx.corruption_info is None:
        raise ValueError("Corruption information is required.")
    selected = np.asarray(mask, dtype=np.uint8).astype(bool)
    info = ctx.corruption_info
    corruption_types = np.asarray(info.corruption_types, dtype=np.int64)
    if corruption_types.shape != selected.shape:
        raise RuntimeError("Corruption-type array does not match mask shape.")

    mapping = trial.corruption_mod.corruption_opt.CORRUPTION_ID_TO_NAME
    results: dict[str, dict[str, float | int]] = {}
    print("\n[corruption-operation retention]")
    print(
        f"{'operation':<28}{'selected':>12}{'total':>12}"
        f"{'retention':>14}{'share_in_mask':>16}"
    )
    print("-" * 82)
    num_selected = int(selected.sum())
    for type_id in sorted(mapping):
        name = str(mapping[type_id])
        type_mask = corruption_types == int(type_id)
        total = int(type_mask.sum())
        kept = int(np.sum(selected & type_mask))
        retention = float(kept / max(1, total))
        share = float(kept / max(1, num_selected))
        results[name] = {
            "selected": kept,
            "total": total,
            "retention": retention,
            "share_in_mask": share,
        }
        print(
            f"{name:<28}{kept:>12d}{total:>12d}"
            f"{retention:>14.6f}{share:>16.6f}"
        )

    corrupted = corruption_types >= 0
    corrupted_kept = int(np.sum(selected & corrupted))
    clean_kept = int(np.sum(selected & ~corrupted))
    print("-" * 82)
    print(
        f"{'all_corrupted':<28}{corrupted_kept:>12d}"
        f"{int(corrupted.sum()):>12d}"
        f"{corrupted_kept / max(1, int(corrupted.sum())):>14.6f}"
        f"{corrupted_kept / max(1, num_selected):>16.6f}"
    )
    print(
        f"{'clean':<28}{clean_kept:>12d}"
        f"{int((~corrupted).sum()):>12d}"
        f"{clean_kept / max(1, int((~corrupted).sum())):>14.6f}"
        f"{clean_kept / max(1, num_selected):>16.6f}"
    )
    return results


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


def keep_tag(keep_items: frozenset[str]) -> str:
    ordered = [name for name in ("A", "C", "T") if name in keep_items]
    return "keep_none" if not ordered else "keep_" + "-".join(ordered)


def run_task(
    experiment: trial.ExperimentName,
    dataset: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trial.set_seed(trial.SEED)
    ctx = trial.build_context(experiment, dataset)
    device = torch.device(args.device) if args.device else trial.CONFIG.global_device

    components = load_or_compute_ct_components(ctx, args)
    new_parts, strategy, epochs, cache_path = load_or_compute_new_a(ctx, args)
    static = trial.build_static_bundle(ctx, args, device)

    sa_raw = np.asarray(static.scores["sa"], dtype=np.float64)
    sa_class_z = np.asarray(
        trial.standard_zscore_by_class(static.scores["sa"], static.labels),
        dtype=np.float64,
    )
    signals = {
        "A_boundary": np.asarray(new_parts["A_boundary"], dtype=np.float64),
        NEW_GAIN_NAME: np.asarray(new_parts[NEW_GAIN_NAME], dtype=np.float64),
        NEW_STABILITY_NAME: np.asarray(
            new_parts[NEW_STABILITY_NAME], dtype=np.float64
        ),
        NEW_A_NAME: np.asarray(new_parts["A_new"], dtype=np.float64),
        "C": np.asarray(components["C"].final_normalized, dtype=np.float64),
        "T": np.asarray(components["T"].final_normalized, dtype=np.float64),
    }
    rows = make_rows(
        ctx=ctx,
        signals=signals,
        sa_raw=sa_raw,
        sa_class_z=sa_class_z,
        args=args,
    )
    print_correlation_table(rows, ctx=ctx, cache_path=cache_path)

    pseudo_label = build_ungated_target(signals[NEW_A_NAME], components)
    print(
        f"[regression] experiment={experiment} dataset={dataset} "
        f"tol={float(args.regression_tol):.1e} | ungated mean(A,C,T)"
    )
    weights, fit = trial.fit_weights(
        ctx=ctx,
        static=static,
        pseudo_label=pseudo_label,
        args=args,
        device=device,
    )

    mask_stats: dict[str, object] | None = None
    corruption_operations: dict[str, dict[str, float | int]] | None = None
    if experiment == "label_noise":
        tqdm.write(
            "[mask] label_noise uses the project's standard group with the "
            "explicit weighted distribution-correction term."
        )
        _, mask_stats = trial.compute_in_memory_learned_group_mask(
            ctx=ctx,
            static=static,
            weights=weights,
            args=args,
            device=device,
        )
    elif experiment == "corruption":
        tqdm.write(
            "[mask] corruption uses candidate-pool scoring without an explicit "
            "distribution term; a pair is sampled by center-repair softmax."
        )
        corruption_mask, mask_stats = compute_corruption_repair_group_mask(
            ctx=ctx,
            static=static,
            weights=weights,
            args=args,
            device=device,
        )
        selected = corruption_mask.astype(bool)
        special = np.asarray(ctx.special_mask, dtype=bool)
        num_selected = int(selected.sum())
        num_corrupted = int(np.sum(selected & special))
        tqdm.write(
            f"[mask] experiment=corruption dataset={ctx.dataset} "
            f"kr={MASK_KEEP_RATIO} selected={num_selected} "
            f"corrupted_selected={num_corrupted} "
            f"ratio={num_corrupted / max(1, num_selected):.6f} "
            f"dataset_baseline={float(special.mean()):.6f} "
            f"distribution_shift={float(mask_stats['distribution_shift']):.6f}; "
            "mask not saved"
        )
        corruption_operations = print_corruption_operation_retention(
            ctx, corruption_mask
        )

    metadata = {
        "experiment": experiment,
        "dataset": dataset,
        "seed": int(trial.SEED),
        "gain_definition": (
            "sqrt(early_mean * middle_mean) * (middle_mean - early_mean)"
        ),
        "stability_definition": (
            "variance_late(true_class_cross_entropy_loss) - "
            "variance_middle(true_class_cross_entropy_loss)"
        ),
        "dynamic_target": "mean(A,C,T), no noise gate",
        "keep_original_metrics": sorted(args.keep_items),
        "a_strategy": strategy,
        "a_epochs": int(epochs),
        "revised_a_cache": str(cache_path),
        "regression_tol": float(args.regression_tol),
        "regression_mse": float(fit["mse"]),
        "mask_solver": (
            None if mask_stats is None else str(mask_stats.get("solver"))
        ),
        "note": (
            "Weights and masks are printed/used in memory only. "
            "Correlation results are saved."
        ),
    }

    del (
        static,
        components,
        new_parts,
        signals,
        pseudo_label,
        weights,
        fit,
        mask_stats,
        corruption_operations,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, metadata


def main() -> None:
    args = parse_args()
    validate_args(args)

    datasets = trial.selected_datasets(args.dataset)
    experiments = trial.selected_experiments(args.experiment)
    tasks = [
        (experiment, dataset)
        for dataset in datasets
        for experiment in experiments
    ]

    print("=" * 130)
    print("change_Ag_2: revised A, no noise gate, experiment-specific group")
    print(
        "G = sqrt(early_mean * middle_mean) * "
        "(middle_mean - early_mean)"
    )
    print(
        "S = Var_late(true-class CE loss) - "
        "Var_middle(true-class CE loss)"
    )
    print("dynamic target = mean(A, C, T); noise_gate is not used or computed")
    print("label_noise group = standard explicit distribution correction")
    print(
        "corruption group = weighted candidate pool + "
        "distribution-repair pair probability"
    )
    print(f"datasets={datasets}")
    print(f"experiments={experiments}")
    print(f"keep original/full-run definitions={sorted(args.keep_items)}")
    print(f"regression_tol={float(args.regression_tol):.1e}")
    print("Weights and masks are not saved.")
    print("=" * 130)

    all_rows: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    for experiment, dataset in tqdm(
        tasks, desc="change_Ag_2 tasks", unit="task"
    ):
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
        out_dir.mkdir(parents=True, exist_ok=True)
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
    print("=" * 130)
    print(f"Combined CSV: {combined_dir / 'correlations.csv'}")
    print(f"Combined JSON: {combined_dir / 'correlations.json'}")
    print(
        "Revised A caches: "
        f"{YYY_ROOT / 'a_balanced_gain_loss_stability_cache_v2'}"
    )
    print("No weights or masks were saved.")
    print("=" * 130)


if __name__ == "__main__":
    main()