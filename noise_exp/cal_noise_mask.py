#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label-noise mask calculation for group-mode selection.

Run from the project root, for example:

    CUDA_VISIBLE_DEVICES=0 python noise_exp/cal_noise_mask.py \
        --dataset tiny-imagenet --seed 22 --weight-group naive --kr 30,50

    CUDA_VISIBLE_DEVICES=0 python noise_exp/cal_noise_mask.py \
        --dataset tiny-imagenet --seed 22 --weight-group learned --kr 30,50

This script is designed for the fixed label-noise experiment:

  - noise file format:
        noise/{dataset}/noise_list_{seed}.txt
    with two integer columns and no header:
        sample_id noisy_label

  - output masks:
        noise_exp/mask/noise_{weight_group}_group/{dataset}/{seed}/mask_{kr}.npz

  - intermediate files are rooted under noise_exp/:
        noise_exp/adapters/
        noise_exp/weights/proxy_logs/
        noise_exp/weights/dynamic_cache/
        noise_exp/weights/scoring_weights.json
        noise_exp/static_scores/

For --weight-group learned, the full pipeline is:
  1. Patch training labels according to noise/{dataset}/noise_list_{seed}.txt.
  2. Train CLIP adapters on noisy labels.
  3. Train CV proxy models on noisy labels.
  4. Learn static-score weights from proxy dynamics and noisy-label static scores.
  5. Compute SA/Div/DDS static scores on noisy labels.
  6. Run group-mode selection and save masks.

For --weight-group naive, stages 3 and 4 are skipped.  The mask is computed
with equal weights for DDS/Div/SA, following calculate_my_mask.py's naive weight
logic.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from math import ceil
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
NOISE_EXP_ROOT = THIS_FILE.parent
PROJECT_ROOT = NOISE_EXP_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
NOISE_ROOT = PROJECT_ROOT / "noise"

ADAPTER_ROOT = NOISE_EXP_ROOT / "adapters"
WEIGHTS_ROOT = NOISE_EXP_ROOT / "weights"
PROXY_LOG_ROOT = WEIGHTS_ROOT / "proxy_logs"
DYNAMIC_CACHE_ROOT = WEIGHTS_ROOT / "dynamic_cache"
STATIC_SCORE_ROOT = NOISE_EXP_ROOT / "static_scores"
MASK_ROOT = NOISE_EXP_ROOT / "mask"
NOISE_GATE_CACHE_VERSION = "noise_gate_v4_learn_after_val_margin"
NOISE_PRIOR_RATIO = 0.2
NOISE_RISK_FACTOR = float(1.0 - np.sqrt(NOISE_PRIOR_RATIO))

# Project imports are intentionally placed after sys.path is set.
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.score_utils import standard_zscore, standard_zscore_by_class  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.training_defaults import get_default_training_config  # noqa: E402
from weights.dynamic_utils import (  # noqa: E402
    FoldLogData,
    load_cv_fold_logs,
    resolve_epoch_windows,
    standard_zscore_dynamic,
)

import learn_scoring_weights as learn_weights_mod  # noqa: E402
import train_adapter as train_adapter_mod  # noqa: E402
import train_proxy as train_proxy_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate noise experiment masks for naive_group or learned_group."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=AVAILABLE_DATASETS,
        help="Dataset name. Supports cifar10, cifar100, tiny-imagenet.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed list. Usually one of 22, 42, 96; comma-separated is also supported.",
    )
    parser.add_argument("--kr", type=str, default="30,50", help="Keep ratios. Default: 30,50.")
    parser.add_argument(
        "--weight-group",
        type=str,
        default="learned",
        choices=("naive", "learned"),
        help="naive: equal static weights and no proxy/weight-learning stage; learned: run full weight-learning pipeline.",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Only used in mask path rule.")
    parser.add_argument(
        "--mode-name",
        type=str,
        default=None,
        help="Mask method name. Default: noise_{weight_group}_group.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--skip-saved", action="store_true", help="Skip existing final masks.")
    parser.add_argument("--force", action="store_true", help="Rerun intermediate stages even if files exist.")
    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--gate-low", type=float, default=0.2)
    parser.add_argument("--gate-high", type=float, default=0.95)
    parser.add_argument("--ratio-lambda", type=float, default=1e-2)
    parser.add_argument("--regression-learning-rate", type=float, default=1e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-8)
    return parser.parse_args()


def parse_ratio_list(text: str) -> list[int]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("--kr cannot be empty.")
    ratios = [int(item) for item in items]
    for ratio in ratios:
        if ratio <= 0 or ratio > 100:
            raise ValueError(f"Invalid keep ratio: {ratio}")
    return ratios


# ---------------------------------------------------------------------------
# Dataset and label-noise utilities
# ---------------------------------------------------------------------------

def _tiny_train_root(data_root: Path = DATA_ROOT) -> Path:
    return data_root / "tiny-imagenet-200" / "train"


def _is_tiny_train_root(root: str | Path) -> bool:
    try:
        return Path(root).resolve() == _tiny_train_root().resolve()
    except FileNotFoundError:
        return Path(root) == _tiny_train_root()


def build_train_dataset(dataset_name: str, transform=None):
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = _tiny_train_root()
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
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


def set_dataset_labels(dataset, labels: np.ndarray) -> None:
    labels_list = [int(x) for x in labels.tolist()]
    updated = False

    if hasattr(dataset, "targets"):
        dataset.targets = labels_list
        updated = True
    if hasattr(dataset, "labels"):
        dataset.labels = labels_list
        updated = True
    if hasattr(dataset, "samples"):
        dataset.samples = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.samples)]
        updated = True
    if hasattr(dataset, "imgs"):
        dataset.imgs = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.imgs)]
        updated = True

    if not updated:
        raise TypeError(
            "The dataset does not expose targets/labels/samples/imgs and cannot be label-patched in-place."
        )


def read_noise_list(dataset_name: str, seed: int) -> np.ndarray:
    path = NOISE_ROOT / dataset_name / f"noise_list_{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Noise list not found: {path}")

    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Noise list must have shape (N,2), got {arr.shape}: {path}")

    return arr


def apply_noise_to_dataset(dataset, dataset_name: str, seed: int, *, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutate a torchvision-style train dataset in-place.

    For CIFAR this updates .targets.  For Tiny-ImageNet/ImageFolder this updates
    .targets, .samples and .imgs so __getitem__ returns the noisy labels.
    """
    clean_targets = extract_labels(dataset).astype(np.int64).copy()
    noisy_targets = clean_targets.copy()

    noise_map = read_noise_list(dataset_name, seed)
    sample_ids = noise_map[:, 0].astype(np.int64)
    new_labels = noise_map[:, 1].astype(np.int64)

    num_classes = len(getattr(dataset, "classes", np.unique(clean_targets)))
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate sample ids in noise list for {dataset_name}, seed={seed}.")
    if np.any(sample_ids < 0) or np.any(sample_ids >= len(clean_targets)):
        raise ValueError(f"Noise sample id out of range for {dataset_name}, seed={seed}.")
    if np.any(new_labels < 0) or np.any(new_labels >= num_classes):
        raise ValueError(f"Noise label out of range for {dataset_name}, seed={seed}.")
    same = new_labels == clean_targets[sample_ids]
    if np.any(same):
        bad = int(np.sum(same))
        raise ValueError(f"{bad} noisy labels equal clean labels; this is not allowed.")

    noisy_targets[sample_ids] = new_labels
    set_dataset_labels(dataset, noisy_targets)

    is_noisy = np.zeros(len(clean_targets), dtype=bool)
    is_noisy[sample_ids] = True

    if verbose:
        print(
            f"[noise] dataset={dataset_name} seed={seed} "
            f"num_noisy={len(sample_ids)}/{len(clean_targets)} "
            f"rate={len(sample_ids) / len(clean_targets):.4f}"
        )

    return clean_targets, noisy_targets, is_noisy


def load_noisy_reference_dataset(dataset_name: str, seed: int, transform=None, *, verbose: bool = False):
    dataset = build_train_dataset(dataset_name, transform=transform)
    clean, noisy, is_noisy = apply_noise_to_dataset(dataset, dataset_name, seed, verbose=verbose)
    return dataset, clean, noisy, is_noisy


@contextlib.contextmanager
def patched_training_label_noise(dataset_name: str, seed: int, *, verbose_once: bool = False) -> Iterator[None]:
    """Patch torchvision dataset constructors used by existing project scripts.

    CIFAR train splits are patched only when train=True.  Tiny-ImageNet is backed
    by ImageFolder, so only ImageFolder calls whose root is data/tiny-imagenet-200/train
    are patched.  Validation/test ImageFolder roots are left clean.
    """
    orig_cifar10 = datasets.CIFAR10
    orig_cifar100 = datasets.CIFAR100
    orig_imagefolder = datasets.ImageFolder
    printed = {"value": False}

    def _should_patch_train(args, kwargs) -> bool:
        if "train" in kwargs:
            return bool(kwargs["train"])
        if len(args) >= 2:
            return bool(args[1])
        return True

    def _verbose_now() -> bool:
        do_verbose = verbose_once and not printed["value"]
        printed["value"] = True
        return do_verbose

    def noisy_cifar10(*args, **kwargs):
        ds = orig_cifar10(*args, **kwargs)
        if dataset_name == CIFAR10 and _should_patch_train(args, kwargs):
            apply_noise_to_dataset(ds, CIFAR10, seed, verbose=_verbose_now())
        return ds

    def noisy_cifar100(*args, **kwargs):
        ds = orig_cifar100(*args, **kwargs)
        if dataset_name == CIFAR100 and _should_patch_train(args, kwargs):
            apply_noise_to_dataset(ds, CIFAR100, seed, verbose=_verbose_now())
        return ds

    def noisy_imagefolder(root, *args, **kwargs):
        ds = orig_imagefolder(root, *args, **kwargs)
        if dataset_name == TINY_IMAGENET and _is_tiny_train_root(root):
            apply_noise_to_dataset(ds, TINY_IMAGENET, seed, verbose=_verbose_now())
        return ds

    datasets.CIFAR10 = noisy_cifar10  # type: ignore[assignment]
    datasets.CIFAR100 = noisy_cifar100  # type: ignore[assignment]
    datasets.ImageFolder = noisy_imagefolder  # type: ignore[assignment]
    try:
        yield
    finally:
        datasets.CIFAR10 = orig_cifar10  # type: ignore[assignment]
        datasets.CIFAR100 = orig_cifar100  # type: ignore[assignment]
        datasets.ImageFolder = orig_imagefolder  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Path patching for existing project modules
# ---------------------------------------------------------------------------

def noise_adapter_dir(dataset_name: str, seed: int) -> Path:
    path = ADAPTER_ROOT / dataset_name / str(seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def noise_adapter_paths(dataset_name: str, seed: int) -> tuple[Path, Path]:
    base = noise_adapter_dir(dataset_name, seed)
    return base / "adapter_image.pt", base / "adapter_context.pt"


@contextlib.contextmanager
def patched_adapter_output() -> Iterator[None]:
    old_train_resolve = getattr(train_adapter_mod, "resolve_adapter_dir", None)
    import model.adapter as adapter_mod

    old_adapter_resolve = getattr(adapter_mod, "resolve_adapter_dir", None)

    def _resolve(dataset_name: str, seed: int) -> Path:
        return noise_adapter_dir(dataset_name, seed)

    if old_train_resolve is not None:
        train_adapter_mod.resolve_adapter_dir = _resolve  # type: ignore[attr-defined]
    if old_adapter_resolve is not None:
        adapter_mod.resolve_adapter_dir = _resolve  # type: ignore[attr-defined]

    try:
        yield
    finally:
        if old_train_resolve is not None:
            train_adapter_mod.resolve_adapter_dir = old_train_resolve  # type: ignore[attr-defined]
        if old_adapter_resolve is not None:
            adapter_mod.resolve_adapter_dir = old_adapter_resolve  # type: ignore[attr-defined]


@contextlib.contextmanager
def patched_proxy_output() -> Iterator[None]:
    old_resolve = train_proxy_mod.resolve_proxy_log_dir

    def _resolve_proxy_log_dir(
        dataset: str,
        seed: int | None = None,
        *,
        proxy_model: str = "resnet18",
        epochs: int,
        root=None,
    ) -> Path:
        del root
        if seed is None:
            raise ValueError("noise proxy logs require an explicit seed.")
        return PROXY_LOG_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))

    train_proxy_mod.resolve_proxy_log_dir = _resolve_proxy_log_dir  # type: ignore[assignment]
    try:
        yield
    finally:
        train_proxy_mod.resolve_proxy_log_dir = old_resolve  # type: ignore[assignment]


@contextlib.contextmanager
def patched_weight_learning_paths() -> Iterator[None]:
    old_cache_dir = getattr(learn_weights_mod, "resolve_dynamic_component_cache_dir", None)
    old_cache_path = getattr(learn_weights_mod, "resolve_dynamic_component_cache_path", None)
    old_static_cache = getattr(learn_weights_mod, "get_or_compute_static_scores", None)

    def _cache_dir(dataset: str, proxy_model: str, seed: int, epochs: int) -> Path:
        return DYNAMIC_CACHE_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))

    def _cache_path(dataset: str, proxy_model: str, seed: int, epochs: int, component_name: str) -> Path:
        return _cache_dir(dataset, proxy_model, seed, epochs) / f"{component_name.strip().upper()}.npz"

    def _static_cache_wrapper(**kwargs):
        kwargs["cache_root"] = STATIC_SCORE_ROOT
        return get_or_compute_static_scores(**kwargs)

    if old_cache_dir is not None:
        learn_weights_mod.resolve_dynamic_component_cache_dir = _cache_dir  # type: ignore[attr-defined]
    if old_cache_path is not None:
        learn_weights_mod.resolve_dynamic_component_cache_path = _cache_path  # type: ignore[attr-defined]
    if old_static_cache is not None:
        learn_weights_mod.get_or_compute_static_scores = _static_cache_wrapper  # type: ignore[attr-defined]

    try:
        yield
    finally:
        if old_cache_dir is not None:
            learn_weights_mod.resolve_dynamic_component_cache_dir = old_cache_dir  # type: ignore[attr-defined]
        if old_cache_path is not None:
            learn_weights_mod.resolve_dynamic_component_cache_path = old_cache_path  # type: ignore[attr-defined]
        if old_static_cache is not None:
            learn_weights_mod.get_or_compute_static_scores = old_static_cache  # type: ignore[attr-defined]


@contextlib.contextmanager
def patched_group_mean_cache() -> Iterator[None]:
    """Kept for backward-compatible structure; group mean cache is local now."""
    yield


# ---------------------------------------------------------------------------
# Helpers for invoking existing scripts in-process
# ---------------------------------------------------------------------------

def call_module_main(module, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "module")] + argv

        if hasattr(module, "main"):
            module.main()
            return

        if hasattr(module, "parse_args") and hasattr(module, "run_once"):
            parsed_args = module.parse_args()
            if hasattr(module, "parse_seed_list"):
                output_seeds = module.parse_seed_list(parsed_args.seed)
            else:
                output_seeds = parse_seed_list(parsed_args.seed)
            if not output_seeds:
                raise ValueError(f"{module.__name__}: --seed must contain at least one seed.")
            module.run_once(parsed_args, output_seeds)
            return

        raise AttributeError(
            f"Module {module.__name__} has neither main() nor parse_args()+run_once()."
        )
    finally:
        sys.argv = old_argv


def proxy_epochs_for_dataset(dataset_name: str) -> int:
    return int(get_default_training_config(dataset_name)["epochs"])


def all_proxy_logs_exist(
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    k_folds: int = 5,
) -> bool:
    log_dir = PROXY_LOG_ROOT / dataset_name / proxy_model / str(int(seed)) / str(int(epochs))
    meta_path = log_dir / "meta.json"
    fold_paths = [log_dir / f"fold_{i}.npz" for i in range(1, k_folds + 1)]
    return meta_path.is_file() and all(path.is_file() for path in fold_paths)


def weights_entry_exists(weights_path: Path, dataset_name: str, seed: int) -> bool:
    if not weights_path.is_file():
        return False
    try:
        data = json.loads(weights_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    entry = data.get(dataset_name)
    if not isinstance(entry, dict):
        return False
    seed_entry = entry.get(str(seed))
    return isinstance(seed_entry, dict) and seed_entry.get("method_version") == learn_weights_mod.METHOD_VERSION


# ---------------------------------------------------------------------------
# Noise-gated dynamic target utilities
# ---------------------------------------------------------------------------

def _noise_gate_cache_path(dataset_name, proxy_model, seed, epochs) -> Path:
    return (
        DYNAMIC_CACHE_ROOT
        / dataset_name
        / proxy_model
        / str(int(seed))
        / str(int(epochs))
        / "noise_gate.npz"
    )


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, samples, classes).")
    if labels.shape != (logits.shape[1],):
        raise ValueError("labels shape mismatch for logits.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[2]):
        raise ValueError("labels out of range for logits.")

    shifted = logits - np.max(logits, axis=2, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=2)) + np.max(logits, axis=2)
    true_logits = np.take_along_axis(logits, labels.reshape(1, -1, 1), axis=2).squeeze(2)
    losses = logsumexp - true_logits
    return np.nan_to_num(losses, nan=0.0, posinf=1.0e6, neginf=0.0).astype(np.float64)


def _percentile_rank_tie_aware(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains NaN/inf values.")
    n = int(values.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = 0.5 * float(i + j - 1)
        ranks[order[i:j]] = avg_rank / float(n - 1)
        i = j
    return ranks.astype(np.float64)


def _validate_gate_thresholds(gate_low: float, gate_high: float) -> tuple[float, float]:
    gate_low = float(gate_low)
    gate_high = float(gate_high)
    if not (0.0 <= gate_low <= 1.0 and 0.0 <= gate_high <= 1.0 and gate_low < gate_high):
        raise ValueError("gate_low and gate_high must satisfy 0 <= gate_low < gate_high <= 1.")
    return gate_low, gate_high


def _build_gate_from_final_risk(final_risk: np.ndarray, gate_low: float, gate_high: float) -> np.ndarray:
    gate_low, gate_high = _validate_gate_thresholds(gate_low, gate_high)
    final_risk = np.asarray(final_risk, dtype=np.float64)
    if not np.all(np.isfinite(final_risk)):
        raise ValueError("final_risk contains NaN/inf values.")
    gate = np.where(
        final_risk <= gate_low,
        1.0,
        np.where(final_risk >= gate_high, 0.0, 1.0 - (final_risk - gate_low) / (gate_high - gate_low)),
    )
    return np.clip(gate, 0.0, 1.0).astype(np.float64)


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _compute_final_noise_risk(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
) -> np.ndarray:
    labels_for_metric = np.asarray(labels_for_metric, dtype=np.int64)
    num_samples = int(labels_for_metric.shape[0])
    learn_window = int(learn_window)
    learn_min_correct = int(learn_min_correct)
    if learn_window <= 0:
        raise ValueError("learn_window must be positive.")
    if learn_min_correct <= 0 or learn_min_correct > learn_window:
        raise ValueError("learn_min_correct must be in [1, learn_window].")

    learn_sum = np.zeros(num_samples, dtype=np.float64)
    learn_count = np.zeros(num_samples, dtype=np.int64)
    loss_var_sum = np.zeros(num_samples, dtype=np.float64)
    loss_var_count = np.zeros(num_samples, dtype=np.int64)
    val_bias_raw = np.full(num_samples, np.nan, dtype=np.float64)

    for fold in tqdm(folds, desc="compute noise gate", unit="fold", leave=False):
        train_idx = np.asarray(fold.train_indices, dtype=np.int64)
        y_train = labels_for_metric[train_idx]
        train_logits = np.asarray(fold.train_logits, dtype=np.float64)
        if train_logits.ndim != 3 or train_logits.shape[1] != train_idx.shape[0]:
            raise ValueError(f"fold {fold.fold_id} train_logits shape mismatch.")

        train_preds = np.argmax(train_logits, axis=2)
        correct = train_preds == y_train.reshape(1, -1)
        num_epochs = int(correct.shape[0])
        for local_i, sample_idx in enumerate(train_idx):
            seq = correct[:, local_i].astype(np.int64)
            learned_time = None
            if num_epochs >= learn_window:
                window_sum = np.convolve(seq, np.ones(learn_window, dtype=np.int64), mode="valid")
                hit = np.flatnonzero(window_sum >= learn_min_correct)
                if hit.size > 0:
                    learned_time = int(hit[0] + learn_window - 1)
            if learned_time is None:
                learn_time_norm = 1.0
                forget_frequency = 1.0
                learn_risk = 1.0
            else:
                learn_time_norm = float(learned_time - learn_window + 1) / float(max(1, num_epochs - learn_window))
                learn_time_norm = float(np.clip(learn_time_norm, 0.0, 1.0))
                remaining = correct[learned_time + 1 :, local_i]
                if remaining.size == 0:
                    forget_frequency = 1.0
                else:
                    forget_frequency = float(np.mean(~remaining))
                learn_risk = 1.0 - (1.0 - learn_time_norm) * (1.0 - forget_frequency)
                learn_risk = float(np.clip(learn_risk, 0.0, 1.0))
            learn_sum[int(sample_idx)] += float(learn_risk)
            learn_count[int(sample_idx)] += 1

        losses = _cross_entropy_from_logits(train_logits, y_train)
        _, mid_idx, late_idx = resolve_epoch_windows(num_epochs)
        mid_late_idx = np.concatenate([mid_idx, late_idx]).astype(np.int64)
        loss_var = np.var(losses[mid_late_idx], axis=0).astype(np.float64)
        loss_var_sum[train_idx] += loss_var
        loss_var_count[train_idx] += 1

        val_idx = np.asarray(fold.val_indices, dtype=np.int64)
        y_val = labels_for_metric[val_idx]
        val_logits = np.asarray(fold.val_logits, dtype=np.float64)
        if val_logits.ndim != 3 or val_logits.shape[1] != val_idx.shape[0]:
            raise ValueError(f"fold {fold.fold_id} val_logits shape mismatch.")
        _, val_mid_idx, val_late_idx = resolve_epoch_windows(int(val_logits.shape[0]))
        val_mid_late_idx = np.concatenate([val_mid_idx, val_late_idx]).astype(np.int64)
        probs = _softmax_logits(val_logits[val_mid_late_idx])
        for local_i, sample_idx in enumerate(val_idx):
            label = int(y_val[local_i])
            probs_i = probs[:, local_i, :]
            label_probs = probs_i[:, label]
            positive_margins = np.maximum(probs_i - label_probs[:, None], 0.0)
            positive_margins[:, label] = 0.0
            class_scores = np.mean(positive_margins, axis=0)
            class_scores[label] = 0.0
            val_bias_raw[int(sample_idx)] = float(np.max(class_scores))

    if np.any(learn_count <= 0) or np.any(loss_var_count <= 0) or np.any(~np.isfinite(val_bias_raw)):
        missing = np.where((learn_count <= 0) | (loss_var_count <= 0) | (~np.isfinite(val_bias_raw)))[0]
        raise ValueError(f"Some samples lack noise gate metrics: {missing[:10]}.")

    learn_risk_raw = learn_sum / learn_count
    loss_var_raw = loss_var_sum / loss_var_count
    learn_risk_rank = _percentile_rank_tie_aware(learn_risk_raw)
    loss_var_rank = _percentile_rank_tie_aware(loss_var_raw)
    val_bias_rank = _percentile_rank_tie_aware(val_bias_raw)
    final_risk = np.minimum.reduce([learn_risk_rank, loss_var_rank, val_bias_rank])
    return final_risk.astype(np.float64)


def _load_noise_gate_cache_if_valid(
    cache_path: Path,
    dataset_name,
    proxy_model,
    seed,
    epochs,
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(np.asarray(data["cache_version"]).item()) != NOISE_GATE_CACHE_VERSION:
                return None
            if str(np.asarray(data["dataset"]).item()) != str(dataset_name):
                return None
            if str(np.asarray(data["proxy_model"]).item()) != str(proxy_model):
                return None
            if int(np.asarray(data["proxy_training_seed"]).item()) != int(seed):
                return None
            if int(np.asarray(data["epochs"]).item()) != int(epochs):
                return None
            if int(np.asarray(data["learn_window"]).item()) != int(learn_window):
                return None
            if int(np.asarray(data["learn_min_correct"]).item()) != int(learn_min_correct):
                return None
            labels_cached = np.asarray(data["labels"], dtype=np.int64)
            labels_all = np.asarray(labels_all, dtype=np.int64)
            if not np.array_equal(labels_cached, labels_all):
                return None
            final_risk = np.asarray(data["final_risk"], dtype=np.float64)
    except Exception:
        return None

    if final_risk.shape != (int(labels_all.shape[0]),) or not np.all(np.isfinite(final_risk)):
        return None
    return {"final_risk": final_risk.astype(np.float64)}


def _save_noise_gate_cache(
    cache_path: Path,
    dataset_name,
    proxy_model,
    seed,
    epochs,
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
    final_risk: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.array(NOISE_GATE_CACHE_VERSION),
        dataset=np.array(dataset_name),
        proxy_model=np.array(proxy_model),
        proxy_training_seed=np.array(int(seed), dtype=np.int64),
        epochs=np.array(int(epochs), dtype=np.int64),
        labels=np.asarray(labels_all, dtype=np.int64),
        learn_window=np.array(int(learn_window), dtype=np.int64),
        learn_min_correct=np.array(int(learn_min_correct), dtype=np.int64),
        final_risk=np.asarray(final_risk, dtype=np.float64),
    )


def get_or_compute_noise_gate(args, seed, epochs, *, force=False) -> dict[str, np.ndarray]:
    cache_path = _noise_gate_cache_path(args.dataset, args.proxy_model, seed, epochs)
    proxy_log_path = PROXY_LOG_ROOT / args.dataset / args.proxy_model / str(int(seed)) / str(int(epochs))
    folds, labels_all = load_cv_fold_logs(proxy_log_path, args.dataset, str(DATA_ROOT))
    labels_all = np.asarray(labels_all, dtype=np.int64)
    _validate_gate_thresholds(args.gate_low, args.gate_high)

    cached = None
    if not force:
        cached = _load_noise_gate_cache_if_valid(
            cache_path,
            args.dataset,
            args.proxy_model,
            seed,
            epochs,
            labels_all,
            args.learn_window,
            args.learn_min_correct,
        )
        if cached is not None:
            print(f"[noise-gate] cache HIT: {cache_path}")
            final_risk = cached["final_risk"]
            return {"final_risk": final_risk, "gate": _build_gate_from_final_risk(final_risk, args.gate_low, args.gate_high)}

    final_risk = _compute_final_noise_risk(folds, labels_all, args.learn_window, args.learn_min_correct)
    _save_noise_gate_cache(
        cache_path,
        args.dataset,
        args.proxy_model,
        seed,
        epochs,
        labels_all,
        args.learn_window,
        args.learn_min_correct,
        final_risk,
    )
    status = "FORCE->RECOMPUTED" if force else "MISS->COMPUTED"
    print(f"[noise-gate] cache {status}: {cache_path}")
    return {"final_risk": final_risk, "gate": _build_gate_from_final_risk(final_risk, args.gate_low, args.gate_high)}

@contextlib.contextmanager
def patched_noise_gate_dynamic_target(args, seed: int, epochs: int) -> Iterator[dict[str, np.ndarray]]:
    gate_data = get_or_compute_noise_gate(args, seed, epochs, force=args.force)
    old_build_dynamic_target = learn_weights_mod.build_dynamic_target

    def _gate_raw_component_then_normalize(result, gate: np.ndarray) -> np.ndarray:
        raw = result.raw_foldwise.astype(np.float64)
        gate = np.asarray(gate, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != gate.shape[0]:
            raise ValueError(f"raw_foldwise/noise_gate shape mismatch: raw={raw.shape}, gate={gate.shape}")
        fold_normalized = np.full(raw.shape, np.nan, dtype=np.float32)
        for f in range(raw.shape[0]):
            finite = np.isfinite(raw[f])
            if not np.any(finite):
                continue
            values = np.maximum(raw[f, finite], 0.0) * gate[finite]
            fold_normalized[f, finite] = standard_zscore_dynamic(values.astype(np.float32))

        finite_counts = np.sum(np.isfinite(fold_normalized), axis=0)
        if np.any(finite_counts <= 0):
            missing = np.where(finite_counts <= 0)[0]
            raise ValueError(f"raw component has samples without finite fold values: {missing[:10]}.")
        aggregated = np.nanmean(fold_normalized.astype(np.float64), axis=0)
        final_normalized = standard_zscore_dynamic(aggregated.astype(np.float32))
        if not np.all(np.isfinite(final_normalized)):
            raise ValueError("gated raw component normalization produced NaN/inf values.")
        return final_normalized.astype(np.float64)

    def _build_dynamic_target_with_noise_gate(component_results):
        a = component_results["A"].final_normalized.astype(np.float64)
        gate = gate_data["gate"].astype(np.float64)
        final_risk = gate_data["final_risk"].astype(np.float64)
        if not (a.shape == gate.shape == final_risk.shape):
            raise ValueError(f"A/noise_gate/final_risk shape mismatch: A={a.shape}, gate={gate.shape}, risk={final_risk.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("A contains NaN/inf values.")

        gated_a = np.minimum(a, 0.0) + gate * np.maximum(a, 0.0)
        gated_c = _gate_raw_component_then_normalize(component_results["C"], gate)
        gated_t = _gate_raw_component_then_normalize(component_results["T"], gate)
        utility_raw = (gated_a + gated_c + gated_t) / 3.0
        u_scores = standard_zscore(utility_raw)
        dynamic_component_values = {
            "A": component_results["A"].final_normalized,
            "C": component_results["C"].final_normalized,
            "T": component_results["T"].final_normalized,
            "noise_gate": gate_data["gate"],
            "noise_final_risk": gate_data["final_risk"],
        }
        return u_scores.astype(np.float64), dynamic_component_values

    learn_weights_mod.build_dynamic_target = _build_dynamic_target_with_noise_gate  # type: ignore[assignment]
    try:
        yield gate_data
    finally:
        learn_weights_mod.build_dynamic_target = old_build_dynamic_target  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_adapter_stage(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[Path, Path]:
    image_path, text_path = noise_adapter_paths(args.dataset, seed)
    if image_path.is_file() and text_path.is_file() and not args.force:
        print(f"[adapter] skip existing adapters: {image_path.parent}")
        return image_path, text_path

    print(f"[adapter] train noisy-label CLIP adapters | dataset={args.dataset} seed={seed}")
    with patched_training_label_noise(args.dataset, seed, verbose_once=True), patched_adapter_output():
        cli = [
            "--dataset", args.dataset,
            "--data-root", str(DATA_ROOT),
            "--clip-model", args.clip_model,
            "--seed", str(seed),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        if args.debug_prompts:
            cli += ["--debug-prompts"]
        call_module_main(train_adapter_mod, cli)

    if not image_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"Adapter training finished but weights are missing: {image_path}, {text_path}")
    return image_path, text_path


def run_proxy_stage(args: argparse.Namespace, seed: int, device: torch.device) -> None:
    epochs = proxy_epochs_for_dataset(args.dataset)
    log_dir = PROXY_LOG_ROOT / args.dataset / args.proxy_model / str(int(seed)) / str(epochs)
    if all_proxy_logs_exist(args.dataset, args.proxy_model, seed, epochs, args.k_folds) and not args.force:
        print(f"[proxy] skip existing proxy logs: {log_dir}")
        return

    print(f"[proxy] train noisy-label CV proxy | dataset={args.dataset} seed={seed} model={args.proxy_model}")
    with patched_training_label_noise(args.dataset, seed, verbose_once=True), patched_proxy_output():
        cli = [
            "--dataset", args.dataset,
            "--data_root", str(DATA_ROOT),
            "--model", args.proxy_model,
            "--seed", str(seed),
            "--k_folds", str(args.k_folds),
            "--num_workers", str(args.num_workers),
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        call_module_main(train_proxy_mod, cli)


def run_weight_learning_stage(args: argparse.Namespace, seed: int, device: torch.device, image_path: Path, text_path: Path) -> None:
    weights_path = WEIGHTS_ROOT / "scoring_weights.json"
    epochs = proxy_epochs_for_dataset(args.dataset)

    if weights_entry_exists(weights_path, args.dataset, seed) and not args.force:
        print(f"[weights] skip existing learned weights: {weights_path} dataset={args.dataset} seed={seed}")
        return

    print(f"[weights] learn static-score weights | dataset={args.dataset} seed={seed}")
    base_ratio_lambda = float(args.ratio_lambda)
    scaled_ratio_lambda = base_ratio_lambda * (NOISE_RISK_FACTOR ** 2)
    # 标签注噪实验中动态监督更不稳定，按先验噪声风险因子的平方降低 ratio 正则强度。
    print(f"[weights] ratio_lambda scaled by noise factor: base={base_ratio_lambda}, scaled={scaled_ratio_lambda}")
    with (
        patched_training_label_noise(args.dataset, seed, verbose_once=True),
        patched_weight_learning_paths(),
    ):
        cli = [
            "--dataset", args.dataset,
            "--data-root", str(DATA_ROOT),
            "--proxy-log", str(PROXY_LOG_ROOT),
            "--proxy-model", args.proxy_model,
            "--proxy-epochs", str(epochs),
            "--adapter-image-path", str(image_path),
            "--adapter-text-path", str(text_path),
            "--clip-model", args.clip_model,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--output", str(weights_path),
            "--seed", str(seed),
            "--proxy-training-seed", str(seed),
            "--ratio-lambda", f"{scaled_ratio_lambda:.12g}",
            "--regression-learning-rate", str(args.regression_learning_rate),
            "--regression-max-iter", str(args.regression_max_iter),
            "--regression-tol", str(args.regression_tol),
            "--learn-window", str(args.learn_window),
            "--learn-min-correct", str(args.learn_min_correct),
            "--gate-low", str(args.gate_low),
            "--gate-high", str(args.gate_high),
            "--use-noise-gate",
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        if args.debug_prompts:
            cli += ["--debug-prompts"]
        if args.force:
            cli += ["--force-noise-gate"]
        call_module_main(learn_weights_mod, cli)


def build_noisy_score_loader(
    preprocess,
    dataset_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset, _, _, _ = load_noisy_reference_dataset(dataset_name, seed, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_class_names_for_noisy_dataset(dataset_name: str) -> list[str]:
    dataset = build_train_dataset(dataset_name, transform=None)
    return list(
        resolve_class_names_for_prompts(
            dataset_name=dataset_name,
            data_root=DATA_ROOT,
            class_names=dataset.classes,  # type: ignore[attr-defined]
        )
    )



# ---------------------------------------------------------------------------
# Local weight and group-selection utilities
# ---------------------------------------------------------------------------

def ensure_scoring_weights(path: Path, dataset_name: str) -> dict[str, dict[str, object]]:
    """Create/read scoring weights and always provide the naive equal-weight group."""
    data: dict[str, dict[str, object]] = {}
    updated = False
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
            updated = True

    dataset_entry = data.get(dataset_name)
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        updated = True

    naive = dataset_entry.get("naive")
    if not isinstance(naive, dict):
        naive = {}
        updated = True

    default_weight = 1.0 / 3.0
    for key in ("dds", "div", "sa"):
        if key not in naive:
            naive[key] = default_weight
            updated = True

    total = 0.0
    for key in ("dds", "div", "sa"):
        try:
            naive[key] = float(naive[key])
        except (TypeError, ValueError):
            naive[key] = default_weight
            updated = True
        total += float(naive[key])

    if total <= 0:
        for key in ("dds", "div", "sa"):
            naive[key] = default_weight
        updated = True
    elif abs(total - 1.0) > 1e-12:
        for key in ("dds", "div", "sa"):
            naive[key] = float(naive[key]) / total
        updated = True

    dataset_entry["naive"] = naive
    data[dataset_name] = dataset_entry

    if updated or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {k: v for k, v in dataset_entry.items() if isinstance(v, dict)}


def _to_weight_triplet(selected: dict[str, object], group_name: str) -> dict[str, float]:
    required = {"dds", "div", "sa"}
    missing = required - set(selected.keys())
    if missing:
        raise ValueError(f"权重组 {group_name} 缺少必要键: {', '.join(sorted(missing))}")

    weights: dict[str, float] = {}
    for key in sorted(required):
        try:
            weights[key] = float(selected[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group_name} 的 {key} 无法转换为 float。") from exc

    total = sum(weights.values())
    if total <= 0:
        return {"dds": 1.0 / 3.0, "div": 1.0 / 3.0, "sa": 1.0 / 3.0}
    return {key: float(value / total) for key, value in weights.items()}


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    weight_group: str,
    seed: int,
) -> dict[str, float]:
    mode = weight_group.strip().lower()
    if mode == "naive":
        selected = all_weights.get("naive")
        if not isinstance(selected, dict):
            raise KeyError("未找到 naive 权重组。")
        return _to_weight_triplet(selected, "naive")
    if mode == "learned":
        selected = all_weights.get(str(seed))
        if not isinstance(selected, dict):
            raise KeyError(f"未找到 learned 权重组（seed={seed}）。")
        return _to_weight_triplet(selected, str(seed))
    raise ValueError("weight-group 仅支持 {'naive', 'learned'}")


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mean_stats_cache_path(dataset_name: str, clip_model: str, adapter_image_path: str) -> Path:
    adapter_sha1 = _hash_file(Path(adapter_image_path))
    clip_tag = clip_model.replace("/", "-").replace(" ", "_")
    return STATIC_SCORE_ROOT / "group_mean_stats" / dataset_name / clip_tag / f"img_adapter_{adapter_sha1}.npz"


def _get_or_compute_group_mean_stats(
    *,
    cache_path: Path,
    image_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(image_features.shape[0])
    feat_dim = int(image_features.shape[1]) if image_features.ndim == 2 else 0

    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_n = int(np.asarray(cached["n_samples"]).item())
            cached_dim = int(np.asarray(cached["feat_dim"]).item())
            cached_cls = int(np.asarray(cached["num_classes"]).item())
            means = np.asarray(cached["full_class_mean"], dtype=np.float32)
            vars_ = np.asarray(cached["full_class_var"], dtype=np.float32)
            if (
                cached_n == n_samples
                and cached_dim == feat_dim
                and cached_cls == int(num_classes)
                and means.shape == (num_classes, feat_dim)
                and vars_.shape == (num_classes,)
            ):
                return means, vars_
        except Exception:
            pass

    full_class_mean = np.zeros((num_classes, feat_dim), dtype=np.float32)
    full_class_var = np.zeros((num_classes,), dtype=np.float32)
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_feats = image_features[class_mask]
        if class_feats.shape[0] == 0:
            continue
        class_mean = np.mean(class_feats, axis=0, dtype=np.float32)
        diff = class_feats - class_mean
        sigma2 = float(np.mean(np.sum(diff * diff, axis=1)))
        full_class_mean[class_id] = class_mean
        full_class_var[class_id] = np.float32(max(sigma2, 0.0))

    np.savez_compressed(
        cache_path,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        n_samples=np.asarray(n_samples, dtype=np.int64),
        feat_dim=np.asarray(feat_dim, dtype=np.int64),
        num_classes=np.asarray(num_classes, dtype=np.int64),
    )
    return full_class_mean, full_class_var


def select_group_mask_local(
    sa_raw_scores: np.ndarray,
    div_metric: Div,
    div_loader: DataLoader,
    image_adapter,
    labels: np.ndarray,
    weights: dict[str, float],
    num_classes: int,
    keep_ratio: int,
    device: torch.device,
    dataset_name: str,
    seed: int,
    weight_group: str,
    clip_model: str,
    adapter_image_path: str,
    div_static_scores: np.ndarray | None = None,
    dds_static_scores: np.ndarray | None = None,
    group_candidate_pool_size: int = 1,
    group_init_count: int = 2,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    del weight_group, div_static_scores
    if keep_ratio <= 0 or keep_ratio > 100:
        raise ValueError("kr 必须在 1-100 之间。")

    num_samples = sa_raw_scores.shape[0]
    labels_np = np.asarray(labels, dtype=np.int64)
    sa_raw_np = np.asarray(sa_raw_scores, dtype=np.float32)
    dds_raw_np = np.asarray(dds_static_scores, dtype=np.float32) if dds_static_scores is not None else np.zeros(num_samples, dtype=np.float32)
    if labels_np.shape[0] != num_samples or dds_raw_np.shape[0] != num_samples:
        raise ValueError("样本数不一致，无法执行 group。")

    sr = float(keep_ratio) / 100.0
    target_size = int(round(sr * num_samples))
    target_size = min(num_samples, max(1, target_size)) if num_samples > 0 else 0
    if target_size <= 0:
        raise ValueError("target_size 必须大于 0。")

    class_indices_list = [np.flatnonzero(labels_np == c).astype(np.int64) for c in range(num_classes)]
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = (
        div_features.detach().cpu().numpy()
        if isinstance(div_features, torch.Tensor)
        else np.asarray(div_features)
    ).astype(np.float32)

    mean_stats_cache_path = _mean_stats_cache_path(
        dataset_name=dataset_name,
        clip_model=clip_model,
        adapter_image_path=adapter_image_path,
    )
    full_class_mean, _ = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache_path,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    full_class_mean_f32 = full_class_mean.astype(np.float32, copy=False)

    def _allocate_class_budgets() -> np.ndarray:
        class_sizes = np.asarray([idx.size for idx in class_indices_list], dtype=np.int64)
        raw = class_sizes.astype(np.float64) * sr
        floor_budget = np.floor(raw).astype(np.int64)
        floor_budget = np.minimum(floor_budget, class_sizes)
        need = int(target_size - np.sum(floor_budget))
        if need <= 0:
            return floor_budget
        frac = raw - floor_budget.astype(np.float64)
        order = np.lexsort((np.arange(num_classes, dtype=np.int64), -frac))
        budgets = floor_budget.copy()
        for class_id in order:
            if need <= 0:
                break
            if budgets[class_id] >= class_sizes[class_id]:
                continue
            budgets[class_id] += 1
            need -= 1
        if need != 0:
            raise RuntimeError("类别预算分配失败，无法满足目标总样本数。")
        return budgets

    class_budgets = _allocate_class_budgets()
    candidate_pool_size = max(1, int(group_candidate_pool_size))
    base_dist_weight_max = max(0.0, 0.8 - 0.005 * keep_ratio)
    # 标签注噪实验中分布修正项可能放大错误标签结构，因此按先验噪声风险因子降低其强度。
    dist_weight_max = base_dist_weight_max * NOISE_RISK_FACTOR
    dist_weight_min = 0.5 * dist_weight_max

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    class_selected_counts = np.zeros(num_classes, dtype=np.int64)
    class_selected_sum = np.zeros((num_classes, div_features_np.shape[1]), dtype=np.float32)
    init_per_class = np.zeros(num_classes, dtype=np.int64)
    requested_init_count = max(0, int(group_init_count))

    for class_id, class_indices in enumerate(class_indices_list):
        budget = int(class_budgets[class_id])
        if class_indices.size == 0 or budget <= 0 or requested_init_count <= 0:
            continue
        init_count = min(requested_init_count, budget, int(class_indices.size))
        init_per_class[class_id] = init_count
        top_pool_size = max(init_count, int(np.ceil(0.5 * class_indices.size)))
        top_pool_size = min(int(class_indices.size), max(1, top_pool_size))
        ranked_by_sa = np.argsort(-sa_raw_np[class_indices], kind="mergesort")[:top_pool_size]
        init_pool = class_indices[ranked_by_sa]
        if init_pool.size <= init_count:
            init_indices = init_pool
        else:
            init_indices = rng.choice(init_pool, size=init_count, replace=False).astype(np.int64)
        selected_mask[init_indices] = 1
        class_selected_counts[class_id] = init_count
        class_selected_sum[class_id] = np.sum(div_features_np[init_indices], axis=0, dtype=np.float32)

    selected_count_history: list[int] = [int(np.sum(selected_mask))]
    total_to_add = int(np.sum(class_budgets) - np.sum(init_per_class))
    pbar = tqdm(total=total_to_add, desc="[group] classwise greedy add", unit="sample")
    round_id = 0
    total_score_acc = 0.0

    while True:
        remaining_by_class = class_budgets - class_selected_counts
        active_classes = np.flatnonzero(remaining_by_class > 0).astype(np.int64)
        if active_classes.size == 0:
            break
        round_id += 1
        remain_total = int(np.sum(remaining_by_class))
        if remain_total < active_classes.size:
            chosen_classes = np.sort(rng.choice(active_classes, size=remain_total, replace=False).astype(np.int64))
        else:
            chosen_classes = active_classes

        for class_id in chosen_classes:
            class_indices = class_indices_list[int(class_id)]
            unselected_mask = selected_mask[class_indices] == 0
            candidate_indices = class_indices[unselected_mask]
            if candidate_indices.size == 0:
                continue

            current_count = int(class_selected_counts[class_id])
            if current_count <= 0:
                continue
            class_budget = int(class_budgets[class_id])
            progress = current_count / float(class_budget) if class_budget > 0 else 1.0
            progress = float(np.clip(progress, 0.0, 1.0))
            dist_weight_t = dist_weight_min + (dist_weight_max - dist_weight_min) * progress
            current_sum = class_selected_sum[class_id]
            mu_full = full_class_mean_f32[class_id]
            mu_sub = current_sum / float(current_count)
            old_dist = float(np.linalg.norm(mu_sub - mu_full))

            dynamic_k = max(3, int(ceil(0.05 * current_count)))

            candidate_features_t = torch.as_tensor(div_features_np[candidate_indices], dtype=torch.float32, device=device)
            reference_indices = class_indices[selected_mask[class_indices] > 0]
            reference_features_t = torch.as_tensor(div_features_np[reference_indices], dtype=torch.float32, device=device)
            div_raw = div_metric._knn_mean_distance_to_reference(
                query_features=candidate_features_t,
                reference_features=reference_features_t,
                k=float(dynamic_k),
                query_indices=torch.as_tensor(candidate_indices, dtype=torch.long, device=device),
                reference_indices=torch.as_tensor(reference_indices, dtype=torch.long, device=device),
            ).detach().cpu().numpy().astype(np.float32)
            div_local = standard_zscore(div_raw)

            candidate_features_np = div_features_np[candidate_indices]
            mu_new = (current_sum[None, :] + candidate_features_np) / float(current_count + 1)
            new_dist = np.linalg.norm(mu_new - mu_full[None, :], axis=1)
            dist_improve = (old_dist - new_dist).astype(np.float32)
            dist_local = standard_zscore(dist_improve)
            sa_local = standard_zscore(sa_raw_np[candidate_indices])
            dds_local = standard_zscore(dds_raw_np[candidate_indices])

            combined_scores = (
                weights["sa"] * sa_local
                + weights["dds"] * dds_local
                + weights["div"] * div_local
                + dist_weight_t * dist_local
            ).astype(np.float32)
            rank = np.argsort(-combined_scores, kind="mergesort")
            pool_n = min(candidate_pool_size, candidate_indices.size)
            pool_indices = candidate_indices[rank[:pool_n]]
            if pool_n == 1:
                picked_idx = int(pool_indices[0])
            else:
                picked_idx = int(rng.choice(pool_indices, size=1, replace=False)[0])

            selected_mask[picked_idx] = 1
            class_selected_counts[class_id] += 1
            class_selected_sum[class_id] += div_features_np[picked_idx]
            total_score_acc += float(np.max(combined_scores))
            selected_count_history.append(int(np.sum(selected_mask)))
            pbar.update(1)
            pbar.set_postfix(active_classes=int(active_classes.size))
    pbar.close()

    final_mask = selected_mask.astype(np.uint8)
    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        selected_by_class[class_id] = int(final_mask[class_indices].sum()) if class_indices.size > 0 else 0

    final_div_scores = np.asarray(
        div_metric.score_dataset_dynamic(
            div_loader,
            adapter=image_adapter,
            selected_mask=final_mask,
            image_features=div_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    selected_bool = final_mask.astype(bool)
    final_div_z = standard_zscore_by_class(final_div_scores, labels_np)
    subset_comprehensive_score = float(
        np.sum(
            (
                weights["sa"] * standard_zscore_by_class(sa_raw_np, labels_np)
                + weights["dds"] * standard_zscore_by_class(dds_raw_np, labels_np)
                + weights["div"] * final_div_z
            )[selected_bool],
            dtype=np.float64,
        )
    )

    class_shift_values: list[float] = []
    for class_id in range(num_classes):
        if class_selected_counts[class_id] <= 0:
            continue
        mu_sub = class_selected_sum[class_id] / float(class_selected_counts[class_id])
        mu_full = full_class_mean_f32[class_id]
        class_shift_values.append(float(np.linalg.norm(mu_sub - mu_full)))
    distribution_shift = float(np.mean(class_shift_values)) if class_shift_values else 0.0

    stats: dict[str, object] = {
        "solver": "group_classwise_greedy_add",
        "sr": float(sr),
        "dist_weight": float(dist_weight_max),
        "dist_weight_schedule": "linear_increase_by_class_progress",
        "dist_weight_max": float(dist_weight_max),
        "dist_weight_min": float(dist_weight_min),
        "noise_prior_ratio": float(NOISE_PRIOR_RATIO),
        "noise_risk_factor": float(NOISE_RISK_FACTOR),
        "base_dist_weight_max": float(base_dist_weight_max),
        "final_rate": float(final_mask.mean()),
        "selected_by_class": selected_by_class,
        "class_budgets": {int(c): int(v) for c, v in enumerate(class_budgets.tolist())},
        "init_per_class": {int(c): int(v) for c, v in enumerate(init_per_class.tolist())},
        "candidate_pool_size": int(candidate_pool_size),
        "selected_count_history": selected_count_history,
        "accumulated_greedy_score": float(total_score_acc),
        "subset_comprehensive_score": subset_comprehensive_score,
        "distribution_shift": distribution_shift,
    }
    return final_mask, selected_by_class, stats

def load_weights_for_stage(args: argparse.Namespace, seed: int) -> dict[str, float]:
    weights_path = WEIGHTS_ROOT / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, args.dataset)
    return load_scoring_weights(all_weights, args.weight_group, seed)


def resolve_mode_name(args: argparse.Namespace) -> str:
    if args.mode_name is not None and args.mode_name.strip():
        return args.mode_name.strip()
    return f"noise_{args.weight_group}_group"


def run_mask_stage(args: argparse.Namespace, seed: int, device: torch.device, image_path: Path, text_path: Path) -> None:
    keep_ratios = parse_ratio_list(args.kr)
    mode_name = resolve_mode_name(args)

    target_paths = [
        resolve_mask_path(
            mode=mode_name,
            dataset=args.dataset,
            model=args.model_name,
            seed=seed,
            keep_ratio=kr,
            root=MASK_ROOT,
        )
        for kr in keep_ratios
    ]
    if args.skip_saved and all(path.exists() for path in target_paths):
        print(f"[mask] skip all existing masks | dataset={args.dataset} seed={seed} kr={keep_ratios}")
        return

    print(
        f"[mask] compute noisy {args.weight_group}_group masks | "
        f"dataset={args.dataset} seed={seed} kr={keep_ratios}"
    )

    class_names = load_class_names_for_noisy_dataset(args.dataset)
    num_classes = len(class_names)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=str(DATA_ROOT),
        debug_prompts=args.debug_prompts,
    )

    dds_loader = build_noisy_score_loader(
        dds_metric.extractor.preprocess, args.dataset, seed, device, args.batch_size, args.num_workers
    )
    div_loader = build_noisy_score_loader(
        div_metric.extractor.preprocess, args.dataset, seed, device, args.batch_size, args.num_workers
    )
    sa_loader = build_noisy_score_loader(
        sa_metric.extractor.preprocess, args.dataset, seed, device, args.batch_size, args.num_workers
    )
    dataset_for_labels, clean_targets, noisy_targets, is_noisy = load_noisy_reference_dataset(
        args.dataset, seed, transform=None, verbose=True
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=seed,
        map_location=device,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    weights = load_weights_for_stage(args, seed)
    print(f"[mask] weights group={args.weight_group} seed={seed}: {weights}")

    num_samples = len(dataset_for_labels)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores

        return {
            "sa": np.asarray(sa_scores, dtype=np.float32),
            "div": np.asarray(div_scores, dtype=np.float32),
            "dds": np.asarray(dds_scores, dtype=np.float32),
            "labels": extract_labels(dataset_for_labels).astype(np.int64),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=STATIC_SCORE_ROOT,
        dataset=args.dataset,
        seed=seed,
        clip_model=args.clip_model,
        adapter_image_path=str(image_path),
        adapter_text_path=str(text_path),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)
    div_scores_np = np.asarray(static_scores["div"], dtype=np.float32)
    dds_scores_np = np.asarray(static_scores["dds"], dtype=np.float32)
    labels_np = np.asarray(static_scores["labels"], dtype=np.int64)

    if not np.array_equal(labels_np, noisy_targets.astype(np.int64)):
        raise RuntimeError("Static-score labels are not identical to noisy targets; abort to avoid invalid mask.")

    with patched_group_mean_cache():
        for kr, mask_path in zip(keep_ratios, target_paths):
            if args.skip_saved and mask_path.exists():
                print(f"[mask] skip existing: {mask_path}")
                continue

            mask, selected_by_class, group_stats = select_group_mask_local(
                sa_scores_np,
                div_metric=div_metric,
                div_loader=div_loader,
                image_adapter=image_adapter,
                labels=labels_np,
                weights=weights,
                num_classes=num_classes,
                keep_ratio=kr,
                device=device,
                dataset_name=args.dataset,
                seed=seed,
                weight_group=args.weight_group,
                clip_model=args.clip_model,
                adapter_image_path=str(image_path),
                div_static_scores=div_scores_np,
                dds_static_scores=dds_scores_np,
                group_candidate_pool_size=args.group_candidate_pool_size,
                group_init_count=args.group_init_count,
            )

            selected = mask.astype(bool)
            num_selected = int(mask.sum())
            num_noisy_selected = int(is_noisy[selected].sum())
            noise_ratio_in_mask = float(num_noisy_selected / max(1, num_selected))

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_path,
                mask=mask.astype(np.uint8),
                selected_indices=np.flatnonzero(mask).astype(np.int64),
                clean_targets=clean_targets.astype(np.int64),
                noisy_targets=noisy_targets.astype(np.int64),
                is_noisy=is_noisy.astype(np.uint8),
                weights=json.dumps(weights, ensure_ascii=False),
                selected_by_class=json.dumps(selected_by_class, ensure_ascii=False),
                group_stats=json.dumps(group_stats, ensure_ascii=False),
                dataset=np.array(args.dataset),
                method=np.array(mode_name),
                weight_group=np.array(args.weight_group),
                seed=np.array(seed, dtype=np.int64),
                keep_ratio=np.array(kr, dtype=np.int64),
                num_selected=np.array(num_selected, dtype=np.int64),
                num_noisy_total=np.array(int(is_noisy.sum()), dtype=np.int64),
                num_noisy_selected=np.array(num_noisy_selected, dtype=np.int64),
                noise_ratio_total=np.array(float(is_noisy.mean()), dtype=np.float32),
                noise_ratio_in_mask=np.array(noise_ratio_in_mask, dtype=np.float32),
            )

            summary_path = mask_path.with_suffix(".json")
            summary = {
                "dataset": args.dataset,
                "method": mode_name,
                "weight_group": args.weight_group,
                "seed": int(seed),
                "keep_ratio": int(kr),
                "num_selected": num_selected,
                "num_noisy_total": int(is_noisy.sum()),
                "num_noisy_selected": num_noisy_selected,
                "noise_ratio_total": float(is_noisy.mean()),
                "noise_ratio_in_mask": noise_ratio_in_mask,
                "weights": weights,
                "mask_path": str(mask_path),
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            print(
                f"[mask] saved {mask_path} | selected={num_selected} "
                f"noisy_selected={num_noisy_selected} "
                f"noise_ratio={noise_ratio_in_mask:.4f}"
            )


def run_one_seed(args: argparse.Namespace, seed: int, device: torch.device) -> None:
    start = time.perf_counter()
    print("=" * 100)
    print(
        f"[start] dataset={args.dataset} seed={seed} device={device} "
        f"weight_group={args.weight_group}"
    )
    print(f"[paths] data={DATA_ROOT} noise={NOISE_ROOT} noise_exp={NOISE_EXP_ROOT}")

    set_seed(seed)
    image_path, text_path = run_adapter_stage(args, seed, device)

    if args.weight_group == "learned":
        run_proxy_stage(args, seed, device)
        run_weight_learning_stage(args, seed, device, image_path, text_path)
    else:
        ensure_scoring_weights(WEIGHTS_ROOT / "scoring_weights.json", args.dataset)
        print("[weights] skip proxy and weight learning for weight_group=naive; use equal DDS/Div/SA weights.")

    run_mask_stage(args, seed, device, image_path, text_path)

    print(f"[done] dataset={args.dataset} seed={seed} elapsed={time.perf_counter() - start:.2f}s")
    print("=" * 100)


def validate_environment() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"data directory not found: {DATA_ROOT}")
    if not NOISE_ROOT.exists():
        raise FileNotFoundError(f"noise directory not found: {NOISE_ROOT}")

    NOISE_EXP_ROOT.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    PROXY_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DYNAMIC_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.strip().lower()
    args.weight_group = args.weight_group.strip().lower()
    validate_environment()

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("--seed cannot be empty.")

    print(
        f"[config] dataset={args.dataset} seeds={seeds} kr={parse_ratio_list(args.kr)} "
        f"weight_group={args.weight_group} mode={resolve_mode_name(args)}"
    )
    print(f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"[config] torch.cuda.is_available={torch.cuda.is_available()} device={device}")

    for seed in seeds:
        noise_path = NOISE_ROOT / args.dataset / f"noise_list_{seed}.txt"
        if not noise_path.is_file():
            raise FileNotFoundError(f"Noise file missing for seed={seed}: {noise_path}")
        run_one_seed(args, int(seed), device)


if __name__ == "__main__":
    main()
