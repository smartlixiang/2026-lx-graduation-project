#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare truncated dynamic-supervision windows on clean and special datasets.

The script runs three experiment variants:

1. normal:
   Standard CIFAR-100 / Tiny-ImageNet data and proxy logs under ``weights/``.
2. label_noise:
   Fixed label-noise data and proxy logs under ``noise_exp/``.
3. corruption:
   Fixed image-corruption data and proxy logs under ``corruption_exp/``.

Fixed settings
--------------
- random seed: 22
- keep ratio for learned_group: 50
- CIFAR-100 dynamic epochs: first 160 epochs
- Tiny-ImageNet dynamic epochs: first 60 epochs
- early/middle/late windows: 0.3 / 0.4 / 0.3, inherited from
  ``weights.dynamic_utils.resolve_epoch_windows``

Dynamic cache layout
--------------------
An experiment layer is inserted to prevent clean, label-noise and corruption
results from overwriting one another:

    yyy/dynamic_cache/normal/cifar100/160/A.npz
    yyy/dynamic_cache/label_noise/cifar100/160/noise_gate.npz
    yyy/dynamic_cache/corruption/tiny-imagenet/60/T.npz

The script never retrains proxy models and never saves learned weights or
selection masks. Existing static-score caches and trained adapters are reused;
missing static-score caches may be computed from the existing adapters.

Examples
--------
Run all datasets and all experiments:

    python yyy/try.py

Run only CIFAR-100:

    python yyy/try.py --dataset cifar100

Run only the label-noise and corruption variants on Tiny-ImageNet:

    python yyy/try.py --dataset tiny-imagenet --experiment special

Force recomputation of truncated dynamic caches:

    python yyy/try.py --force-dynamic
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
YYY_ROOT = THIS_FILE.parent
PROJECT_ROOT = YYY_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Project imports are intentionally placed after PROJECT_ROOT is inserted.
import calculate_my_mask as mask_mod  # noqa: E402
import learn_scoring_weights as weight_mod  # noqa: E402
from corruption_exp import cal_corruption_mask as corruption_mod  # noqa: E402
from dataset.dataset_config import CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters, resolve_adapter_paths  # noqa: E402
from noise_exp import cal_noise_mask as noise_mod  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.score_utils import standard_zscore_by_class  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from weights import (  # noqa: E402
    AbsorptionGainScore,
    ConfusionComplementarityScore,
    TransferabilityAlignmentScore,
)
from weights.dynamic_utils import (  # noqa: E402
    DynamicComponentResult,
    FoldLogData,
    resolve_epoch_windows,
)


ExperimentName = Literal["normal", "label_noise", "corruption"]

SEED = 22
KEEP_RATIO = 50
PROXY_MODEL = "resnet18"
SUPPORTED_DATASETS = (CIFAR100, TINY_IMAGENET)
EXPERIMENTS: tuple[ExperimentName, ...] = ("normal", "label_noise", "corruption")

DYNAMIC_EPOCHS = {
    CIFAR100: 160,
    TINY_IMAGENET: 60,
}
SOURCE_PROXY_EPOCHS = {
    CIFAR100: 200,
    TINY_IMAGENET: 90,
}
PROXY_LOG_ROOTS = {
    "normal": PROJECT_ROOT / "weights" / "proxy_logs",
    "label_noise": PROJECT_ROOT / "noise_exp" / "weights" / "proxy_logs",
    "corruption": PROJECT_ROOT / "corruption_exp" / "weights" / "proxy_logs",
}
DYNAMIC_CACHE_VERSION = "yyy_truncated_dynamic_v1"
COMPONENT_NAMES = ("A", "C", "T")


@dataclass
class ExperimentContext:
    experiment: ExperimentName
    dataset: str
    labels: np.ndarray
    dataset_factory: Callable[[object | None], object]
    adapter_image_path: Path
    adapter_text_path: Path
    static_cache_root: Path
    group_mean_cache_root: Path
    special_mask: np.ndarray | None = None
    special_label: str | None = None
    normal_label: str | None = None
    corruption_info: object | None = None


@dataclass
class StaticBundle:
    scores: dict[str, np.ndarray]
    labels: np.ndarray
    class_names: list[str]
    div_metric: Div
    div_loader: DataLoader
    image_adapter: torch.nn.Module
    adapter_image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test truncated dynamic windows for normal, label-noise and corruption experiments."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", CIFAR100, TINY_IMAGENET),
        help="Dataset to run. Default: both datasets.",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=("all", "special", *EXPERIMENTS),
        help="Experiment variant. 'special' means label_noise + corruption.",
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--proxy-model", default=PROXY_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument("--dds-k", type=int, default=5)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.8)

    parser.add_argument("--learn-window", type=int, default=5)
    parser.add_argument("--learn-min-correct", type=int, default=4)
    parser.add_argument(
        "--normal-gate-low",
        type=float,
        default=0.2,
        help="Normal-experiment gate lower threshold; matches learn_scoring_weights.py.",
    )
    parser.add_argument("--normal-gate-high", type=float, default=0.8)
    parser.add_argument(
        "--special-gate-low",
        type=float,
        default=0.2,
        help="Label-noise/corruption lower threshold; matches the special-experiment scripts.",
    )
    parser.add_argument("--special-gate-high", type=float, default=0.8)

    parser.add_argument("--ratio-lambda", type=float, default=5e-3)
    parser.add_argument("--regression-learning-rate", type=float, default=2e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-8)

    parser.add_argument(
        "--group-candidate-pool-size",
        type=int,
        default=None,
        help="Override experiment defaults. Noise default=1; corruption default=5.",
    )
    parser.add_argument(
        "--group-init-count",
        type=int,
        default=None,
        help="Override experiment defaults. Noise default=2; corruption default=10.",
    )
    parser.add_argument("--hist-bins", type=int, default=60)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument(
        "--force-dynamic",
        action="store_true",
        help="Ignore yyy dynamic caches and recompute A/C/T/noise_gate.",
    )
    return parser.parse_args()


def selected_datasets(name: str) -> list[str]:
    return list(SUPPORTED_DATASETS) if name == "all" else [name]


def selected_experiments(name: str) -> list[ExperimentName]:
    if name == "all":
        return list(EXPERIMENTS)
    if name == "special":
        return ["label_noise", "corruption"]
    return [name]  # type: ignore[list-item]


def extract_labels(dataset: object) -> np.ndarray:
    num_samples = len(dataset)  # type: ignore[arg-type]
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if len(values) == num_samples:
                return np.asarray(values, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([label for _, label in getattr(dataset, "samples")], dtype=np.int64)
    return np.asarray([int(dataset[index][1]) for index in range(num_samples)], dtype=np.int64)  # type: ignore[index]


def build_clean_dataset(dataset_name: str, transform=None):
    return weight_mod._build_dataset(dataset_name, str(DATA_ROOT), transform)


def build_context(experiment: ExperimentName, dataset_name: str) -> ExperimentContext:
    if experiment == "normal":
        reference = build_clean_dataset(dataset_name, transform=None)
        labels = extract_labels(reference)
        image_path, text_path = resolve_adapter_paths(dataset_name, SEED)
        return ExperimentContext(
            experiment=experiment,
            dataset=dataset_name,
            labels=labels,
            dataset_factory=lambda transform: build_clean_dataset(dataset_name, transform),
            adapter_image_path=image_path,
            adapter_text_path=text_path,
            static_cache_root=PROJECT_ROOT / "static_scores",
            group_mean_cache_root=PROJECT_ROOT / "static_scores" / "group_mean_stats",
        )

    if experiment == "label_noise":
        reference, _, noisy_targets, is_noisy = noise_mod.load_noisy_reference_dataset(
            dataset_name, SEED, transform=None, verbose=True
        )
        image_path, text_path = noise_mod.noise_adapter_paths(dataset_name, SEED)
        return ExperimentContext(
            experiment=experiment,
            dataset=dataset_name,
            labels=np.asarray(noisy_targets, dtype=np.int64),
            dataset_factory=lambda transform: noise_mod.load_noisy_reference_dataset(
                dataset_name, SEED, transform=transform, verbose=False
            )[0],
            adapter_image_path=image_path,
            adapter_text_path=text_path,
            static_cache_root=noise_mod.STATIC_SCORE_ROOT,
            group_mean_cache_root=noise_mod.STATIC_SCORE_ROOT / "group_mean_stats",
            special_mask=np.asarray(is_noisy, dtype=bool),
            special_label="Noisy-label samples",
            normal_label="Unmodified-label samples",
        )

    raw = corruption_mod.build_raw_train_dataset(dataset_name)
    labels = corruption_mod.extract_labels(raw).astype(np.int64)
    info = corruption_mod.load_corruption_info(
        dataset_name,
        SEED,
        num_samples=len(raw),
        strict_expected_size=True,
    )
    image_path, text_path = corruption_mod.adapter_paths(dataset_name, SEED)

    def build_corrupted(transform):
        raw_dataset = corruption_mod.build_raw_train_dataset(dataset_name)
        return corruption_mod.FixedCorruptionDataset(
            raw_dataset,
            transform=transform,
            target_transform=None,
            corruption_info=info,
        )

    # Match corruption_exp/cal_corruption_mask.py's existing static-score cache root.
    static_root = (
        corruption_mod.STATIC_SCORE_ROOT
        / dataset_name
        / str(SEED)
        / f"corruption_{info.list_hash[:12]}"
    )
    return ExperimentContext(
        experiment=experiment,
        dataset=dataset_name,
        labels=labels,
        dataset_factory=build_corrupted,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
        static_cache_root=static_root,
        group_mean_cache_root=corruption_mod.STATIC_SCORE_ROOT / "group_mean_stats",
        special_mask=np.asarray(info.is_corrupted, dtype=bool),
        special_label="Corrupted images",
        normal_label="Uncorrupted images",
        corruption_info=info,
    )


def validate_required_adapters(ctx: ExperimentContext) -> None:
    missing = [
        path
        for path in (ctx.adapter_image_path, ctx.adapter_text_path)
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required trained adapter files are missing. This script does not retrain adapters:\n"
            f"{formatted}"
        )


def find_proxy_log_dir(
    experiment: ExperimentName,
    dataset_name: str,
    proxy_model: str,
    cutoff: int,
) -> Path:
    root = PROXY_LOG_ROOTS[experiment]
    seed_dir = root / dataset_name / proxy_model / str(SEED)
    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"Proxy-log seed directory not found: {seed_dir}\n"
            "This script only reuses existing proxy logs and will not train a proxy model."
        )

    preferred = seed_dir / str(SOURCE_PROXY_EPOCHS[dataset_name])
    if preferred.is_dir() and any(preferred.glob("fold_*.npz")):
        return preferred

    candidates = [
        path
        for path in seed_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and int(path.name) >= cutoff
        and any(path.glob("fold_*.npz"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No proxy-log directory with at least {cutoff} epochs under {seed_dir}."
        )
    return max(candidates, key=lambda path: int(path.name))


def fold_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.removeprefix("fold_")
    return (int(suffix) if suffix.isdigit() else 10**9, path.name)


def load_truncated_cv_logs(
    log_dir: Path,
    labels_all: np.ndarray,
    cutoff: int,
) -> list[FoldLogData]:
    """Load existing fold logs while retaining only epochs ``[:cutoff]``.

    Each NPZ file is opened separately. The retained prefix is copied so the
    full source array can be released before the next fold is loaded.
    """
    fold_paths = sorted(log_dir.glob("fold_*.npz"), key=fold_sort_key)
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {log_dir}")

    num_samples = int(labels_all.shape[0])
    seen_val = np.zeros(num_samples, dtype=np.int64)
    folds: list[FoldLogData] = []

    for fold_id, path in enumerate(
        tqdm(fold_paths, desc="Loading truncated proxy folds", unit="fold")
    ):
        with np.load(path, allow_pickle=False) as data:
            required = {"train_indices", "val_indices", "train_logits", "val_logits"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{path} missing keys: {sorted(missing)}")

            train_indices = np.asarray(data["train_indices"], dtype=np.int64)
            val_indices = np.asarray(data["val_indices"], dtype=np.int64)
            train_source = data["train_logits"]
            val_source = data["val_logits"]

            if train_source.ndim != 3 or val_source.ndim != 3:
                raise ValueError(f"Logits in {path} must be three-dimensional.")
            available = min(int(train_source.shape[0]), int(val_source.shape[0]))
            if available < cutoff:
                raise ValueError(
                    f"{path} contains only {available} epochs, below cutoff={cutoff}."
                )
            if train_source.shape[1] != train_indices.size:
                raise ValueError(f"train_logits sample dimension mismatch: {path}")
            if val_source.shape[1] != val_indices.size:
                raise ValueError(f"val_logits sample dimension mismatch: {path}")
            if train_source.shape[2] != val_source.shape[2]:
                raise ValueError(f"train/val class dimension mismatch: {path}")

            # Copy the prefix to detach it from the full source ndarray.
            train_logits = np.array(
                train_source[:cutoff], dtype=np.float32, copy=True
            )
            val_logits = np.array(
                val_source[:cutoff], dtype=np.float32, copy=True
            )
            del train_source, val_source

        if np.any(train_indices < 0) or np.any(train_indices >= num_samples):
            raise ValueError(f"train_indices out of range: {path}")
        if np.any(val_indices < 0) or np.any(val_indices >= num_samples):
            raise ValueError(f"val_indices out of range: {path}")
        seen_val[val_indices] += 1

        folds.append(
            FoldLogData(
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                train_logits=train_logits,
                val_logits=val_logits,
            )
        )
        gc.collect()

    if not np.all(seen_val == 1):
        bad = np.flatnonzero(seen_val != 1)
        raise ValueError(
            "Each sample must occur in exactly one validation fold; "
            f"violations={bad[:10].tolist()}."
        )
    return folds


def dynamic_cache_dir(ctx: ExperimentContext, cutoff: int) -> Path:
    return (
        YYY_ROOT
        / "dynamic_cache"
        / ctx.experiment
        / ctx.dataset
        / str(int(cutoff))
    )


def component_cache_path(
    ctx: ExperimentContext,
    cutoff: int,
    component_name: str,
) -> Path:
    return dynamic_cache_dir(ctx, cutoff) / f"{component_name.upper()}.npz"


def gate_cache_path(ctx: ExperimentContext, cutoff: int) -> Path:
    return dynamic_cache_dir(ctx, cutoff) / "noise_gate.npz"


def load_component_cache(
    path: Path,
    *,
    ctx: ExperimentContext,
    cutoff: int,
    source_log_dir: Path,
) -> DynamicComponentResult | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "cache_version",
                "experiment",
                "dataset",
                "seed",
                "epochs",
                "source_log_dir",
                "labels",
                "raw_foldwise",
                "fold_normalized",
                "aggregated",
                "final_normalized",
            }
            if not required.issubset(set(data.files)):
                return None
            if str(np.asarray(data["cache_version"]).item()) != DYNAMIC_CACHE_VERSION:
                return None
            if str(np.asarray(data["experiment"]).item()) != ctx.experiment:
                return None
            if str(np.asarray(data["dataset"]).item()) != ctx.dataset:
                return None
            if int(np.asarray(data["seed"]).item()) != SEED:
                return None
            if int(np.asarray(data["epochs"]).item()) != cutoff:
                return None
            if str(np.asarray(data["source_log_dir"]).item()) != str(source_log_dir.resolve()):
                return None
            if not np.array_equal(
                np.asarray(data["labels"], dtype=np.int64), ctx.labels
            ):
                return None

            raw_foldwise = np.asarray(data["raw_foldwise"], dtype=np.float32)
            fold_normalized = np.asarray(data["fold_normalized"], dtype=np.float32)
            aggregated = np.asarray(data["aggregated"], dtype=np.float32)
            final_normalized = np.asarray(data["final_normalized"], dtype=np.float32)
    except Exception:
        return None

    num_samples = int(ctx.labels.size)
    if raw_foldwise.ndim != 2 or raw_foldwise.shape[1] != num_samples:
        return None
    if fold_normalized.shape != raw_foldwise.shape:
        return None
    if aggregated.shape != (num_samples,) or final_normalized.shape != (num_samples,):
        return None
    if not np.all(np.isfinite(aggregated)):
        return None
    if not np.all(np.isfinite(final_normalized)):
        return None

    return DynamicComponentResult(
        raw_foldwise=raw_foldwise,
        fold_normalized=fold_normalized,
        aggregated=aggregated,
        final_normalized=final_normalized,
    )


def save_component_cache(
    path: Path,
    *,
    ctx: ExperimentContext,
    cutoff: int,
    source_log_dir: Path,
    component_name: str,
    result: DynamicComponentResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.array(DYNAMIC_CACHE_VERSION, dtype=np.str_),
        experiment=np.array(ctx.experiment, dtype=np.str_),
        component_name=np.array(component_name, dtype=np.str_),
        dataset=np.array(ctx.dataset, dtype=np.str_),
        seed=np.array(SEED, dtype=np.int64),
        epochs=np.array(cutoff, dtype=np.int64),
        source_log_dir=np.array(str(source_log_dir.resolve()), dtype=np.str_),
        labels=ctx.labels.astype(np.int64),
        raw_foldwise=result.raw_foldwise.astype(np.float32),
        fold_normalized=result.fold_normalized.astype(np.float32),
        aggregated=result.aggregated.astype(np.float32),
        final_normalized=result.final_normalized.astype(np.float32),
    )


def gate_thresholds(
    ctx: ExperimentContext,
    args: argparse.Namespace,
) -> tuple[float, float]:
    if ctx.experiment == "normal":
        return float(args.normal_gate_low), float(args.normal_gate_high)
    return float(args.special_gate_low), float(args.special_gate_high)


def load_gate_cache(
    path: Path,
    *,
    ctx: ExperimentContext,
    cutoff: int,
    source_log_dir: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "cache_version",
                "noise_gate_version",
                "experiment",
                "dataset",
                "seed",
                "epochs",
                "source_log_dir",
                "labels",
                "learn_window",
                "learn_min_correct",
                "final_risk",
            }
            if not required.issubset(set(data.files)):
                return None
            if str(np.asarray(data["cache_version"]).item()) != DYNAMIC_CACHE_VERSION:
                return None
            if str(np.asarray(data["noise_gate_version"]).item()) != weight_mod.NOISE_GATE_CACHE_VERSION:
                return None
            if str(np.asarray(data["experiment"]).item()) != ctx.experiment:
                return None
            if str(np.asarray(data["dataset"]).item()) != ctx.dataset:
                return None
            if int(np.asarray(data["seed"]).item()) != SEED:
                return None
            if int(np.asarray(data["epochs"]).item()) != cutoff:
                return None
            if str(np.asarray(data["source_log_dir"]).item()) != str(source_log_dir.resolve()):
                return None
            if int(np.asarray(data["learn_window"]).item()) != int(args.learn_window):
                return None
            if int(np.asarray(data["learn_min_correct"]).item()) != int(args.learn_min_correct):
                return None
            if not np.array_equal(
                np.asarray(data["labels"], dtype=np.int64), ctx.labels
            ):
                return None
            final_risk = np.asarray(data["final_risk"], dtype=np.float64)
    except Exception:
        return None

    if final_risk.shape != (ctx.labels.size,) or not np.all(np.isfinite(final_risk)):
        return None
    gate_low, gate_high = gate_thresholds(ctx, args)
    gate = weight_mod._build_gate_from_final_risk(
        final_risk, gate_low, gate_high
    )
    return {"final_risk": final_risk, "gate": gate}


def save_gate_cache(
    path: Path,
    *,
    ctx: ExperimentContext,
    cutoff: int,
    source_log_dir: Path,
    args: argparse.Namespace,
    final_risk: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.array(DYNAMIC_CACHE_VERSION, dtype=np.str_),
        noise_gate_version=np.array(weight_mod.NOISE_GATE_CACHE_VERSION, dtype=np.str_),
        experiment=np.array(ctx.experiment, dtype=np.str_),
        dataset=np.array(ctx.dataset, dtype=np.str_),
        seed=np.array(SEED, dtype=np.int64),
        epochs=np.array(cutoff, dtype=np.int64),
        source_log_dir=np.array(str(source_log_dir.resolve()), dtype=np.str_),
        labels=ctx.labels.astype(np.int64),
        learn_window=np.array(int(args.learn_window), dtype=np.int64),
        learn_min_correct=np.array(int(args.learn_min_correct), dtype=np.int64),
        final_risk=np.asarray(final_risk, dtype=np.float64),
    )


def dynamic_component_computers():
    return {
        "A": AbsorptionGainScore(),
        "C": ConfusionComplementarityScore(),
        "T": TransferabilityAlignmentScore(),
    }


def load_or_compute_dynamic_supervision(
    ctx: ExperimentContext,
    args: argparse.Namespace,
) -> tuple[dict[str, DynamicComponentResult], dict[str, np.ndarray], np.ndarray]:
    cutoff = DYNAMIC_EPOCHS[ctx.dataset]
    source_log_dir = find_proxy_log_dir(
        ctx.experiment, ctx.dataset, args.proxy_model, cutoff
    )

    early, middle, late = resolve_epoch_windows(cutoff)
    tqdm.write(
        f"[dynamic] experiment={ctx.experiment} dataset={ctx.dataset} "
        f"source={source_log_dir} cutoff={cutoff} "
        f"windows=1-{early[-1] + 1},"
        f"{middle[0] + 1}-{middle[-1] + 1},"
        f"{late[0] + 1}-{late[-1] + 1}"
    )

    component_results: dict[str, DynamicComponentResult] = {}
    missing_components: list[str] = []

    for name in COMPONENT_NAMES:
        path = component_cache_path(ctx, cutoff, name)
        result = None
        if not args.force_dynamic:
            result = load_component_cache(
                path,
                ctx=ctx,
                cutoff=cutoff,
                source_log_dir=source_log_dir,
            )
        if result is None:
            missing_components.append(name)
            tqdm.write(f"[dynamic] cache MISS: {path}")
        else:
            component_results[name] = result
            tqdm.write(f"[dynamic] cache HIT: {path}")

    gate_path = gate_cache_path(ctx, cutoff)
    gate_data = None
    if not args.force_dynamic:
        gate_data = load_gate_cache(
            gate_path,
            ctx=ctx,
            cutoff=cutoff,
            source_log_dir=source_log_dir,
            args=args,
        )
    if gate_data is None:
        tqdm.write(f"[dynamic] cache MISS: {gate_path}")
    else:
        tqdm.write(f"[dynamic] cache HIT: {gate_path}")

    if missing_components or gate_data is None:
        # Label-noise proxy logits must be interpreted using the noisy labels.
        patch_context = (
            noise_mod.patched_training_label_noise(
                ctx.dataset, SEED, verbose_once=False
            )
            if ctx.experiment == "label_noise"
            else contextlib.nullcontext()
        )
        with patch_context:
            folds = load_truncated_cv_logs(
                source_log_dir,
                labels_all=ctx.labels,
                cutoff=cutoff,
            )

        computers = dynamic_component_computers()
        for name in tqdm(
            missing_components,
            desc=f"{ctx.experiment}/{ctx.dataset} dynamic components",
            unit="component",
        ):
            result = computers[name].compute(folds=folds, labels_all=ctx.labels)
            component_results[name] = result
            path = component_cache_path(ctx, cutoff, name)
            save_component_cache(
                path,
                ctx=ctx,
                cutoff=cutoff,
                source_log_dir=source_log_dir,
                component_name=name,
                result=result,
            )
            tqdm.write(f"[dynamic] saved: {path}")

        if gate_data is None:
            final_risk = weight_mod._compute_final_noise_risk(
                folds,
                ctx.labels,
                learn_window=int(args.learn_window),
                learn_min_correct=int(args.learn_min_correct),
            )
            gate_low, gate_high = gate_thresholds(ctx, args)
            gate = weight_mod._build_gate_from_final_risk(
                final_risk, gate_low, gate_high
            )
            gate_data = {"final_risk": final_risk, "gate": gate}
            save_gate_cache(
                gate_path,
                ctx=ctx,
                cutoff=cutoff,
                source_log_dir=source_log_dir,
                args=args,
                final_risk=final_risk,
            )
            tqdm.write(f"[dynamic] saved: {gate_path}")

        del folds
        gc.collect()

    if set(component_results) != set(COMPONENT_NAMES):
        raise RuntimeError(
            f"Incomplete dynamic components: {sorted(component_results)}"
        )
    if gate_data is None:
        raise RuntimeError("Noise gate was not loaded or computed.")

    pseudo_label, _ = weight_mod.build_dynamic_target(
        component_results,
        gate=gate_data["gate"],
        use_noise_gate=True,
    )
    return component_results, gate_data, pseudo_label


def class_names_for_context(ctx: ExperimentContext) -> list[str]:
    raw = build_clean_dataset(ctx.dataset, transform=None)
    return list(
        resolve_class_names_for_prompts(
            dataset_name=ctx.dataset,
            data_root=str(DATA_ROOT),
            class_names=raw.classes,
        )
    )


def make_loader(
    ctx: ExperimentContext,
    preprocess,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = ctx.dataset_factory(preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def build_static_bundle(
    ctx: ExperimentContext,
    args: argparse.Namespace,
    device: torch.device,
) -> StaticBundle:
    validate_required_adapters(ctx)
    class_names = class_names_for_context(ctx)

    dds_metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=int(args.dds_k),
        important_eigval_ratio=float(args.dds_important_eigval_ratio),
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=float(args.div_k),
    )
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=ctx.dataset,
        data_root=str(DATA_ROOT),
        debug_prompts=bool(args.debug_prompts),
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=ctx.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=SEED,
        map_location=device,
        adapter_image_path=ctx.adapter_image_path,
        adapter_text_path=ctx.adapter_text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    div_loader = make_loader(
        ctx,
        div_metric.extractor.preprocess,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    def compute_scores() -> dict[str, np.ndarray]:
        dds_loader = make_loader(
            ctx,
            dds_metric.extractor.preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        sa_loader = make_loader(
            ctx,
            sa_metric.extractor.preprocess,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dds_scores = dds_metric.score_dataset(
            tqdm(dds_loader, desc=f"{ctx.experiment} Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores = div_metric.score_dataset(
            tqdm(div_loader, desc=f"{ctx.experiment} Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores = sa_metric.score_dataset(
            tqdm(sa_loader, desc=f"{ctx.experiment} Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {
            "sa": np.asarray(sa_scores, dtype=np.float32),
            "div": np.asarray(div_scores, dtype=np.float32),
            "dds": np.asarray(dds_scores, dtype=np.float32),
            "labels": ctx.labels.astype(np.int64),
        }

    scores = get_or_compute_static_scores(
        cache_root=ctx.static_cache_root,
        dataset=ctx.dataset,
        seed=SEED,
        clip_model=args.clip_model,
        adapter_image_path=str(ctx.adapter_image_path),
        adapter_text_path=str(ctx.adapter_text_path),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=int(ctx.labels.size),
        compute_fn=compute_scores,
    )
    labels = np.asarray(scores["labels"], dtype=np.int64)
    if not np.array_equal(labels, ctx.labels):
        raise RuntimeError(
            f"Static-score labels do not match {ctx.experiment} reference labels."
        )

    return StaticBundle(
        scores={
            "sa": np.asarray(scores["sa"], dtype=np.float32),
            "div": np.asarray(scores["div"], dtype=np.float32),
            "dds": np.asarray(scores["dds"], dtype=np.float32),
            "labels": labels,
        },
        labels=labels,
        class_names=class_names,
        div_metric=div_metric,
        div_loader=div_loader,
        image_adapter=image_adapter,
        adapter_image_path=ctx.adapter_image_path,
    )


def fit_weights(
    ctx: ExperimentContext,
    static: StaticBundle,
    pseudo_label: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, object]]:
    sa_z = standard_zscore_by_class(static.scores["sa"], static.labels)
    div_z = standard_zscore_by_class(static.scores["div"], static.labels)
    dds_z = standard_zscore_by_class(static.scores["dds"], static.labels)
    features = np.stack([sa_z, div_z, dds_z], axis=1).astype(np.float64)

    ratio_lambda = float(args.ratio_lambda)
    if ctx.experiment == "label_noise":
        # Match noise_exp/cal_noise_mask.py:
        # ratio_lambda *= (1 - sqrt(noise_prior))^2.
        ratio_lambda *= float(noise_mod.NOISE_RISK_FACTOR) ** 2

    fit = weight_mod.fit_softplus_ratio_regression(
        features,
        np.asarray(pseudo_label, dtype=np.float64),
        ratio_lambda=ratio_lambda,
        learning_rate=float(args.regression_learning_rate),
        max_iter=int(args.regression_max_iter),
        tol=float(args.regression_tol),
        device=device,
    )
    normalized = np.asarray(fit["normalized_weights"], dtype=np.float64)
    weights = {
        "sa": float(normalized[0]),
        "div": float(normalized[1]),
        "dds": float(normalized[2]),
    }
    tqdm.write(
        f"[weights] experiment={ctx.experiment} dataset={ctx.dataset} seed={SEED} "
        f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
        f"DDS={weights['dds']:.6f}, bias={float(fit['bias']):.6f}, "
        f"MSE={float(fit['mse']):.6f}, ratio_lambda={ratio_lambda:.8g}"
    )
    return weights, fit


def plot_special_distributions(
    ctx: ExperimentContext,
    pseudo_label: np.ndarray,
    gate: np.ndarray,
    bins: int,
) -> Path:
    if ctx.special_mask is None:
        raise ValueError("Special-sample mask is required for histograms.")

    special = np.asarray(ctx.special_mask, dtype=bool)
    normal = ~special
    output_dir = (
        YYY_ROOT
        / "histograms"
        / ctx.experiment
        / ctx.dataset
        / str(DYNAMIC_EPOCHS[ctx.dataset])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pseudo_label_and_noise_gate.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(
        np.asarray(pseudo_label)[normal],
        bins=bins,
        density=True,
        alpha=0.65,
        color="tab:blue",
        label=ctx.normal_label,
    )
    axes[0].hist(
        np.asarray(pseudo_label)[special],
        bins=bins,
        density=True,
        alpha=0.65,
        color="tab:orange",
        label=ctx.special_label,
    )
    axes[0].set_title("Dynamic pseudo-label distribution")
    axes[0].set_xlabel("Pseudo-label")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].hist(
        np.asarray(gate)[normal],
        bins=bins,
        density=True,
        alpha=0.65,
        color="tab:blue",
        label=ctx.normal_label,
    )
    axes[1].hist(
        np.asarray(gate)[special],
        bins=bins,
        density=True,
        alpha=0.65,
        color="tab:orange",
        label=ctx.special_label,
    )
    axes[1].set_title("Noise-gate distribution")
    axes[1].set_xlabel("Gate value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"{ctx.experiment} | {ctx.dataset} | seed={SEED} | "
        f"epochs={DYNAMIC_EPOCHS[ctx.dataset]}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    tqdm.write(f"[histogram] saved: {output_path}")
    return output_path


def sha1_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@contextlib.contextmanager
def patched_group_mean_cache(ctx: ExperimentContext) -> Iterator[None]:
    """Place group full-class-mean caches in the matching experiment tree."""
    old_resolver = mask_mod._mean_stats_cache_path

    def resolver(
        dataset_name: str,
        clip_model: str,
        adapter_image_path: str,
    ) -> Path:
        clip_tag = clip_model.replace("/", "-").replace(" ", "_")
        adapter_hash = sha1_file(Path(adapter_image_path))
        return (
            ctx.group_mean_cache_root
            / dataset_name
            / clip_tag
            / f"img_adapter_{adapter_hash}.npz"
        )

    mask_mod._mean_stats_cache_path = resolver
    try:
        yield
    finally:
        mask_mod._mean_stats_cache_path = old_resolver


def group_defaults(
    ctx: ExperimentContext,
    args: argparse.Namespace,
) -> tuple[int, int, float]:
    if ctx.experiment == "label_noise":
        pool_size = (
            int(args.group_candidate_pool_size)
            if args.group_candidate_pool_size is not None
            else 1
        )
        init_count = (
            int(args.group_init_count)
            if args.group_init_count is not None
            else 2
        )
        dist_factor = float(noise_mod.NOISE_RISK_FACTOR)
        return pool_size, init_count, dist_factor

    if ctx.experiment == "corruption":
        pool_size = (
            int(args.group_candidate_pool_size)
            if args.group_candidate_pool_size is not None
            else 5
        )
        init_count = (
            int(args.group_init_count)
            if args.group_init_count is not None
            else 10
        )
        dist_factor = float(corruption_mod.CORRUPTION_RISK_FACTOR)
        return pool_size, init_count, dist_factor

    raise ValueError("learned_group mask is requested only for special experiments.")


def compute_in_memory_learned_group_mask(
    ctx: ExperimentContext,
    static: StaticBundle,
    weights: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    if ctx.special_mask is None:
        raise ValueError("Special-sample mask is required.")

    pool_size, init_count, dist_factor = group_defaults(ctx, args)
    with patched_group_mean_cache(ctx):
        mask, _, stats = mask_mod.select_group_mask(
            np.asarray(static.scores["sa"], dtype=np.float32),
            div_metric=static.div_metric,
            div_loader=static.div_loader,
            image_adapter=static.image_adapter,
            labels=static.labels,
            weights=weights,
            num_classes=len(static.class_names),
            keep_ratio=KEEP_RATIO,
            device=device,
            dataset_name=ctx.dataset,
            seed=SEED,
            weight_group="learned",
            clip_model=args.clip_model,
            adapter_image_path=str(static.adapter_image_path),
            div_static_scores=np.asarray(static.scores["div"], dtype=np.float32),
            dds_static_scores=np.asarray(static.scores["dds"], dtype=np.float32),
            group_candidate_pool_size=pool_size,
            group_init_count=init_count,
            dist_weight_factor=dist_factor,
        )

    mask = np.asarray(mask, dtype=np.uint8)
    expected = int(round(ctx.labels.size * KEEP_RATIO / 100.0))
    if mask.shape != (ctx.labels.size,):
        raise RuntimeError(f"Mask shape mismatch: {mask.shape}")
    if int(mask.sum()) != expected:
        raise RuntimeError(
            f"Mask selected {int(mask.sum())} samples; expected {expected}."
        )

    selected = mask.astype(bool)
    special = np.asarray(ctx.special_mask, dtype=bool)
    num_selected = int(selected.sum())
    num_special = int(special[selected].sum())
    ratio = float(num_special / max(1, num_selected))
    total_ratio = float(special.mean())

    kind = "noisy-label" if ctx.experiment == "label_noise" else "corrupted-image"
    tqdm.write(
        f"[mask] experiment={ctx.experiment} dataset={ctx.dataset} "
        f"learned_group kr={KEEP_RATIO} selected={num_selected} "
        f"{kind}_selected={num_special} ratio={ratio:.6f} "
        f"(dataset baseline={total_ratio:.6f}) "
        f"dist_weight_factor={dist_factor:.6f}; mask not saved"
    )
    stats = dict(stats)
    stats.update(
        {
            "num_selected": num_selected,
            "num_special_selected": num_special,
            "special_ratio_in_mask": ratio,
            "special_ratio_total": total_ratio,
            "dist_weight_factor": dist_factor,
        }
    )
    return mask, stats


def run_task(
    experiment: ExperimentName,
    dataset_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    set_seed(SEED)
    ctx = build_context(experiment, dataset_name)
    stage_count = 5 if experiment != "normal" else 3

    with tqdm(
        total=stage_count,
        desc=f"{experiment}/{dataset_name}",
        unit="stage",
        leave=True,
    ) as stages:
        _, gate_data, pseudo_label = load_or_compute_dynamic_supervision(
            ctx, args
        )
        stages.update(1)

        if experiment != "normal":
            plot_special_distributions(
                ctx,
                pseudo_label=pseudo_label,
                gate=gate_data["gate"],
                bins=int(args.hist_bins),
            )
            stages.update(1)

        static = build_static_bundle(ctx, args, device)
        stages.update(1)

        weights, _ = fit_weights(
            ctx,
            static=static,
            pseudo_label=pseudo_label,
            args=args,
            device=device,
        )
        stages.update(1)

        if experiment != "normal":
            compute_in_memory_learned_group_mask(
                ctx,
                static=static,
                weights=weights,
                args=args,
                device=device,
            )
            stages.update(1)

    del static
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if int(args.hist_bins) <= 1:
        raise ValueError("--hist-bins must be greater than 1.")
    if int(args.learn_window) <= 0:
        raise ValueError("--learn-window must be positive.")
    if not (0 < int(args.learn_min_correct) <= int(args.learn_window)):
        raise ValueError("--learn-min-correct must lie in [1, learn-window].")

    device = torch.device(args.device) if args.device else CONFIG.global_device
    datasets_to_run = selected_datasets(args.dataset)
    experiments_to_run = selected_experiments(args.experiment)
    tasks = [
        (experiment, dataset_name)
        for dataset_name in datasets_to_run
        for experiment in experiments_to_run
    ]

    print("=" * 100)
    print(
        f"yyy truncated-dynamics experiment | seed={SEED} | kr={KEEP_RATIO} | "
        f"device={device}"
    )
    print(f"datasets={datasets_to_run}")
    print(f"experiments={experiments_to_run}")
    print(f"dynamic epochs={DYNAMIC_EPOCHS}")
    print("=" * 100)

    with tqdm(total=len(tasks), desc="All experiment tasks", unit="task") as overall:
        for experiment, dataset_name in tasks:
            tqdm.write("\n" + "=" * 100)
            tqdm.write(
                f"[start] experiment={experiment} dataset={dataset_name} "
                f"seed={SEED} cutoff={DYNAMIC_EPOCHS[dataset_name]}"
            )
            run_task(experiment, dataset_name, args, device)
            overall.update(1)

    print("=" * 100)
    print("All requested tasks finished.")
    print(f"Dynamic caches: {YYY_ROOT / 'dynamic_cache'}")
    print(f"Histograms: {YYY_ROOT / 'histograms'}")
    print("Learned weights and learned_group masks were not saved.")
    print("=" * 100)


if __name__ == "__main__":
    main()