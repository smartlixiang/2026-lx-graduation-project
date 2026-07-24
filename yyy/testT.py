#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test a validation-dominant replacement for dynamic component T.

This script is self-contained with respect to ``yyy/``: it imports only the
formal project modules from the repository root and does not depend on any
other Python file under ``yyy``.

New T definition
----------------
For each CV fold f and class c, let L[f,c,t] be the mean validation
cross-entropy loss at epoch t. Using the project's early/middle/late windows:

    G[f,c] = mean_early(L[f,c]) - mean_middle(L[f,c])
    S[f,c] = std_middle(L[f,c]) - std_late(L[f,c])

For sample i with class c and held-out fold h(i):

    G_in(i)  = mean_{f != h(i)} G[f,c]
    S_in(i)  = mean_{f != h(i)} S[f,c]

In the held-out fold, the sample itself is removed from the class validation
loss trajectory before computing G_out(i) and S_out(i). The inclusion effects
are:

    D_G(i) = G_in(i) - G_out(i)
    D_S(i) = S_in(i) - S_out(i)

The final component is:

    T(i) = z((z(D_G(i)) + z(D_S(i))) / 2)

No new tunable hyperparameter is introduced.

Behavior
--------
- normal: learn SA/Div/DDS weights in memory; do not save weights or masks;
- corruption: learn weights, calculate the current center-repair group mask in
  memory, and print retained counts for five corruption types;
- A/C and static SA/Div/DDS must already exist in the corresponding formal
  experiment caches; this script never recomputes them;
- only the new T cache is written, under ``yyy/test_T_cache``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
YYY_ROOT = THIS_FILE.parent
PROJECT_ROOT = YYY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import calculate_my_mask as mask_mod  # noqa: E402
import learn_scoring_weights as learn_mod  # noqa: E402
from corruption_exp import cal_corruption_mask as corr_mod  # noqa: E402
from dataset.dataset_config import CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.proxy_log_utils import (  # noqa: E402
    compute_loss_from_logits,
    resolve_proxy_log_path,
)
from utils.score_utils import standard_zscore, standard_zscore_by_class  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import (  # noqa: E402
    NORMALIZATION_VERSION,
    _hash_file,
    _validate_metric_cache,
)
from utils.training_defaults import get_proxy_training_config  # noqa: E402
from weights.dynamic_utils import (  # noqa: E402
    DynamicComponentResult,
    FoldLogData,
    load_cv_fold_logs,
    resolve_epoch_windows,
    safe_standardize,
    standard_zscore_dynamic,
)

Experiment = Literal["normal", "corruption"]
SUPPORTED_DATASETS = (CIFAR100, TINY_IMAGENET)
CORRUPTION_NAMES = (
    "gaussian_noise",
    "partial_occlusion",
    "resolution_degradation",
    "fog",
    "motion_blur",
)
T_CACHE_FORMAT = "validation_inclusion_effect_v1"


@dataclass
class StaticBundle:
    scores: dict[str, np.ndarray]
    labels: np.ndarray
    class_names: list[str]
    div_metric: Div
    image_adapter: object
    div_loader: DataLoader | None
    adapter_image_path: Path
    adapter_text_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test validation-inclusion T on normal and corruption experiments."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *SUPPORTED_DATASETS),
        help="Default: test CIFAR-100 and Tiny-ImageNet.",
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=("all", "normal", "corruption"),
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--kr", type=int, default=50)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--force-t", action="store_true")
    parser.add_argument("--ratio-lambda", type=float, default=1e-3)
    parser.add_argument("--regression-learning-rate", type=float, default=2e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-6)
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=2)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < int(args.kr) <= 100:
        raise ValueError("--kr must be in [1, 100].")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid data-loader arguments.")
    if args.regression_learning_rate <= 0:
        raise ValueError("--regression-learning-rate must be positive.")
    if args.regression_max_iter <= 0 or args.regression_tol <= 0:
        raise ValueError("Invalid regression stopping arguments.")
    if args.ratio_lambda < 0:
        raise ValueError("--ratio-lambda must be non-negative.")
    if args.group_candidate_pool_size <= 0:
        raise ValueError("--group-candidate-pool-size must be positive.")
    if args.group_init_count < 1:
        raise ValueError("--group-init-count must be at least 1.")


def selected_datasets(value: str) -> list[str]:
    return list(SUPPORTED_DATASETS) if value == "all" else [value]


def selected_experiments(value: str) -> list[Experiment]:
    return ["normal", "corruption"] if value == "all" else [value]  # type: ignore[list-item]


def expected_epochs(dataset: str) -> int:
    return int(get_proxy_training_config(dataset)["epochs"])


def normal_proxy_root() -> Path:
    return PROJECT_ROOT / "weights" / "proxy_logs"


def proxy_root(experiment: Experiment) -> Path:
    return normal_proxy_root() if experiment == "normal" else corr_mod.PROXY_LOG_ROOT


def resolve_source_proxy_dir(
    experiment: Experiment,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> Path:
    return resolve_proxy_log_path(
        proxy_root(experiment),
        dataset,
        seed=seed,
        proxy_model=proxy_model,
        max_epoch=epochs,
    )


def load_folds(
    experiment: Experiment,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> tuple[list[FoldLogData], np.ndarray, Path]:
    source = resolve_source_proxy_dir(
        experiment, dataset, proxy_model, seed, epochs
    )
    if experiment == "corruption":
        if corruption_info is None:
            raise ValueError("corruption_info is required.")
        context = corr_mod.patched_training_corruption(dataset, corruption_info)
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context:
        folds, labels = load_cv_fold_logs(
            source,
            dataset_name=dataset,
            data_root=str(PROJECT_ROOT / "data"),
            max_epochs=epochs,
        )
    source_epochs = int(folds[0].train_logits.shape[0])
    tqdm.write(
        f"[proxy] experiment={experiment} dataset={dataset} "
        f"target_epochs={epochs} loaded_epochs={source_epochs} source={source}"
    )
    return folds, np.asarray(labels, dtype=np.int64), source


def t_cache_path(
    experiment: Experiment,
    dataset: str,
    seed: int,
    epochs: int,
) -> Path:
    return (
        YYY_ROOT
        / "test_T_cache"
        / experiment
        / dataset
        / str(int(seed))
        / str(int(epochs))
        / "T.npz"
    )


def _fold_class_loss_statistics(
    folds: list[FoldLogData],
    labels_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fold/class G,S and per-sample held-out G,S (leave-one-out)."""
    if not folds:
        raise ValueError("No folds were loaded.")
    num_folds = len(folds)
    num_samples = int(labels_all.size)
    num_classes = int(labels_all.max(initial=-1)) + 1
    num_epochs = int(folds[0].val_logits.shape[0])
    early, middle, late = resolve_epoch_windows(num_epochs)

    fold_g = np.full((num_folds, num_classes), np.nan, dtype=np.float64)
    fold_s = np.full((num_folds, num_classes), np.nan, dtype=np.float64)
    out_g = np.full(num_samples, np.nan, dtype=np.float64)
    out_s = np.full(num_samples, np.nan, dtype=np.float64)
    held_fold = np.full(num_samples, -1, dtype=np.int64)

    for f_idx, fold in enumerate(
        tqdm(folds, desc="new T: validation trajectories", unit="fold")
    ):
        val_idx = np.asarray(fold.val_indices, dtype=np.int64)
        val_labels = labels_all[val_idx]
        val_loss = compute_loss_from_logits(
            np.asarray(fold.val_logits, dtype=np.float32),
            val_labels,
        ).astype(np.float64)
        held_fold[val_idx] = f_idx

        for class_id in np.unique(val_labels):
            local = np.flatnonzero(val_labels == int(class_id))
            if local.size == 0:
                continue
            class_loss = val_loss[:, local]
            trajectory = np.mean(class_loss, axis=1, dtype=np.float64)
            fold_g[f_idx, int(class_id)] = (
                float(np.mean(trajectory[early]))
                - float(np.mean(trajectory[middle]))
            )
            fold_s[f_idx, int(class_id)] = (
                float(np.std(trajectory[middle]))
                - float(np.std(trajectory[late]))
            )

            if local.size == 1:
                loo = trajectory[:, None]
            else:
                total = np.sum(class_loss, axis=1, dtype=np.float64)
                loo = (total[:, None] - class_loss) / float(local.size - 1)

            loo_g = np.mean(loo[early], axis=0) - np.mean(loo[middle], axis=0)
            loo_s = np.std(loo[middle], axis=0) - np.std(loo[late], axis=0)
            global_idx = val_idx[local]
            out_g[global_idx] = loo_g
            out_s[global_idx] = loo_s

    if np.any(held_fold < 0):
        missing = np.flatnonzero(held_fold < 0)
        raise ValueError(f"Samples missing validation-fold assignment: {missing[:10]}")

    in_g = np.zeros(num_samples, dtype=np.float64)
    in_s = np.zeros(num_samples, dtype=np.float64)
    for sample_id in range(num_samples):
        class_id = int(labels_all[sample_id])
        held = int(held_fold[sample_id])
        other_g = np.delete(fold_g[:, class_id], held)
        other_s = np.delete(fold_s[:, class_id], held)
        finite_g = other_g[np.isfinite(other_g)]
        finite_s = other_s[np.isfinite(other_s)]
        if finite_g.size == 0 or finite_s.size == 0:
            raise ValueError(
                f"No finite training-inclusion fold statistic for sample={sample_id}."
            )
        in_g[sample_id] = float(np.mean(finite_g))
        in_s[sample_id] = float(np.mean(finite_s))

    if not np.all(np.isfinite(out_g)) or not np.all(np.isfinite(out_s)):
        raise ValueError("Non-finite held-out validation statistics in new T.")
    return in_g - out_g, in_s - out_s, held_fold, fold_g


def compute_new_t(
    folds: list[FoldLogData], labels_all: np.ndarray
) -> dict[str, np.ndarray]:
    mean_effect, stability_effect, held_fold, _ = _fold_class_loss_statistics(
        folds, labels_all
    )
    mean_z = safe_standardize(mean_effect)
    stability_z = safe_standardize(stability_effect)
    raw = 0.5 * (mean_z + stability_z)
    final = standard_zscore_dynamic(raw)
    final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    if not np.all(np.isfinite(final)):
        raise ValueError("New T contains non-finite values.")
    return {
        "mean_effect": mean_effect.astype(np.float32),
        "stability_effect": stability_effect.astype(np.float32),
        "mean_effect_z": np.asarray(mean_z, dtype=np.float32),
        "stability_effect_z": np.asarray(stability_z, dtype=np.float32),
        "raw": np.asarray(raw, dtype=np.float32),
        "final_normalized": final,
        "held_fold": held_fold.astype(np.int64),
    }


def load_t_cache(
    path: Path,
    *,
    experiment: Experiment,
    dataset: str,
    seed: int,
    epochs: int,
    source_proxy_dir: Path,
    labels: np.ndarray,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "cache_format",
                "experiment",
                "dataset",
                "seed",
                "epochs",
                "source_proxy_dir",
                "labels",
                "mean_effect",
                "stability_effect",
                "mean_effect_z",
                "stability_effect_z",
                "raw",
                "final_normalized",
                "held_fold",
                "corruption_list_hash",
            }
            if not required.issubset(set(data.files)):
                return None
            checks = (
                (str(data["cache_format"].item()), T_CACHE_FORMAT),
                (str(data["experiment"].item()), experiment),
                (str(data["dataset"].item()), dataset),
                (int(data["seed"].item()), int(seed)),
                (int(data["epochs"].item()), int(epochs)),
                (str(data["source_proxy_dir"].item()), str(source_proxy_dir)),
                (
                    str(data["corruption_list_hash"].item()),
                    "" if corruption_info is None else corruption_info.list_hash,
                ),
            )
            if any(actual != expected for actual, expected in checks):
                return None
            if not np.array_equal(
                np.asarray(data["labels"], dtype=np.int64), labels
            ):
                return None
            result = {
                key: np.asarray(data[key])
                for key in (
                    "mean_effect",
                    "stability_effect",
                    "mean_effect_z",
                    "stability_effect_z",
                    "raw",
                    "final_normalized",
                    "held_fold",
                )
            }
    except Exception:
        return None
    n = int(labels.size)
    if any(np.asarray(value).shape != (n,) for value in result.values()):
        return None
    if not np.all(np.isfinite(result["final_normalized"])):
        return None
    return result


def save_t_cache(
    path: Path,
    *,
    experiment: Experiment,
    dataset: str,
    seed: int,
    epochs: int,
    source_proxy_dir: Path,
    labels: np.ndarray,
    result: dict[str, np.ndarray],
    corruption_info: corr_mod.CorruptionInfo | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_format=np.array(T_CACHE_FORMAT, dtype=np.str_),
        experiment=np.array(experiment, dtype=np.str_),
        dataset=np.array(dataset, dtype=np.str_),
        seed=np.array(int(seed), dtype=np.int64),
        epochs=np.array(int(epochs), dtype=np.int64),
        source_proxy_dir=np.array(str(source_proxy_dir), dtype=np.str_),
        corruption_list_hash=np.array(
            "" if corruption_info is None else corruption_info.list_hash,
            dtype=np.str_,
        ),
        labels=np.asarray(labels, dtype=np.int64),
        **{key: np.asarray(value) for key, value in result.items()},
    )


def load_or_compute_t(
    *,
    experiment: Experiment,
    dataset: str,
    seed: int,
    proxy_model: str,
    epochs: int,
    force: bool,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, Path, Path]:
    folds, labels, source = load_folds(
        experiment,
        dataset,
        proxy_model,
        seed,
        epochs,
        corruption_info,
    )
    path = t_cache_path(experiment, dataset, seed, epochs)
    result = None
    if not force:
        result = load_t_cache(
            path,
            experiment=experiment,
            dataset=dataset,
            seed=seed,
            epochs=epochs,
            source_proxy_dir=source,
            labels=labels,
            corruption_info=corruption_info,
        )
    if result is not None:
        tqdm.write(f"[new-T] cache HIT: {path}")
    else:
        tqdm.write(f"[new-T] cache MISS: {path}")
        result = compute_new_t(folds, labels)
        save_t_cache(
            path,
            experiment=experiment,
            dataset=dataset,
            seed=seed,
            epochs=epochs,
            source_proxy_dir=source,
            labels=labels,
            result=result,
            corruption_info=corruption_info,
        )
        tqdm.write(f"[new-T] saved: {path}")
    del folds
    gc.collect()
    return result, labels, source, path


def _dynamic_cache_candidates(
    experiment: Experiment,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    component: str,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> list[Path]:
    if experiment == "normal":
        return [
            PROJECT_ROOT
            / "weights"
            / "dynamic_cache"
            / dataset
            / proxy_model
            / str(int(seed))
            / str(int(epochs))
            / f"{component}.npz"
        ]

    if corruption_info is None:
        raise ValueError("corruption_info is required.")
    base = (
        corr_mod.DYNAMIC_CACHE_ROOT
        / dataset
        / proxy_model
        / str(int(seed))
        / str(int(epochs))
    )
    candidates: list[Path] = []
    # Current formal resolver first; this automatically follows the repository's
    # active corruption cache layout.
    with corr_mod.patched_project_paths(corruption_info):
        candidates.append(
            learn_mod.resolve_dynamic_component_cache_path(
                dataset, proxy_model, seed, epochs, component
            )
        )
    # Compatibility with un-hashed and older one-level special caches.
    candidates.append(base / f"{component}.npz")
    candidates.extend(sorted(base.glob(f"*/{component}.npz")))
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def load_formal_component(
    *,
    experiment: Experiment,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    component: str,
    expected_labels: np.ndarray,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> DynamicComponentResult:
    reasons: list[str] = []
    for path in _dynamic_cache_candidates(
        experiment,
        dataset,
        proxy_model,
        seed,
        epochs,
        component,
        corruption_info,
    ):
        result, labels, reason = learn_mod._load_dynamic_component_cache_with_reason(
            cache_path=path,
            component_name=component,
            dataset=dataset,
            proxy_model=proxy_model,
            proxy_training_seed=seed,
            epochs=epochs,
        )
        if result is None or labels is None:
            reasons.append(f"{path}: {reason}")
            continue
        if not np.array_equal(labels, expected_labels):
            reasons.append(f"{path}: labels mismatch")
            continue
        tqdm.write(f"[dynamic] {component} cache HIT: {path}")
        return result
    raise FileNotFoundError(
        f"No valid formal {component} cache for {experiment}/{dataset}.\n"
        + "\n".join(reasons)
    )


def build_class_names(dataset: str) -> list[str]:
    raw = corr_mod.build_raw_train_dataset(dataset, PROJECT_ROOT / "data")
    return list(
        resolve_class_names_for_prompts(
            dataset_name=dataset,
            data_root=str(PROJECT_ROOT / "data"),
            class_names=raw.classes,
        )
    )


def static_cache_expected_meta(
    *,
    dataset: str,
    seed: int,
    clip_model: str,
    adapter_image_path: Path,
    adapter_text_path: Path,
    div_metric: Div,
    dds_metric: DifficultyDirection,
    sa_metric: SemanticAlignment,
    num_samples: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "clip_model": clip_model,
        "adapter_image_path": str(adapter_image_path),
        "adapter_text_path": str(adapter_text_path),
        "adapter_image_sha1": _hash_file(adapter_image_path),
        "adapter_text_sha1": _hash_file(adapter_text_path),
        "div_k": float(div_metric.k),
        "dds_k": int(dds_metric.k),
        "dds_eigval_lower_bound": float(dds_metric.eigval_lower_bound),
        "dds_eigval_upper_bound": float(dds_metric.eigval_upper_bound),
        "prompt_template": sa_metric.prompt_template,
        "num_samples": int(num_samples),
        "score_storage": NORMALIZATION_VERSION,
        "score_version": NORMALIZATION_VERSION,
    }


def load_static_cache_strict(
    cache_search_root: Path,
    expected_meta: dict[str, object],
    num_samples: int,
) -> dict[str, np.ndarray]:
    if not cache_search_root.exists():
        raise FileNotFoundError(f"Static cache root not found: {cache_search_root}")
    matches: list[dict[str, np.ndarray]] = []
    for sa_path in cache_search_root.rglob("SA_cache.npz"):
        cache_dir = sa_path.parent
        loaded: dict[str, np.ndarray] = {}
        labels_ref: np.ndarray | None = None
        ok = True
        for metric_name, key in (("SA", "sa"), ("Div", "div"), ("DDS", "dds")):
            validated = _validate_metric_cache(
                cache_dir, metric_name, expected_meta, num_samples
            )
            if validated is None:
                ok = False
                break
            scores, labels, _ = validated
            if labels_ref is None:
                labels_ref = np.asarray(labels, dtype=np.int64)
            elif not np.array_equal(labels_ref, labels):
                ok = False
                break
            loaded[key] = np.asarray(scores, dtype=np.float32)
        if ok and labels_ref is not None:
            loaded["labels"] = labels_ref
            matches.append(loaded)
    if not matches:
        raise FileNotFoundError(
            f"No valid SA/Div/DDS cache matching current adapter under {cache_search_root}"
        )
    if len(matches) > 1:
        tqdm.write(
            f"[static] multiple equivalent cache matches under {cache_search_root}; "
            "using the first validated match"
        )
    return matches[0]


def build_static_bundle(
    *,
    experiment: Experiment,
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    corruption_info: corr_mod.CorruptionInfo | None,
) -> StaticBundle:
    class_names = build_class_names(dataset)
    dds = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    div = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    sa = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset,
        data_root=str(PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )

    if experiment == "normal":
        image_adapter, text_adapter, paths = load_trained_adapters(
            dataset_name=dataset,
            clip_model=args.clip_model,
            input_dim=dds.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_path = Path(paths["image_path"])
        text_path = Path(paths["text_path"])
        cache_root = PROJECT_ROOT / "static_scores"
        div_loader = None
    else:
        if corruption_info is None:
            raise ValueError("corruption_info is required.")
        image_path, text_path = corr_mod.adapter_paths(dataset, seed)
        image_adapter, text_adapter, _ = load_trained_adapters(
            dataset_name=dataset,
            clip_model=args.clip_model,
            input_dim=dds.extractor.embed_dim,
            seed=seed,
            map_location=device,
            adapter_image_path=image_path,
            adapter_text_path=text_path,
        )
        cache_root = corr_mod.STATIC_SCORE_ROOT
        corrupted_dataset = corr_mod.FixedCorruptionDataset(
            corr_mod.build_raw_train_dataset(dataset),
            div.extractor.preprocess,
            None,
            corruption_info=corruption_info,
        )
        div_loader = DataLoader(
            corrupted_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    image_adapter.to(device).eval()
    text_adapter.to(device).eval()
    num_samples = len(corr_mod.build_raw_train_dataset(dataset))
    expected_meta = static_cache_expected_meta(
        dataset=dataset,
        seed=seed,
        clip_model=args.clip_model,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
        div_metric=div,
        dds_metric=dds,
        sa_metric=sa,
        num_samples=num_samples,
    )
    scores = load_static_cache_strict(cache_root, expected_meta, num_samples)
    labels = np.asarray(scores["labels"], dtype=np.int64)
    tqdm.write(
        f"[static] experiment={experiment} dataset={dataset} cache HIT under {cache_root}"
    )
    return StaticBundle(
        scores=scores,
        labels=labels,
        class_names=class_names,
        div_metric=div,
        image_adapter=image_adapter,
        div_loader=div_loader,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return 0.0
    xx, yy = x[finite], y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return 0.0
    return pearson(average_ranks(x[finite]), average_ranks(y[finite]))


def print_correlations(
    experiment: Experiment,
    dataset: str,
    t_score: np.ndarray,
    static: StaticBundle,
) -> None:
    print(f"\n[{experiment}/{dataset}] new T vs class-standardized static metrics")
    print(f"{'metric':<10}{'Pearson':>14}{'Spearman':>14}")
    print("-" * 38)
    for key, label in (("sa", "SA"), ("div", "Div"), ("dds", "DDS")):
        feature = standard_zscore_by_class(static.scores[key], static.labels)
        print(
            f"{label:<10}{pearson(t_score, feature):>14.6f}"
            f"{spearman(t_score, feature):>14.6f}"
        )


def fit_weights(
    *,
    a_score: np.ndarray,
    c_score: np.ndarray,
    t_score: np.ndarray,
    static: StaticBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, object]]:
    dynamic_target = standard_zscore(
        (
            np.asarray(a_score, dtype=np.float32)
            + np.asarray(c_score, dtype=np.float32)
            + np.asarray(t_score, dtype=np.float32)
        )
        / 3.0
    ).astype(np.float64)
    features = np.column_stack(
        [
            standard_zscore_by_class(static.scores["sa"], static.labels),
            standard_zscore_by_class(static.scores["div"], static.labels),
            standard_zscore_by_class(static.scores["dds"], static.labels),
        ]
    ).astype(np.float64)
    fit = learn_mod.fit_softplus_ratio_regression(
        features=features,
        targets=dynamic_target,
        ratio_lambda=float(args.ratio_lambda),
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
    return weights, fit


def print_weights(
    experiment: Experiment,
    dataset: str,
    weights: dict[str, float],
    fit: dict[str, object],
) -> None:
    print(
        f"[weights] experiment={experiment} dataset={dataset} "
        f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
        f"DDS={weights['dds']:.6f}, MSE={float(fit['mse']):.6f}, "
        f"iterations={int(fit['iterations'])}"
    )


def print_corruption_t_means(
    t_score: np.ndarray, info: corr_mod.CorruptionInfo
) -> None:
    parts = []
    for type_id, name in enumerate(CORRUPTION_NAMES):
        mask = np.asarray(info.corruption_types) == type_id
        value = float(np.mean(t_score[mask])) if np.any(mask) else float("nan")
        parts.append(f"{name}={value:.6f}")
    print("[new-T corruption means] " + ", ".join(parts))


def run_corruption_mask(
    *,
    dataset: str,
    seed: int,
    weights: dict[str, float],
    static: StaticBundle,
    info: corr_mod.CorruptionInfo,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if static.div_loader is None:
        raise RuntimeError("Corruption Div loader was not built.")
    mask, _, stats = mask_mod.select_group_mask_by_center_repair(
        np.asarray(static.scores["sa"], dtype=np.float32),
        div_metric=static.div_metric,
        div_loader=static.div_loader,
        image_adapter=static.image_adapter,
        labels=static.labels,
        weights=weights,
        num_classes=len(static.class_names),
        keep_ratio=int(args.kr),
        device=device,
        seed=seed,
        dds_static_scores=np.asarray(static.scores["dds"], dtype=np.float32),
        group_candidate_pool_size=int(args.group_candidate_pool_size),
        group_init_count=int(args.group_init_count),
    )
    selected = np.asarray(mask, dtype=np.uint8).astype(bool)
    selected_types = np.asarray(info.corruption_types, dtype=np.int16)[selected]
    parts = [
        f"{name}={int(np.sum(selected_types == type_id))}"
        for type_id, name in enumerate(CORRUPTION_NAMES)
    ]
    total_corrupted_selected = int(np.sum(selected_types >= 0))
    total_corrupted = int(np.sum(info.is_corrupted))
    print(
        f"[mask] experiment=corruption dataset={dataset} seed={seed} kr={args.kr} "
        + ", ".join(parts)
        + f", corrupted_selected={total_corrupted_selected}, "
        f"total_corrupted={total_corrupted}, selected={int(selected.sum())}, "
        f"distribution_shift={float(stats.get('distribution_shift', 0.0)):.6f}; not saved"
    )


def build_corruption_info(dataset: str, seed: int) -> corr_mod.CorruptionInfo:
    raw = corr_mod.build_raw_train_dataset(dataset)
    return corr_mod.load_corruption_info(
        dataset,
        seed,
        num_samples=len(raw),
        strict_expected_size=True,
    )


def run_task(
    experiment: Experiment,
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    set_seed(int(args.seed))
    epochs = expected_epochs(dataset)
    info = build_corruption_info(dataset, args.seed) if experiment == "corruption" else None

    t_data, labels, _, cache_path = load_or_compute_t(
        experiment=experiment,
        dataset=dataset,
        seed=int(args.seed),
        proxy_model=args.proxy_model,
        epochs=epochs,
        force=bool(args.force_t),
        corruption_info=info,
    )
    t_score = np.asarray(t_data["final_normalized"], dtype=np.float32)

    a_result = load_formal_component(
        experiment=experiment,
        dataset=dataset,
        proxy_model=args.proxy_model,
        seed=int(args.seed),
        epochs=epochs,
        component="A",
        expected_labels=labels,
        corruption_info=info,
    )
    c_result = load_formal_component(
        experiment=experiment,
        dataset=dataset,
        proxy_model=args.proxy_model,
        seed=int(args.seed),
        epochs=epochs,
        component="C",
        expected_labels=labels,
        corruption_info=info,
    )
    static = build_static_bundle(
        experiment=experiment,
        dataset=dataset,
        seed=int(args.seed),
        args=args,
        device=device,
        corruption_info=info,
    )
    if not np.array_equal(static.labels, labels):
        raise ValueError("Static-cache labels do not match proxy-log labels.")

    print(f"\n[new-T] cache={cache_path}")
    print_correlations(experiment, dataset, t_score, static)
    if info is not None:
        print_corruption_t_means(t_score, info)

    weights, fit = fit_weights(
        a_score=a_result.final_normalized,
        c_score=c_result.final_normalized,
        t_score=t_score,
        static=static,
        args=args,
        device=device,
    )
    print_weights(experiment, dataset, weights, fit)

    if experiment == "corruption":
        if info is None:
            raise RuntimeError("Missing corruption information.")
        run_corruption_mask(
            dataset=dataset,
            seed=int(args.seed),
            weights=weights,
            static=static,
            info=info,
            args=args,
            device=device,
        )

    del static, a_result, c_result, t_data, t_score, weights, fit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    tasks = [
        (experiment, dataset)
        for dataset in selected_datasets(args.dataset)
        for experiment in selected_experiments(args.experiment)
    ]

    print("=" * 112)
    print("test_T: validation-dominant inclusion-effect dynamic component")
    print(f"tasks={tasks} | seed={args.seed} | kr={args.kr} | device={device}")
    print("A/C and SA/Div/DDS are cache-only; weights and masks are not saved.")
    print(f"new T cache root={YYY_ROOT / 'test_T_cache'}")
    print("=" * 112)

    for experiment, dataset in tqdm(tasks, desc="test_T tasks", unit="task"):
        run_task(experiment, dataset, args, device)


if __name__ == "__main__":
    main()