#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the clipped training-inclusion effect R.

This script is independent of every other Python file under yyy/. It imports
only modules from the project root.

Two variants are evaluated:
  1. T*R: multiply cached raw fold-wise T by clipped R before fold normalization.
  2. R:   replace T with the raw signed effect, while retaining the fold
          normalization and aggregation pipeline used by the existing dynamic
          components. No truncation is applied for this variant.

Definition:
    signed_effect = (U_in - U_out) / (|U_in| + |U_out| + eps)
    R = clip(signed_effect, 0, 1)  # used for T*R and cache storage

Only R is cached, under yyy/R_cache/. Existing A/C/T and SA/Div/DDS caches are
read-only. Learned weights and corruption masks are not saved.

Default:
  seed=22, kr=50
  normal datasets: cifar10,cifar100,tiny-imagenet
  corruption datasets: cifar100,tiny-imagenet

Example:
    CUDA_VISIBLE_DEVICES=0 python yyy/test_R.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
YYY_ROOT = THIS_FILE.parent
PROJECT_ROOT = YYY_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Several project paths are relative to the repository root.
os.chdir(PROJECT_ROOT)

import calculate_my_mask as mask_mod  # noqa: E402
import learn_scoring_weights as learn_weights_mod  # noqa: E402
from corruption_exp import cal_corruption_mask as corruption_mod  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import Div  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.score_utils import standard_zscore, standard_zscore_by_class  # noqa: E402
from utils.training_defaults import get_proxy_training_config  # noqa: E402
from weights.dynamic_utils import (  # noqa: E402
    DynamicComponentResult,
    EPS,
    resolve_epoch_windows,
    standard_zscore_dynamic,
)

NORMAL_DATASETS = ("cifar10", "cifar100", "tiny-imagenet")
CORRUPTION_DATASETS = ("cifar100", "tiny-imagenet")
COMPONENT_NAMES = ("A", "C", "T")
VARIANTS = ("T*R", "R")


@dataclass(frozen=True)
class ExperimentPaths:
    name: str
    proxy_root: Path
    dynamic_root: Path
    static_root: Path
    adapter_root: Path


@dataclass(frozen=True)
class FoldLayout:
    fold_id: int
    path: Path
    train_indices: np.ndarray
    val_indices: np.ndarray
    source_epochs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clipped R using existing normal/corruption caches."
    )
    parser.add_argument(
        "--normal-datasets",
        default=",".join(NORMAL_DATASETS),
        help="Comma-separated normal datasets; pass an empty string to skip.",
    )
    parser.add_argument(
        "--corruption-datasets",
        default=",".join(CORRUPTION_DATASETS),
        help="Comma-separated corruption datasets; pass an empty string to skip.",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--kr", type=int, default=50)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--ratio-lambda", type=float, default=1e-3)
    parser.add_argument("--regression-learning-rate", type=float, default=2e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-8)
    parser.add_argument(
        "--force-r",
        action="store_true",
        help="Recompute R even when a valid yyy/R_cache entry exists.",
    )
    return parser.parse_args()


def parse_dataset_list(text: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    values = [item.strip().lower() for item in text.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed_set]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}; allowed={sorted(allowed_set)}")
    return values


def experiment_paths(experiment: str) -> ExperimentPaths:
    if experiment == "normal":
        return ExperimentPaths(
            name="normal",
            proxy_root=PROJECT_ROOT / "weights" / "proxy_logs",
            dynamic_root=PROJECT_ROOT / "weights" / "dynamic_cache",
            static_root=PROJECT_ROOT / "static_scores",
            adapter_root=PROJECT_ROOT / "adapter_weights",
        )
    if experiment == "corruption":
        root = PROJECT_ROOT / "corruption_exp"
        return ExperimentPaths(
            name="corruption",
            proxy_root=root / "weights" / "proxy_logs",
            dynamic_root=root / "weights" / "dynamic_cache",
            static_root=root / "static_scores",
            adapter_root=root / "adapters",
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def adapter_paths(paths: ExperimentPaths, dataset: str, seed: int) -> tuple[Path, Path]:
    directory = paths.adapter_root / dataset / str(int(seed))
    return directory / "adapter_image.pt", directory / "adapter_context.pt"


def target_epochs(dataset: str) -> int:
    return int(get_proxy_training_config(dataset)["epochs"])


def _fold_number(path: Path) -> int:
    match = re.fullmatch(r"fold_(\d+)\.npz", path.name)
    if match is None:
        raise ValueError(f"Invalid fold filename: {path}")
    return int(match.group(1))


def _read_meta_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _validate_proxy_candidate(
    log_dir: Path,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    required_epochs: int,
    k_folds: int,
) -> tuple[bool, str, list[FoldLayout]]:
    if not log_dir.is_dir():
        return False, "not a directory", []

    meta = _read_meta_json(log_dir / "meta.json")
    # Old logs are supported. Only fields present in meta.json are checked.
    expected_meta = {
        "dataset": dataset,
        "model": proxy_model,
        "seed": int(seed),
        "k_folds": int(k_folds),
    }
    for key, expected in expected_meta.items():
        if key in meta and meta[key] != expected:
            return False, f"meta mismatch: {key}={meta[key]!r}, expected={expected!r}", []
    if "epochs" in meta and int(meta["epochs"]) < int(required_epochs):
        return False, f"meta epochs={meta['epochs']} < required={required_epochs}", []

    fold_paths = sorted(log_dir.glob("fold_*.npz"), key=_fold_number)
    if len(fold_paths) != int(k_folds):
        return False, f"found {len(fold_paths)} folds, expected {k_folds}", []

    layouts: list[FoldLayout] = []
    for fold_id, fold_path in enumerate(fold_paths):
        try:
            with np.load(fold_path, allow_pickle=False) as data:
                required = {"train_indices", "val_indices", "train_logits", "val_logits"}
                missing = sorted(required - set(data.files))
                if missing:
                    return False, f"{fold_path.name} missing keys {missing}", []
                train_indices = np.asarray(data["train_indices"], dtype=np.int64)
                val_indices = np.asarray(data["val_indices"], dtype=np.int64)
                train_shape = tuple(data["train_logits"].shape)
                val_shape = tuple(data["val_logits"].shape)
        except Exception as exc:
            return False, f"cannot read {fold_path.name}: {exc}", []

        if len(train_shape) != 3 or len(val_shape) != 3:
            return False, f"{fold_path.name} logits are not 3D", []
        if train_shape[0] != val_shape[0]:
            return False, f"{fold_path.name} train/val epoch mismatch", []
        if train_shape[0] < int(required_epochs):
            return (
                False,
                f"{fold_path.name} has {train_shape[0]} epochs, required={required_epochs}",
                [],
            )
        if train_shape[1] != train_indices.size or val_shape[1] != val_indices.size:
            return False, f"{fold_path.name} sample dimension mismatch", []
        if train_shape[2] != val_shape[2]:
            return False, f"{fold_path.name} class dimension mismatch", []

        layouts.append(
            FoldLayout(
                fold_id=fold_id,
                path=fold_path,
                train_indices=train_indices,
                val_indices=val_indices,
                source_epochs=int(train_shape[0]),
            )
        )
    return True, "ok", layouts


def resolve_proxy_logs(
    paths: ExperimentPaths,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    required_epochs: int,
    k_folds: int,
) -> tuple[Path, list[FoldLayout]]:
    seed_dir = paths.proxy_root / dataset / proxy_model / str(int(seed))
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Proxy seed directory not found: {seed_dir}")

    candidates: list[tuple[int, Path]] = []
    for path in seed_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            source_epochs = int(path.name)
            if source_epochs >= int(required_epochs):
                candidates.append((source_epochs, path))
    candidates.sort(key=lambda item: (item[0] != int(required_epochs), item[0]))

    rejected: list[str] = []
    for source_epochs, log_dir in candidates:
        valid, reason, layouts = _validate_proxy_candidate(
            log_dir,
            dataset=dataset,
            proxy_model=proxy_model,
            seed=seed,
            required_epochs=required_epochs,
            k_folds=k_folds,
        )
        if valid:
            if source_epochs == int(required_epochs):
                print(f"[proxy] exact log hit: epochs={required_epochs}, path={log_dir}")
            else:
                print(
                    f"[proxy] reuse longer log: source_epochs={source_epochs}, "
                    f"target_epochs={required_epochs}, path={log_dir}"
                )
            return log_dir, layouts
        rejected.append(f"{log_dir}: {reason}")

    details = "\n  ".join(rejected) if rejected else "no usable epoch directory"
    raise FileNotFoundError(
        f"No valid proxy logs for experiment={paths.name}, dataset={dataset}, "
        f"seed={seed}, required_epochs={required_epochs}.\n  {details}"
    )


def _component_metadata_matches(
    data,
    *,
    component: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> bool:
    expected = {
        "component_name": component,
        "dataset": dataset,
        "proxy_model": proxy_model,
        "proxy_training_seed": int(seed),
        "epochs": int(epochs),
    }
    for key, value in expected.items():
        if key not in data.files or data[key].item() != value:
            return False
    return True


def _candidate_dynamic_dirs(
    root: Path,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> list[Path]:
    base = root / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    candidates: list[Path] = []
    if all((base / f"{name}.npz").is_file() for name in COMPONENT_NAMES):
        candidates.append(base)
    if base.is_dir():
        for a_path in base.rglob("A.npz"):
            parent = a_path.parent
            if all((parent / f"{name}.npz").is_file() for name in COMPONENT_NAMES):
                if parent not in candidates:
                    candidates.append(parent)
    return candidates


def load_dynamic_components(
    paths: ExperimentPaths,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> tuple[dict[str, DynamicComponentResult], np.ndarray, Path]:
    candidates = _candidate_dynamic_dirs(
        paths.dynamic_root,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
    )
    valid: list[tuple[Path, dict[str, DynamicComponentResult], np.ndarray]] = []

    for directory in candidates:
        results: dict[str, DynamicComponentResult] = {}
        labels_ref: np.ndarray | None = None
        ok = True
        for component in COMPONENT_NAMES:
            try:
                with np.load(directory / f"{component}.npz", allow_pickle=False) as data:
                    if not _component_metadata_matches(
                        data,
                        component=component,
                        dataset=dataset,
                        proxy_model=proxy_model,
                        seed=seed,
                        epochs=epochs,
                    ):
                        ok = False
                        break
                    labels = np.asarray(data["labels"], dtype=np.int64)
                    result = DynamicComponentResult(
                        raw_foldwise=np.asarray(data["raw_foldwise"], dtype=np.float32),
                        fold_normalized=np.asarray(data["fold_normalized"], dtype=np.float32),
                        aggregated=np.asarray(data["aggregated"], dtype=np.float32),
                        final_normalized=np.asarray(data["final_normalized"], dtype=np.float32),
                    )
            except Exception:
                ok = False
                break

            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                ok = False
                break
            results[component] = result

        if ok and labels_ref is not None and len(results) == 3:
            valid.append((directory, results, labels_ref))

    if not valid:
        base = paths.dynamic_root / dataset / proxy_model / str(int(seed)) / str(int(epochs))
        raise FileNotFoundError(
            f"No valid A/C/T cache bundle under {base}. "
            "The test script never recomputes A/C/T."
        )

    # Canonical non-nested paths take precedence. No hash path is constructed.
    valid.sort(key=lambda item: (len(item[0].relative_to(paths.dynamic_root).parts), str(item[0])))
    shortest = len(valid[0][0].relative_to(paths.dynamic_root).parts)
    same_depth = [
        item for item in valid
        if len(item[0].relative_to(paths.dynamic_root).parts) == shortest
    ]
    if len(same_depth) > 1:
        choices = "\n  ".join(str(item[0]) for item in valid)
        raise RuntimeError(f"Multiple valid dynamic cache bundles found:\n  {choices}")

    directory, results, labels = valid[0]
    print(f"[dynamic] A/C/T cache hit: {directory}")
    return results, labels, directory


def _resolve_meta_path(value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _same_path(value: object, expected: Path) -> bool:
    return _resolve_meta_path(value) == expected.resolve(strict=False)


def _load_static_metric(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        if not {"scores", "labels", "indices", "meta"}.issubset(set(data.files)):
            raise ValueError(f"Invalid static cache: {path}")
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        indices = np.asarray(data["indices"], dtype=np.int64)
        meta_value = data["meta"].item()
        meta = json.loads(str(meta_value))
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid static cache metadata: {path}")
    return scores, labels, indices, meta


def load_static_scores_readonly(
    paths: ExperimentPaths,
    *,
    dataset: str,
    seed: int,
    clip_model: str,
    num_samples: int,
    expected_image_adapter: Path,
    expected_text_adapter: Path,
) -> tuple[dict[str, np.ndarray], Path]:
    matches: list[tuple[Path, dict[str, np.ndarray]]] = []

    for sa_path in sorted(paths.static_root.rglob("SA_cache.npz")):
        directory = sa_path.parent
        div_path = directory / "Div_cache.npz"
        dds_path = directory / "DDS_cache.npz"
        if not div_path.is_file() or not dds_path.is_file():
            continue
        try:
            sa, labels_sa, indices_sa, meta_sa = _load_static_metric(sa_path)
            div, labels_div, indices_div, meta_div = _load_static_metric(div_path)
            dds, labels_dds, indices_dds, meta_dds = _load_static_metric(dds_path)
        except Exception:
            continue

        metas = (meta_sa, meta_div, meta_dds)
        if any(meta.get("dataset") != dataset for meta in metas):
            continue
        if any(int(meta.get("seed", -1)) != int(seed) for meta in metas):
            continue
        if any(meta.get("clip_model") != clip_model for meta in metas):
            continue
        if any(int(meta.get("num_samples", -1)) != int(num_samples) for meta in metas):
            continue
        if any(abs(float(meta.get("div_k", np.nan)) - 0.05) > 1e-12 for meta in metas):
            continue
        if any(int(meta.get("dds_k", -1)) != 5 for meta in metas):
            continue
        if any(
            not _same_path(meta.get("adapter_image_path", ""), expected_image_adapter)
            for meta in metas
        ):
            continue
        if any(
            not _same_path(meta.get("adapter_text_path", ""), expected_text_adapter)
            for meta in metas
        ):
            continue
        if not (
            np.array_equal(labels_sa, labels_div)
            and np.array_equal(labels_sa, labels_dds)
            and np.array_equal(indices_sa, indices_div)
            and np.array_equal(indices_sa, indices_dds)
        ):
            continue
        if labels_sa.shape != (num_samples,):
            continue
        if not np.array_equal(indices_sa, np.arange(num_samples, dtype=indices_sa.dtype)):
            continue
        if any(array.shape != (num_samples,) for array in (sa, div, dds)):
            continue

        matches.append(
            (
                directory,
                {"sa": sa, "div": div, "dds": dds, "labels": labels_sa},
            )
        )

    if not matches:
        raise FileNotFoundError(
            f"No matching static cache under {paths.static_root} for "
            f"experiment={paths.name}, dataset={dataset}, seed={seed}. "
            "The test script never recomputes SA/Div/DDS."
        )

    matches.sort(
        key=lambda item: (
            len(item[0].relative_to(paths.static_root).parts),
            str(item[0]),
        )
    )
    shortest = len(matches[0][0].relative_to(paths.static_root).parts)
    same_depth = [
        item for item in matches
        if len(item[0].relative_to(paths.static_root).parts) == shortest
    ]
    if len(same_depth) > 1:
        choices = "\n  ".join(str(item[0]) for item in matches)
        raise RuntimeError(f"Multiple matching static cache bundles found:\n  {choices}")

    directory, scores = matches[0]
    print(f"[static] SA/Div/DDS cache hit: {directory}")
    return scores, directory


def _source_signature(log_dir: Path, layouts: list[FoldLayout]) -> str:
    records: list[dict[str, object]] = []
    for path in [log_dir / "meta.json"] + [layout.path for layout in layouts]:
        if not path.is_file():
            continue
        stat = path.stat()
        records.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def r_cache_path(
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> Path:
    return (
        YYY_ROOT
        / "R_cache"
        / experiment
        / dataset
        / proxy_model
        / str(int(seed))
        / str(int(epochs))
        / "R.npz"
    )


def _cross_entropy_for_epochs(
    logits: np.ndarray,
    labels: np.ndarray,
    epoch_indices: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(logits[epoch_indices], dtype=np.float64)
    max_logits = np.max(selected, axis=2, keepdims=True)
    shifted = selected - max_logits
    logsumexp = np.log(
        np.sum(np.exp(np.clip(shifted, -50.0, 50.0)), axis=2)
    ) + max_logits.squeeze(2)
    true_logits = np.take_along_axis(
        selected,
        labels.reshape(1, -1, 1),
        axis=2,
    ).squeeze(2)
    return (logsumexp - true_logits).astype(np.float64)


def compute_r_from_proxy_logs(
    *,
    layouts: list[FoldLayout],
    labels_all: np.ndarray,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(labels_all.shape[0])
    in_sum = np.zeros(num_samples, dtype=np.float64)
    in_count = np.zeros(num_samples, dtype=np.int64)
    u_out = np.full(num_samples, np.nan, dtype=np.float64)
    val_seen = np.zeros(num_samples, dtype=np.int64)

    _, mid_idx, late_idx = resolve_epoch_windows(int(epochs))
    selected_epochs = np.unique(np.concatenate([mid_idx, late_idx]))
    selected_epochs = selected_epochs[selected_epochs >= 1]
    if selected_epochs.size == 0:
        selected_epochs = np.array([int(epochs) - 1], dtype=np.int64)
    previous_epochs = selected_epochs - 1

    for layout in layouts:
        train_idx = layout.train_indices
        val_idx = layout.val_indices
        y_train = labels_all[train_idx]
        y_val = labels_all[val_idx]

        with np.load(layout.path, allow_pickle=False) as data:
            val_logits = np.asarray(data["val_logits"][:epochs], dtype=np.float32)

        previous_loss = _cross_entropy_for_epochs(
            val_logits, y_val, previous_epochs
        )
        current_loss = _cross_entropy_for_epochs(
            val_logits, y_val, selected_epochs
        )
        signed_improvement = previous_loss - current_loss
        per_val_sample_u = np.mean(signed_improvement, axis=0)

        all_classes = np.unique(np.concatenate([y_train, y_val]))
        for class_id_value in all_classes:
            class_id = int(class_id_value)
            val_local = np.flatnonzero(y_val == class_id)
            train_global = train_idx[y_train == class_id]

            if val_local.size == 0:
                class_u = 0.0
            else:
                class_values = per_val_sample_u[val_local]
                class_u = float(np.mean(class_values))

            in_sum[train_global] += class_u
            in_count[train_global] += 1

            if val_local.size == 0:
                continue
            class_values = per_val_sample_u[val_local]
            if val_local.size == 1:
                leave_one_out = np.array([class_u], dtype=np.float64)
            else:
                leave_one_out = (
                    float(np.sum(class_values)) - class_values
                ) / float(val_local.size - 1)

            global_val = val_idx[val_local]
            u_out[global_val] = leave_one_out
            val_seen[global_val] += 1

        del val_logits, previous_loss, current_loss, signed_improvement

    if np.any(in_count <= 0):
        missing = np.flatnonzero(in_count <= 0)
        raise ValueError(f"Some samples never appeared in training folds: {missing[:10]}")
    if np.any(val_seen != 1):
        bad = np.flatnonzero(val_seen != 1)
        raise ValueError(f"Validation assignment is not exactly once: {bad[:10]}")
    if not np.all(np.isfinite(u_out)):
        bad = np.flatnonzero(~np.isfinite(u_out))
        raise ValueError(f"U_out contains invalid entries: {bad[:10]}")

    u_in = in_sum / in_count
    signed_effect = (u_in - u_out) / (
        np.abs(u_in) + np.abs(u_out) + EPS
    )
    signed_effect = np.nan_to_num(
        signed_effect, nan=0.0, posinf=1.0, neginf=-1.0
    )

    # Direct clip requested by the experiment design.
    r = np.clip(signed_effect, 0.0, 1.0).astype(np.float32)
    return (
        r,
        u_in.astype(np.float32),
        u_out.astype(np.float32),
        signed_effect.astype(np.float32),
    )


def load_or_compute_r(
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    labels_all: np.ndarray,
    log_dir: Path,
    layouts: list[FoldLayout],
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    path = r_cache_path(
        experiment=experiment,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
    )
    signature = _source_signature(log_dir, layouts)

    if path.is_file() and not force:
        try:
            with np.load(path, allow_pickle=False) as data:
                valid = (
                    str(data["experiment"].item()) == experiment
                    and str(data["dataset"].item()) == dataset
                    and str(data["proxy_model"].item()) == proxy_model
                    and int(data["seed"].item()) == int(seed)
                    and int(data["epochs"].item()) == int(epochs)
                    and str(data["proxy_log_path"].item()) == str(log_dir)
                    and str(data["source_signature"].item()) == signature
                    and np.array_equal(
                        np.asarray(data["labels"], dtype=np.int64),
                        labels_all.astype(np.int64, copy=False),
                    )
                )
                if valid:
                    r = np.asarray(data["R"], dtype=np.float32)
                    u_in = np.asarray(data["U_in"], dtype=np.float32)
                    u_out = np.asarray(data["U_out"], dtype=np.float32)
                    signed_effect = np.asarray(
                        data["signed_effect"], dtype=np.float32
                    )
                    arrays = (r, u_in, u_out, signed_effect)
                    if all(
                        array.shape == labels_all.shape
                        and np.all(np.isfinite(array))
                        for array in arrays
                    ):
                        print(f"[R] cache hit: {path}")
                        return r, u_in, u_out, signed_effect, path
        except Exception:
            pass

    print(f"[R] cache miss: compute from {log_dir}")
    r, u_in, u_out, signed_effect = compute_r_from_proxy_logs(
        layouts=layouts,
        labels_all=labels_all,
        epochs=epochs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment=np.array(experiment, dtype=np.str_),
        dataset=np.array(dataset, dtype=np.str_),
        proxy_model=np.array(proxy_model, dtype=np.str_),
        seed=np.array(int(seed), dtype=np.int64),
        epochs=np.array(int(epochs), dtype=np.int64),
        proxy_log_path=np.array(str(log_dir), dtype=np.str_),
        source_signature=np.array(signature, dtype=np.str_),
        labels=labels_all.astype(np.int64, copy=False),
        R=r,
        U_in=u_in,
        U_out=u_out,
        signed_effect=signed_effect,
    )
    print(f"[R] cache saved: {path}")
    return r, u_in, u_out, signed_effect, path


def _aggregate_fold_normalized(
    fold_normalized: np.ndarray,
    layouts: list[FoldLayout],
    num_samples: int,
) -> np.ndarray:
    total = np.zeros(num_samples, dtype=np.float64)
    count = np.zeros(num_samples, dtype=np.int64)
    for fold_id, layout in enumerate(layouts):
        values = fold_normalized[fold_id, layout.train_indices]
        finite = np.isfinite(values)
        indices = layout.train_indices[finite]
        total[indices] += values[finite].astype(np.float64)
        count[indices] += 1
    if np.any(count <= 0):
        missing = np.flatnonzero(count <= 0)
        raise ValueError(f"Component aggregation missing samples: {missing[:10]}")
    return (total / count).astype(np.float32)


def build_variant_component(
    *,
    variant: str,
    r: np.ndarray,
    signed_effect: np.ndarray,
    cached_t: DynamicComponentResult,
    layouts: list[FoldLayout],
    num_samples: int,
) -> DynamicComponentResult:
    expected_shape = (len(layouts), num_samples)
    if cached_t.raw_foldwise.shape != expected_shape:
        raise ValueError(
            "Cached T raw_foldwise does not match proxy folds: "
            f"T={cached_t.raw_foldwise.shape}, expected={expected_shape}"
        )

    raw_foldwise = np.full(expected_shape, np.nan, dtype=np.float32)
    fold_normalized = np.full(expected_shape, np.nan, dtype=np.float32)

    for fold_id, layout in enumerate(layouts):
        train_idx = layout.train_indices
        if variant == "T*R":
            base_t = np.asarray(
                cached_t.raw_foldwise[fold_id, train_idx],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(base_t)):
                raise ValueError(
                    f"Cached T contains invalid train values in fold {fold_id}"
                )
            values = base_t * r[train_idx]
        elif variant == "R":
            values = signed_effect[train_idx]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        raw_foldwise[fold_id, train_idx] = values
        fold_normalized[fold_id, train_idx] = standard_zscore_dynamic(values)

    aggregated = _aggregate_fold_normalized(
        fold_normalized, layouts, num_samples
    )
    final_normalized = standard_zscore_dynamic(aggregated)
    if not np.all(np.isfinite(final_normalized)):
        raise ValueError(f"{variant} final component contains NaN/inf")

    return DynamicComponentResult(
        raw_foldwise=raw_foldwise,
        fold_normalized=fold_normalized,
        aggregated=aggregated,
        final_normalized=final_normalized,
    )


def build_dynamic_target(
    a_result: DynamicComponentResult,
    c_result: DynamicComponentResult,
    third_result: DynamicComponentResult,
) -> np.ndarray:
    raw = (
        np.asarray(a_result.final_normalized, dtype=np.float64)
        + np.asarray(c_result.final_normalized, dtype=np.float64)
        + np.asarray(third_result.final_normalized, dtype=np.float64)
    ) / 3.0
    return np.asarray(standard_zscore(raw), dtype=np.float64)


def learn_weights_in_memory(
    *,
    static_scores: dict[str, np.ndarray],
    dynamic_target: np.ndarray,
    device: torch.device,
    ratio_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, float], dict[str, object]]:
    labels = np.asarray(static_scores["labels"], dtype=np.int64)
    sa_z = standard_zscore_by_class(
        np.asarray(static_scores["sa"], dtype=np.float32), labels
    )
    div_z = standard_zscore_by_class(
        np.asarray(static_scores["div"], dtype=np.float32), labels
    )
    dds_z = standard_zscore_by_class(
        np.asarray(static_scores["dds"], dtype=np.float32), labels
    )
    features = np.stack([sa_z, div_z, dds_z], axis=1).astype(np.float64)

    fit = learn_weights_mod.fit_softplus_ratio_regression(
        features,
        np.asarray(dynamic_target, dtype=np.float64),
        ratio_lambda,
        learning_rate,
        max_iter,
        tol,
        device,
    )
    normalized = np.asarray(fit["normalized_weights"], dtype=np.float64)
    weights = {
        "sa": float(normalized[0]),
        "div": float(normalized[1]),
        "dds": float(normalized[2]),
    }
    return weights, fit


def print_r_summary(
    experiment: str,
    dataset: str,
    r: np.ndarray,
    u_in: np.ndarray,
    u_out: np.ndarray,
) -> None:
    quantiles = np.quantile(r, [0.0, 0.25, 0.5, 0.75, 1.0])
    print(
        f"[R] experiment={experiment} dataset={dataset} "
        f"mean={float(np.mean(r)):.6f} "
        f"positive_rate={float(np.mean(r > 0.0)):.6f} "
        f"q0/q25/q50/q75/q100="
        f"{','.join(f'{value:.6f}' for value in quantiles)} "
        f"mean_U_in={float(np.mean(u_in)):.6e} "
        f"mean_U_out={float(np.mean(u_out)):.6e}"
    )


def prepare_common_data(
    *,
    paths: ExperimentPaths,
    dataset: str,
    seed: int,
    proxy_model: str,
    clip_model: str,
    k_folds: int,
    force_r: bool,
) -> tuple[
    dict[str, DynamicComponentResult],
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    list[FoldLayout],
]:
    epochs = target_epochs(dataset)
    components, labels, _ = load_dynamic_components(
        paths,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
    )
    log_dir, layouts = resolve_proxy_logs(
        paths,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        required_epochs=epochs,
        k_folds=k_folds,
    )
    if components["T"].raw_foldwise.shape[0] != len(layouts):
        raise ValueError(
            f"T cache folds={components['T'].raw_foldwise.shape[0]}, "
            f"proxy folds={len(layouts)}"
        )

    image_path, text_path = adapter_paths(paths, dataset, seed)
    if not image_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(
            f"Adapter files missing for experiment={paths.name}: "
            f"{image_path}, {text_path}"
        )

    static_scores, _ = load_static_scores_readonly(
        paths,
        dataset=dataset,
        seed=seed,
        clip_model=clip_model,
        num_samples=labels.shape[0],
        expected_image_adapter=image_path,
        expected_text_adapter=text_path,
    )
    if not np.array_equal(
        labels, np.asarray(static_scores["labels"], dtype=np.int64)
    ):
        raise ValueError(
            f"Dynamic/static labels mismatch: experiment={paths.name}, "
            f"dataset={dataset}"
        )

    r, u_in, u_out, signed_effect, _ = load_or_compute_r(
        experiment=paths.name,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
        labels_all=labels,
        log_dir=log_dir,
        layouts=layouts,
        force=force_r,
    )
    print_r_summary(paths.name, dataset, r, u_in, u_out)
    return components, labels, static_scores, r, signed_effect, layouts


def run_normal_dataset(
    *,
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> None:
    print(f"\n{'=' * 80}\nNORMAL | dataset={dataset} seed={args.seed}\n{'=' * 80}")
    paths = experiment_paths("normal")
    components, labels, static_scores, r, signed_effect, layouts = prepare_common_data(
        paths=paths,
        dataset=dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        clip_model=args.clip_model,
        k_folds=args.k_folds,
        force_r=args.force_r,
    )

    for variant in VARIANTS:
        third = build_variant_component(
            variant=variant,
            r=r,
            signed_effect=signed_effect,
            cached_t=components["T"],
            layouts=layouts,
            num_samples=labels.shape[0],
        )
        dynamic_target = build_dynamic_target(
            components["A"], components["C"], third
        )
        weights, fit = learn_weights_in_memory(
            static_scores=static_scores,
            dynamic_target=dynamic_target,
            device=device,
            ratio_lambda=args.ratio_lambda,
            learning_rate=args.regression_learning_rate,
            max_iter=args.regression_max_iter,
            tol=args.regression_tol,
        )
        print(
            f"[normal][{variant}] weights: "
            f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
            f"DDS={weights['dds']:.6f}, "
            f"bias={float(fit['bias']):.6f}, "
            f"mse={float(fit['mse']):.6e}"
        )


def _print_corruption_retention(
    *,
    variant: str,
    mask: np.ndarray,
    info,
) -> None:
    selected = np.asarray(mask, dtype=np.uint8).astype(bool)
    parts: list[str] = []
    for type_id in range(corruption_mod.corruption_opt.NUM_CORRUPTION_TYPES):
        name = corruption_mod.corruption_opt.CORRUPTION_ID_TO_NAME[type_id]
        type_mask = info.corruption_types == type_id
        total = int(np.sum(type_mask))
        retained = int(np.sum(selected & type_mask))
        ratio = float(retained / total) if total > 0 else 0.0
        parts.append(f"{name}={retained}/{total} ({ratio:.6f})")
    print(f"[corruption][{variant}] retained ratios | " + " | ".join(parts))


def run_corruption_dataset(
    *,
    args: argparse.Namespace,
    dataset: str,
    device: torch.device,
) -> None:
    print(
        f"\n{'=' * 80}\nCORRUPTION | dataset={dataset} "
        f"seed={args.seed} kr={args.kr}\n{'=' * 80}"
    )
    paths = experiment_paths("corruption")
    components, labels, static_scores, r, signed_effect, layouts = prepare_common_data(
        paths=paths,
        dataset=dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        clip_model=args.clip_model,
        k_folds=args.k_folds,
        force_r=args.force_r,
    )

    raw_dataset = corruption_mod.build_raw_train_dataset(dataset)
    info = corruption_mod.load_corruption_info(
        dataset,
        args.seed,
        num_samples=len(raw_dataset),
        strict_expected_size=True,
    )
    clean_labels = corruption_mod.extract_labels(raw_dataset)
    if not np.array_equal(labels, clean_labels):
        raise ValueError(
            "Corruption dynamic/static labels do not match clean labels."
        )

    class_names = corruption_mod.build_class_names(dataset)
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=0.05,
    )
    image_path, text_path = adapter_paths(paths, dataset, args.seed)
    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=dataset,
        clip_model=args.clip_model,
        input_dim=div_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    corrupted_dataset = corruption_mod.FixedCorruptionDataset(
        corruption_mod.build_raw_train_dataset(dataset),
        div_metric.extractor.preprocess,
        None,
        corruption_info=info,
    )
    div_loader = DataLoader(
        corrupted_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Encode once and use the same Div feature tensor for both variants.
    encoded_features, encoded_labels = div_metric._encode_images(
        div_loader, image_adapter
    )
    original_encode = div_metric._encode_images

    def cached_encode(_loader, _adapter):
        return encoded_features, encoded_labels

    div_metric._encode_images = cached_encode  # type: ignore[method-assign]
    try:
        for variant in VARIANTS:
            third = build_variant_component(
                variant=variant,
                r=r,
                signed_effect=signed_effect,
                cached_t=components["T"],
                layouts=layouts,
                num_samples=labels.shape[0],
            )
            dynamic_target = build_dynamic_target(
                components["A"], components["C"], third
            )
            weights, fit = learn_weights_in_memory(
                static_scores=static_scores,
                dynamic_target=dynamic_target,
                device=device,
                ratio_lambda=args.ratio_lambda,
                learning_rate=args.regression_learning_rate,
                max_iter=args.regression_max_iter,
                tol=args.regression_tol,
            )
            print(
                f"[corruption][{variant}] weights: "
                f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
                f"DDS={weights['dds']:.6f}, "
                f"bias={float(fit['bias']):.6f}, "
                f"mse={float(fit['mse']):.6e}"
            )

            mask, selected_by_class, stats = (
                mask_mod.select_group_mask_by_center_repair(
                    np.asarray(static_scores["sa"], dtype=np.float32),
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels,
                    weights=weights,
                    num_classes=len(class_names),
                    keep_ratio=args.kr,
                    device=device,
                    seed=args.seed,
                    dds_static_scores=np.asarray(
                        static_scores["dds"], dtype=np.float32
                    ),
                    group_candidate_pool_size=(
                        args.group_candidate_pool_size
                    ),
                    group_init_count=args.group_init_count,
                )
            )
            expected = int(round(labels.shape[0] * args.kr / 100.0))
            if int(mask.sum()) != expected:
                raise RuntimeError(
                    f"{variant} mask selected {int(mask.sum())}, "
                    f"expected {expected}"
                )
            if sum(selected_by_class.values()) != expected:
                raise RuntimeError(
                    f"{variant} selected_by_class total mismatch"
                )
            print(
                f"[corruption][{variant}] mask summary: "
                f"selected={int(mask.sum())}, "
                f"distribution_shift="
                f"{float(stats['distribution_shift']):.6f}, "
                f"subset_score="
                f"{float(stats['subset_comprehensive_score']):.6f}"
            )
            _print_corruption_retention(
                variant=variant,
                mask=mask,
                info=info,
            )
    finally:
        div_metric._encode_images = original_encode  # type: ignore[method-assign]

    del encoded_features, encoded_labels
    del image_adapter, text_adapter, div_metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.kr <= 0 or args.kr > 100:
        raise ValueError("--kr must be in 1..100")
    if args.k_folds <= 1:
        raise ValueError("--k-folds must be greater than 1")

    normal_datasets = parse_dataset_list(
        args.normal_datasets, NORMAL_DATASETS
    )
    corruption_datasets = parse_dataset_list(
        args.corruption_datasets, CORRUPTION_DATASETS
    )
    device = (
        torch.device(args.device)
        if args.device is not None
        else CONFIG.global_device
    )

    print(
        f"device={device} seed={args.seed} kr={args.kr} "
        f"normal={normal_datasets} corruption={corruption_datasets}"
    )
    print(f"R cache root: {YYY_ROOT / 'R_cache'}")
    print("Only R is saved. Learned weights and masks remain in memory.")

    for dataset in normal_datasets:
        run_normal_dataset(args=args, dataset=dataset, device=device)

    for dataset in corruption_datasets:
        run_corruption_dataset(args=args, dataset=dataset, device=device)


if __name__ == "__main__":
    main()
