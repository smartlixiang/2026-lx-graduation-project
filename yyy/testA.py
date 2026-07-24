#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise ablation of the three sub-components of the current dynamic A.

Independent of all other Python files under yyy/. It imports only formal
project modules. Defaults: normal/corruption on cifar100 and tiny-imagenet,
seed=22, kr=50.

Only A sub-component caches are saved under yyy/A_component_cache/. Learned
weights and corruption masks remain in memory.
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
    resolve_epoch_windows,
    safe_standardize,
    standard_zscore_dynamic,
)
import importlib.util

# Dynamically load testR.py so we can reuse its R-cache logic without
# creating an import-time package dependency.
spec = importlib.util.spec_from_file_location("testR_module", THIS_FILE.parent / "testR.py")
testR_mod = importlib.util.module_from_spec(spec)
import sys as _sys
# Register the module in sys.modules so dataclasses and other imports
# that inspect the module can find it during exec_module.
_sys.modules[spec.name] = testR_mod
spec.loader.exec_module(testR_mod)

DATASETS = ("cifar100", "tiny-imagenet")
COMPONENT_NAMES = ("A", "C", "T")
A_SUBCOMPONENTS = ("boundary", "gain", "stability")
A_VARIANTS: dict[str, tuple[str, str]] = {
    "boundary+gain": ("boundary", "gain"),
    "boundary+stability": ("boundary", "stability"),
    "gain+stability": ("gain", "stability"),
}


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
    p = argparse.ArgumentParser(description="Pairwise ablation of current A sub-components.")
    p.add_argument("--normal-datasets", default=",".join(DATASETS))
    p.add_argument("--corruption-datasets", default=",".join(DATASETS))
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--kr", type=int, default=50)
    p.add_argument("--proxy-model", default="resnet18")
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--group-candidate-pool-size", type=int, default=10)
    p.add_argument("--group-init-count", type=int, default=2)
    p.add_argument("--ratio-lambda", type=float, default=5e-3)
    p.add_argument("--regression-learning-rate", type=float, default=2e-3)
    p.add_argument("--regression-max-iter", type=int, default=10000)
    p.add_argument("--regression-tol", type=float, default=1e-8)
    p.add_argument("--epoch-chunk-size", type=int, default=4)
    p.add_argument("--force-a-components", action="store_true")
    return p.parse_args()


def parse_dataset_list(text: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    values = [x.strip().lower() for x in text.split(",") if x.strip()]
    invalid = [x for x in values if x not in allowed_set]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}; allowed={sorted(allowed_set)}")
    return values


def experiment_paths(experiment: str) -> ExperimentPaths:
    if experiment == "normal":
        return ExperimentPaths(
            "normal",
            PROJECT_ROOT / "weights" / "proxy_logs",
            PROJECT_ROOT / "weights" / "dynamic_cache",
            PROJECT_ROOT / "static_scores",
            PROJECT_ROOT / "adapter_weights",
        )
    if experiment == "corruption":
        root = PROJECT_ROOT / "corruption_exp"
        return ExperimentPaths(
            "corruption",
            root / "weights" / "proxy_logs",
            root / "weights" / "dynamic_cache",
            root / "static_scores",
            root / "adapters",
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def adapter_paths(paths: ExperimentPaths, dataset: str, seed: int) -> tuple[Path, Path]:
    root = paths.adapter_root / dataset / str(int(seed))
    return root / "adapter_image.pt", root / "adapter_context.pt"


def target_epochs(dataset: str) -> int:
    return int(get_proxy_training_config(dataset)["epochs"])


def _fold_number(path: Path) -> int:
    m = re.fullmatch(r"fold_(\d+)\.npz", path.name)
    if m is None:
        raise ValueError(f"Invalid fold filename: {path}")
    return int(m.group(1))


def _read_meta(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except Exception:
        value = {}
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
    meta = _read_meta(log_dir / "meta.json")
    expected = {"dataset": dataset, "model": proxy_model, "seed": seed, "k_folds": k_folds}
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            return False, f"meta mismatch: {key}", []
    if "epochs" in meta and int(meta["epochs"]) < required_epochs:
        return False, "meta epochs too short", []

    fold_paths = sorted(log_dir.glob("fold_*.npz"), key=_fold_number)
    if len(fold_paths) != k_folds:
        return False, f"found {len(fold_paths)} folds, expected {k_folds}", []
    layouts: list[FoldLayout] = []
    for fold_id, path in enumerate(fold_paths):
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"train_indices", "val_indices", "train_logits", "val_logits"}
                missing = sorted(required - set(data.files))
                if missing:
                    return False, f"{path.name} missing {missing}", []
                train_idx = np.asarray(data["train_indices"], dtype=np.int64)
                val_idx = np.asarray(data["val_indices"], dtype=np.int64)
                train_shape = tuple(data["train_logits"].shape)
                val_shape = tuple(data["val_logits"].shape)
        except Exception as exc:
            return False, f"cannot read {path.name}: {exc}", []
        if len(train_shape) != 3 or len(val_shape) != 3:
            return False, f"{path.name} logits are not 3D", []
        if train_shape[0] != val_shape[0] or train_shape[0] < required_epochs:
            return False, f"{path.name} epoch mismatch/short", []
        if train_shape[1] != train_idx.size or val_shape[1] != val_idx.size:
            return False, f"{path.name} sample mismatch", []
        if train_shape[2] != val_shape[2]:
            return False, f"{path.name} class mismatch", []
        layouts.append(FoldLayout(fold_id, path, train_idx, val_idx, int(train_shape[0])))
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
    candidates = [
        (int(p.name), p)
        for p in seed_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and int(p.name) >= required_epochs
    ]
    candidates.sort(key=lambda x: (x[0] != required_epochs, x[0]))
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
            if source_epochs == required_epochs:
                print(f"[proxy] exact log hit: epochs={required_epochs}, path={log_dir}")
            else:
                print(
                    f"[proxy] reuse longer log: source_epochs={source_epochs}, "
                    f"target_epochs={required_epochs}, path={log_dir}"
                )
            return log_dir, layouts
        rejected.append(f"{log_dir}: {reason}")
    details = "\n  ".join(rejected) if rejected else "no usable epoch directory"
    raise FileNotFoundError(f"No valid proxy logs under {seed_dir}:\n  {details}")


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
    return all(key in data.files and data[key].item() == value for key, value in expected.items())


def _candidate_dynamic_dirs(
    root: Path,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> list[Path]:
    base = root / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    out: list[Path] = []
    if all((base / f"{name}.npz").is_file() for name in COMPONENT_NAMES):
        out.append(base)
    if base.is_dir():
        for a_path in base.rglob("A.npz"):
            parent = a_path.parent
            if all((parent / f"{name}.npz").is_file() for name in COMPONENT_NAMES):
                if parent not in out:
                    out.append(parent)
    return out


def load_dynamic_components(
    paths: ExperimentPaths,
    *,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> tuple[dict[str, DynamicComponentResult], np.ndarray, Path]:
    valid: list[tuple[Path, dict[str, DynamicComponentResult], np.ndarray]] = []
    for directory in _candidate_dynamic_dirs(
        paths.dynamic_root,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
    ):
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
        raise FileNotFoundError(f"No valid A/C/T cache bundle under {base}")

    # Use the canonical shortest existing path. The script never constructs hash paths.
    valid.sort(key=lambda x: (len(x[0].relative_to(paths.dynamic_root).parts), str(x[0])))
    shortest = len(valid[0][0].relative_to(paths.dynamic_root).parts)
    same_depth = [x for x in valid if len(x[0].relative_to(paths.dynamic_root).parts) == shortest]
    if len(same_depth) > 1:
        raise RuntimeError("Multiple equally preferred dynamic cache bundles:\n  " + "\n  ".join(str(x[0]) for x in same_depth))
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
        required = {"scores", "labels", "indices", "meta"}
        if not required.issubset(set(data.files)):
            raise ValueError(f"Invalid static cache: {path}")
        scores = np.asarray(data["scores"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        indices = np.asarray(data["indices"], dtype=np.int64)
        meta = json.loads(str(data["meta"].item()))
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid static metadata: {path}")
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
        if any(int(meta.get("seed", -1)) != seed for meta in metas):
            continue
        if any(meta.get("clip_model") != clip_model for meta in metas):
            continue
        if any(int(meta.get("num_samples", -1)) != num_samples for meta in metas):
            continue
        if any(abs(float(meta.get("div_k", np.nan)) - 0.05) > 1e-12 for meta in metas):
            continue
        if any(int(meta.get("dds_k", -1)) != 5 for meta in metas):
            continue
        if any(not _same_path(meta.get("adapter_image_path", ""), expected_image_adapter) for meta in metas):
            continue
        if any(not _same_path(meta.get("adapter_text_path", ""), expected_text_adapter) for meta in metas):
            continue
        if not (
            np.array_equal(labels_sa, labels_div)
            and np.array_equal(labels_sa, labels_dds)
            and np.array_equal(indices_sa, indices_div)
            and np.array_equal(indices_sa, indices_dds)
        ):
            continue
        if labels_sa.shape != (num_samples,) or any(x.shape != (num_samples,) for x in (sa, div, dds)):
            continue
        if not np.array_equal(indices_sa, np.arange(num_samples, dtype=indices_sa.dtype)):
            continue
        matches.append((directory, {"sa": sa, "div": div, "dds": dds, "labels": labels_sa}))

    if not matches:
        raise FileNotFoundError(
            f"No matching static cache for experiment={paths.name}, dataset={dataset}, seed={seed}"
        )
    matches.sort(key=lambda x: (len(x[0].relative_to(paths.static_root).parts), str(x[0])))
    shortest = len(matches[0][0].relative_to(paths.static_root).parts)
    same_depth = [x for x in matches if len(x[0].relative_to(paths.static_root).parts) == shortest]
    if len(same_depth) > 1:
        raise RuntimeError("Multiple equally preferred static cache bundles:\n  " + "\n  ".join(str(x[0]) for x in same_depth))
    directory, scores = matches[0]
    print(f"[static] SA/Div/DDS cache hit: {directory}")
    return scores, directory


def _source_signature(log_dir: Path, layouts: list[FoldLayout]) -> str:
    records: list[dict[str, object]] = []
    for path in [log_dir / "meta.json"] + [layout.path for layout in layouts]:
        if path.is_file():
            stat = path.stat()
            records.append({"name": path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def a_component_cache_path(
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    component: str,
) -> Path:
    return (
        YYY_ROOT / "A_component_cache" / experiment / dataset / proxy_model
        / str(int(seed)) / str(int(epochs)) / f"{component}.npz"
    )


def _aggregate_training_values(
    fold_values: np.ndarray,
    layouts: list[FoldLayout],
    num_samples: int,
) -> np.ndarray:
    total = np.zeros(num_samples, dtype=np.float64)
    count = np.zeros(num_samples, dtype=np.int64)
    for fold_id, layout in enumerate(layouts):
        values = np.asarray(fold_values[fold_id, layout.train_indices], dtype=np.float32)
        finite = np.isfinite(values)
        indices = layout.train_indices[finite]
        total[indices] += values[finite].astype(np.float64)
        count[indices] += 1
    if np.any(count <= 0):
        missing = np.flatnonzero(count <= 0)
        raise ValueError(f"A aggregation missing samples: {missing[:10]}")
    return (total / count).astype(np.float32)


def _true_label_loss_and_probability(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 3:
        raise ValueError("train_logits must be 3D")
    if labels.shape != (logits.shape[1],):
        raise ValueError("labels do not align with train_logits")
    epochs, num_samples, _ = logits.shape
    loss = np.empty((epochs, num_samples), dtype=np.float32)
    probability = np.empty((epochs, num_samples), dtype=np.float32)
    sample_ids = np.arange(num_samples, dtype=np.int64)
    for start in range(0, epochs, chunk_size):
        end = min(epochs, start + chunk_size)
        chunk = np.asarray(logits[start:end], dtype=np.float32)
        max_logits = np.max(chunk, axis=2)
        shifted = chunk - max_logits[:, :, None]
        logsumexp = max_logits + np.log(np.sum(np.exp(np.clip(shifted, -50.0, 50.0)), axis=2))
        true_logits = chunk[:, sample_ids, labels]
        chunk_loss = logsumexp - true_logits
        loss[start:end] = chunk_loss.astype(np.float32)
        probability[start:end] = np.exp(-chunk_loss).astype(np.float32)
        del chunk, max_logits, shifted, logsumexp, true_logits, chunk_loss
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(probability)):
        raise ValueError("true-label loss/probability contains NaN or infinity")
    return loss, probability


def compute_a_subcomponents(
    *,
    layouts: list[FoldLayout],
    labels_all: np.ndarray,
    epochs: int,
    epoch_chunk_size: int,
) -> dict[str, DynamicComponentResult]:
    num_samples = labels_all.shape[0]
    num_folds = len(layouts)
    raw_foldwise = {
        name: np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        for name in A_SUBCOMPONENTS
    }
    fold_standardized = {
        name: np.full((num_folds, num_samples), np.nan, dtype=np.float32)
        for name in A_SUBCOMPONENTS
    }

    for fold_id, layout in enumerate(layouts):
        print(f"[A-components] fold={fold_id + 1}/{num_folds} read {layout.path}")
        train_idx = layout.train_indices
        y_train = labels_all[train_idx]
        with np.load(layout.path, allow_pickle=False) as data:
            train_logits = np.asarray(data["train_logits"][:epochs], dtype=np.float32)
        loss, true_prob = _true_label_loss_and_probability(
            train_logits,
            y_train,
            chunk_size=epoch_chunk_size,
        )
        del train_logits

        early_idx, mid_idx, late_idx = resolve_epoch_windows(epochs)
        early_mean = np.mean(true_prob[early_idx], axis=0)
        mid_mean = np.mean(true_prob[mid_idx], axis=0)
        boundary = np.mean(true_prob[early_idx] * (1.0 - true_prob[early_idx]), axis=0)
        support = np.sqrt(np.clip(early_mean * mid_mean, 0.0, 1.0))
        gain = support * (mid_mean - early_mean)
        stability = np.var(loss[late_idx], axis=0) - np.var(loss[mid_idx], axis=0)
        fold_raw = {
            "boundary": np.asarray(boundary, dtype=np.float32),
            "gain": np.asarray(gain, dtype=np.float32),
            "stability": np.asarray(stability, dtype=np.float32),
        }
        for name, values in fold_raw.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} invalid in fold {fold_id}")
            raw_foldwise[name][fold_id, train_idx] = values
            # Exact inner standardization used by the formal A implementation.
            fold_standardized[name][fold_id, train_idx] = safe_standardize(values)
        del loss, true_prob, early_mean, mid_mean, boundary, support, gain, stability

    out: dict[str, DynamicComponentResult] = {}
    for name in A_SUBCOMPONENTS:
        aggregated = _aggregate_training_values(fold_standardized[name], layouts, num_samples)
        final_normalized = standard_zscore_dynamic(aggregated)
        out[name] = DynamicComponentResult(
            raw_foldwise=raw_foldwise[name],
            fold_normalized=fold_standardized[name],
            aggregated=aggregated,
            final_normalized=final_normalized,
        )
    return out


def _save_a_component_cache(
    path: Path,
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    log_dir: Path,
    source_signature: str,
    labels: np.ndarray,
    component: str,
    result: DynamicComponentResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        experiment=np.array(experiment, dtype=np.str_),
        dataset=np.array(dataset, dtype=np.str_),
        proxy_model=np.array(proxy_model, dtype=np.str_),
        seed=np.array(seed, dtype=np.int64),
        epochs=np.array(epochs, dtype=np.int64),
        proxy_log_path=np.array(str(log_dir), dtype=np.str_),
        source_signature=np.array(source_signature, dtype=np.str_),
        component=np.array(component, dtype=np.str_),
        labels=labels.astype(np.int64, copy=False),
        raw_foldwise=result.raw_foldwise.astype(np.float32),
        fold_normalized=result.fold_normalized.astype(np.float32),
        aggregated=result.aggregated.astype(np.float32),
        final_normalized=result.final_normalized.astype(np.float32),
    )


def _load_a_component_cache(
    path: Path,
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    log_dir: Path,
    source_signature: str,
    labels: np.ndarray,
    component: str,
    num_folds: int,
) -> DynamicComponentResult | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            valid = (
                str(data["experiment"].item()) == experiment
                and str(data["dataset"].item()) == dataset
                and str(data["proxy_model"].item()) == proxy_model
                and int(data["seed"].item()) == seed
                and int(data["epochs"].item()) == epochs
                and str(data["proxy_log_path"].item()) == str(log_dir)
                and str(data["source_signature"].item()) == source_signature
                and str(data["component"].item()) == component
                and np.array_equal(np.asarray(data["labels"], dtype=np.int64), labels)
            )
            if not valid:
                return None
            result = DynamicComponentResult(
                raw_foldwise=np.asarray(data["raw_foldwise"], dtype=np.float32),
                fold_normalized=np.asarray(data["fold_normalized"], dtype=np.float32),
                aggregated=np.asarray(data["aggregated"], dtype=np.float32),
                final_normalized=np.asarray(data["final_normalized"], dtype=np.float32),
            )
    except Exception:
        return None
    n = labels.shape[0]
    if result.raw_foldwise.shape != (num_folds, n) or result.fold_normalized.shape != (num_folds, n):
        return None
    if result.aggregated.shape != (n,) or result.final_normalized.shape != (n,):
        return None
    if not np.all(np.isfinite(result.aggregated)) or not np.all(np.isfinite(result.final_normalized)):
        return None
    return result


def load_or_compute_a_subcomponents(
    *,
    experiment: str,
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    labels: np.ndarray,
    log_dir: Path,
    layouts: list[FoldLayout],
    force: bool,
    epoch_chunk_size: int,
) -> tuple[dict[str, DynamicComponentResult], Path]:
    signature = _source_signature(log_dir, layouts)
    loaded: dict[str, DynamicComponentResult] = {}
    if not force:
        for component in A_SUBCOMPONENTS:
            path = a_component_cache_path(
                experiment=experiment,
                dataset=dataset,
                proxy_model=proxy_model,
                seed=seed,
                epochs=epochs,
                component=component,
            )
            result = _load_a_component_cache(
                path,
                experiment=experiment,
                dataset=dataset,
                proxy_model=proxy_model,
                seed=seed,
                epochs=epochs,
                log_dir=log_dir,
                source_signature=signature,
                labels=labels,
                component=component,
                num_folds=len(layouts),
            )
            if result is None:
                loaded = {}
                break
            loaded[component] = result
    cache_dir = a_component_cache_path(
        experiment=experiment,
        dataset=dataset,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
        component="boundary",
    ).parent
    if len(loaded) == 3:
        print(f"[A-components] cache hit: {cache_dir}")
        return loaded, cache_dir

    print(f"[A-components] cache miss: compute from {log_dir}")
    computed = compute_a_subcomponents(
        layouts=layouts,
        labels_all=labels,
        epochs=epochs,
        epoch_chunk_size=epoch_chunk_size,
    )
    for component, result in computed.items():
        _save_a_component_cache(
            a_component_cache_path(
                experiment=experiment,
                dataset=dataset,
                proxy_model=proxy_model,
                seed=seed,
                epochs=epochs,
                component=component,
            ),
            experiment=experiment,
            dataset=dataset,
            proxy_model=proxy_model,
            seed=seed,
            epochs=epochs,
            log_dir=log_dir,
            source_signature=signature,
            labels=labels,
            component=component,
            result=result,
        )
    print(f"[A-components] cache saved: {cache_dir}")
    return computed, cache_dir


def build_a_variant(
    *,
    components: dict[str, DynamicComponentResult],
    names: tuple[str, ...],
    layouts: list[FoldLayout],
    num_samples: int,
) -> DynamicComponentResult:
    raw_foldwise = np.full((len(layouts), num_samples), np.nan, dtype=np.float32)
    fold_normalized = np.full((len(layouts), num_samples), np.nan, dtype=np.float32)
    for fold_id, layout in enumerate(layouts):
        train_idx = layout.train_indices
        values = np.zeros(train_idx.size, dtype=np.float32)
        for name in names:
            values += np.asarray(components[name].fold_normalized[fold_id, train_idx], dtype=np.float32)
        raw_foldwise[fold_id, train_idx] = values
        fold_normalized[fold_id, train_idx] = standard_zscore_dynamic(values)
    aggregated = _aggregate_training_values(fold_normalized, layouts, num_samples)
    final_normalized = standard_zscore_dynamic(aggregated)
    return DynamicComponentResult(raw_foldwise, fold_normalized, aggregated, final_normalized)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xx) & np.isfinite(yy)
    if finite.sum() < 2:
        return 0.0
    xx, yy = xx[finite], yy[finite]
    if np.std(xx) < 1e-12 or np.std(yy) < 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def build_dynamic_target(
    a_result: DynamicComponentResult,
    c_result: DynamicComponentResult,
    t_result: DynamicComponentResult,
) -> np.ndarray:
    raw = (
        np.asarray(a_result.final_normalized, dtype=np.float64)
        + np.asarray(c_result.final_normalized, dtype=np.float64)
        + np.asarray(t_result.final_normalized, dtype=np.float64)
    ) / 3.0
    return np.asarray(standard_zscore(raw), dtype=np.float64)


def learn_weights_in_memory(
    *,
    static_scores: dict[str, np.ndarray],
    dynamic_target: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, object]]:
    labels = np.asarray(static_scores["labels"], dtype=np.int64)
    features = np.stack(
        [
            standard_zscore_by_class(np.asarray(static_scores["sa"], dtype=np.float32), labels),
            standard_zscore_by_class(np.asarray(static_scores["div"], dtype=np.float32), labels),
            standard_zscore_by_class(np.asarray(static_scores["dds"], dtype=np.float32), labels),
        ],
        axis=1,
    ).astype(np.float64)
    fit = learn_weights_mod.fit_softplus_ratio_regression(
        features,
        dynamic_target,
        args.ratio_lambda,
        args.regression_learning_rate,
        args.regression_max_iter,
        args.regression_tol,
        device,
    )
    normalized = np.asarray(fit["normalized_weights"], dtype=np.float64)
    return {"sa": float(normalized[0]), "div": float(normalized[1]), "dds": float(normalized[2])}, fit


def print_a_component_corruption_means(
    dataset: str,
    components: dict[str, DynamicComponentResult],
    info,
) -> None:
    print(f"[corruption][{dataset}] A sub-component means after final z-score:")
    for type_id in range(corruption_mod.corruption_opt.NUM_CORRUPTION_TYPES):
        name = corruption_mod.corruption_opt.CORRUPTION_ID_TO_NAME[type_id]
        type_mask = info.corruption_types == type_id
        values = [
            f"{component}={float(np.mean(components[component].final_normalized[type_mask])):.6f}"
            for component in A_SUBCOMPONENTS
        ]
        print(f"  {name}: " + ", ".join(values))
    clean_mask = ~np.asarray(info.is_corrupted, dtype=bool)
    values = [
        f"{component}={float(np.mean(components[component].final_normalized[clean_mask])):.6f}"
        for component in A_SUBCOMPONENTS
    ]
    print("  clean(reference): " + ", ".join(values))


def print_corruption_retention(variant: str, mask: np.ndarray, info) -> None:
    selected = np.asarray(mask, dtype=np.uint8).astype(bool)
    selected_total = int(selected.sum())
    corrupted_mask = np.asarray(info.is_corrupted, dtype=bool)
    corrupted_total = int(corrupted_mask.sum())
    corrupted_selected = int(np.sum(selected & corrupted_mask))
    print(
        f"[corruption][A={variant}] corrupted_selected={corrupted_selected}/{corrupted_total}, "
        f"retention={corrupted_selected / corrupted_total:.6f}, "
        f"share_in_mask={corrupted_selected / selected_total:.6f}"
    )
    for type_id in range(corruption_mod.corruption_opt.NUM_CORRUPTION_TYPES):
        name = corruption_mod.corruption_opt.CORRUPTION_ID_TO_NAME[type_id]
        type_mask = info.corruption_types == type_id
        total = int(type_mask.sum())
        retained = int(np.sum(selected & type_mask))
        print(
            f"  {name}: retained={retained}/{total}, "
            f"retention={retained / total:.6f}, "
            f"share_in_mask={retained / selected_total:.6f}"
        )


def prepare_common_data(
    *,
    paths: ExperimentPaths,
    dataset: str,
    args: argparse.Namespace,
) -> tuple[
    dict[str, DynamicComponentResult],
    dict[str, DynamicComponentResult],
    np.ndarray,
    dict[str, np.ndarray],
    list[FoldLayout],
]:
    epochs = target_epochs(dataset)
    dynamic, labels, _ = load_dynamic_components(
        paths,
        dataset=dataset,
        proxy_model=args.proxy_model,
        seed=args.seed,
        epochs=epochs,
    )
    log_dir, layouts = resolve_proxy_logs(
        paths,
        dataset=dataset,
        proxy_model=args.proxy_model,
        seed=args.seed,
        required_epochs=epochs,
        k_folds=args.k_folds,
    )
    image_path, text_path = adapter_paths(paths, dataset, args.seed)
    if not image_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"Adapter files missing: {image_path}, {text_path}")
    static_scores, _ = load_static_scores_readonly(
        paths,
        dataset=dataset,
        seed=args.seed,
        clip_model=args.clip_model,
        num_samples=labels.shape[0],
        expected_image_adapter=image_path,
        expected_text_adapter=text_path,
    )
    if not np.array_equal(labels, np.asarray(static_scores["labels"], dtype=np.int64)):
        raise ValueError("Dynamic/static labels mismatch")
    a_components, _ = load_or_compute_a_subcomponents(
        experiment=paths.name,
        dataset=dataset,
        proxy_model=args.proxy_model,
        seed=args.seed,
        epochs=epochs,
        labels=labels,
        log_dir=log_dir,
        layouts=layouts,
        force=args.force_a_components,
        epoch_chunk_size=args.epoch_chunk_size,
    )

    reconstructed = build_a_variant(
        components=a_components,
        names=A_SUBCOMPONENTS,
        layouts=layouts,
        num_samples=labels.shape[0],
    )
    cached = np.asarray(dynamic["A"].final_normalized, dtype=np.float64)
    rebuilt = np.asarray(reconstructed.final_normalized, dtype=np.float64)
    print(
        "[A-check] all-three reconstruction vs cached A | "
        f"pearson={safe_pearson(cached, rebuilt):.8f}, "
        f"max_abs_diff={float(np.max(np.abs(cached - rebuilt))):.8e}"
    )

    # Attempt to load/compute R (from yyy/testR.py) and replace the cached T
    # component with an R-based component so that downstream pseudo-label
    # construction uses the R variant instead of the original T.
    try:
        r, u_in, u_out, signed_effect, r_path = testR_mod.load_or_compute_r(
            experiment=paths.name,
            dataset=dataset,
            proxy_model=args.proxy_model,
            seed=args.seed,
            epochs=epochs,
            labels_all=labels,
            log_dir=log_dir,
            layouts=layouts,
            force=False,
        )
        r_based_t = testR_mod.build_variant_component(
            variant="R",
            r=r,
            signed_effect=signed_effect,
            cached_t=dynamic["T"],
            layouts=layouts,
            num_samples=labels.shape[0],
        )
        dynamic["T"] = r_based_t
        print(f"[A/R] replaced cached T with R-based component: {r_path}")
    except Exception as exc:  # keep original T on failure
        print(f"[A/R] could not load/compute R; keeping cached T: {exc}")
    return dynamic, a_components, labels, static_scores, layouts


def run_normal_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> None:
    print(f"\n{'=' * 88}\nNORMAL | dataset={dataset} seed={args.seed}\n{'=' * 88}")
    paths = experiment_paths("normal")
    dynamic, a_components, labels, static_scores, layouts = prepare_common_data(
        paths=paths,
        dataset=dataset,
        args=args,
    )
    for variant, names in A_VARIANTS.items():
        a_variant = build_a_variant(
            components=a_components,
            names=names,
            layouts=layouts,
            num_samples=labels.shape[0],
        )
        target = build_dynamic_target(a_variant, dynamic["C"], dynamic["T"])
        weights, fit = learn_weights_in_memory(
            static_scores=static_scores,
            dynamic_target=target,
            device=device,
            args=args,
        )
        print(
            f"[normal][A={variant}] weights: "
            f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, DDS={weights['dds']:.6f}, "
            f"bias={float(fit['bias']):.6f}, mse={float(fit['mse']):.6e}"
        )


def run_corruption_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> None:
    print(
        f"\n{'=' * 88}\nCORRUPTION | dataset={dataset} seed={args.seed} kr={args.kr}\n{'=' * 88}"
    )
    paths = experiment_paths("corruption")
    dynamic, a_components, labels, static_scores, layouts = prepare_common_data(
        paths=paths,
        dataset=dataset,
        args=args,
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
        raise ValueError("Corruption labels do not match clean labels")
    print_a_component_corruption_means(dataset, a_components, info)

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

    encoded_features, encoded_labels = div_metric._encode_images(div_loader, image_adapter)
    original_encode = div_metric._encode_images

    def cached_encode(_loader, _adapter):
        return encoded_features, encoded_labels

    div_metric._encode_images = cached_encode  # type: ignore[method-assign]
    try:
        for variant, names in A_VARIANTS.items():
            a_variant = build_a_variant(
                components=a_components,
                names=names,
                layouts=layouts,
                num_samples=labels.shape[0],
            )
            target = build_dynamic_target(a_variant, dynamic["C"], dynamic["T"])
            weights, fit = learn_weights_in_memory(
                static_scores=static_scores,
                dynamic_target=target,
                device=device,
                args=args,
            )
            print(
                f"[corruption][A={variant}] weights: "
                f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, DDS={weights['dds']:.6f}, "
                f"bias={float(fit['bias']):.6f}, mse={float(fit['mse']):.6e}"
            )
            mask, selected_by_class, stats = mask_mod.select_group_mask_by_center_repair(
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
                dds_static_scores=np.asarray(static_scores["dds"], dtype=np.float32),
                group_candidate_pool_size=args.group_candidate_pool_size,
                group_init_count=args.group_init_count,
            )
            expected = int(round(labels.shape[0] * args.kr / 100.0))
            if int(mask.sum()) != expected or sum(selected_by_class.values()) != expected:
                raise RuntimeError(f"Mask size mismatch for A={variant}")
            print(
                f"[corruption][A={variant}] mask summary: selected={int(mask.sum())}, "
                f"distribution_shift={float(stats['distribution_shift']):.6f}, "
                f"subset_score={float(stats['subset_comprehensive_score']):.6f}"
            )
            print_corruption_retention(variant, mask, info)
    finally:
        div_metric._encode_images = original_encode  # type: ignore[method-assign]

    del encoded_features, encoded_labels, image_adapter, text_adapter, div_metric
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not 1 <= args.kr <= 100:
        raise ValueError("--kr must be in 1..100")
    if args.k_folds <= 1:
        raise ValueError("--k-folds must be greater than 1")
    if args.epoch_chunk_size <= 0:
        raise ValueError("--epoch-chunk-size must be positive")

    normal = parse_dataset_list(args.normal_datasets, DATASETS)
    corruption = parse_dataset_list(args.corruption_datasets, DATASETS)
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    print(
        f"device={device} seed={args.seed} kr={args.kr} "
        f"normal={normal} corruption={corruption}"
    )
    print(f"A component cache root: {YYY_ROOT / 'A_component_cache'}")
    print("Only A sub-components are saved; learned weights and masks remain in memory.")

    for dataset in normal:
        run_normal_dataset(args, dataset, device)
    for dataset in corruption:
        run_corruption_dataset(args, dataset, device)


if __name__ == "__main__":
    main()