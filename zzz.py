#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze cached static scores of corrupted samples.

This script is intended to be placed in the project root and run directly:

    python zzz.py

Default behavior:
- datasets: CIFAR-100 and Tiny-ImageNet;
- seed: 22;
- reads the existing corruption manifest, static-score caches and learned weights;
- never recomputes SA/Div/DDS;
- applies the project's class-wise standard z-score over the complete training set;
- calculates each corruption type separately;
- reports population mean and variance for SA, Div, DDS and the learned-weight total;
- saves CSV and JSON summaries under ``corruption_exp/analysis``.

The total score is

    total = w_sa * z_sa + w_div * z_div + w_dds * z_dds

where all z-scores are computed class-wise on the complete training set, matching
``utils.score_utils.standard_zscore_by_class``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.score_utils import standard_zscore_by_class
except ImportError as exc:  # pragma: no cover - only used for a clearer runtime error
    raise ImportError(
        "zzz.py must be placed in the project root so that "
        "utils.score_utils can be imported."
    ) from exc

CORRUPTION_DATA_ROOT = PROJECT_ROOT / "corruption_data"
CORRUPTION_EXP_ROOT = PROJECT_ROOT / "corruption_exp"
STATIC_SCORE_ROOT = CORRUPTION_EXP_ROOT / "static_scores"
WEIGHTS_PATH = CORRUPTION_EXP_ROOT / "weights" / "scoring_weights.json"
DEFAULT_OUTPUT_DIR = CORRUPTION_EXP_ROOT / "analysis"

SUPPORTED_DATASETS = ("cifar100", "tiny-imagenet")
EXPECTED_TRAIN_SIZES = {"cifar100": 50_000, "tiny-imagenet": 100_000}
CORRUPTION_ID_TO_NAME = {
    0: "gaussian_noise",
    1: "partial_occlusion",
    2: "resolution_degradation",
    3: "fog",
    4: "motion_blur",
}
METRIC_KEYS = ("sa", "div", "dds")
CACHE_FILENAMES = {
    "sa": "SA_cache.npz",
    "div": "Div_cache.npz",
    "dds": "DDS_cache.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize cached corruption SA/Div/DDS scores after class-wise "
            "z-score standardization; no static score is recomputed."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(SUPPORTED_DATASETS),
        help="Comma-separated dataset names. Default: cifar100,tiny-imagenet.",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument(
        "--cache-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help=(
            "Override the static-score search directory for one dataset. "
            "Repeat this option for multiple datasets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save CSV and JSON summaries.",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=(0, 1),
        default=0,
        help="Variance ddof. Default 0 means population variance.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print the result; do not write CSV/JSON files.",
    )
    return parser.parse_args()


def parse_datasets(text: str) -> list[str]:
    datasets = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not datasets:
        raise ValueError("--datasets cannot be empty")
    unknown = sorted(set(datasets) - set(SUPPORTED_DATASETS))
    if unknown:
        raise ValueError(
            f"unsupported datasets: {unknown}; supported={SUPPORTED_DATASETS}"
        )
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"duplicate datasets in --datasets: {datasets}")
    return datasets


def parse_cache_overrides(items: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --cache-dir {item!r}; expected DATASET=PATH")
        dataset, path_text = item.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"unsupported cache override dataset: {dataset!r}")
        if dataset in overrides:
            raise ValueError(f"duplicate cache override for {dataset}")
        path = Path(path_text.strip()).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        overrides[dataset] = path.resolve()
    return overrides


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_corruption_manifest(dataset: str, seed: int) -> dict[str, Any]:
    path = CORRUPTION_DATA_ROOT / dataset / f"corruption_list_{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"corruption manifest not found: {path}")

    rows = np.loadtxt(path, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(f"invalid corruption manifest shape {rows.shape}: {path}")

    sample_ids = rows[:, 0]
    type_ids = rows[:, 1]
    num_samples = EXPECTED_TRAIN_SIZES[dataset]
    expected_total = int(round(num_samples * 0.2))
    expected_per_type = expected_total // len(CORRUPTION_ID_TO_NAME)

    if rows.shape[0] != expected_total:
        raise ValueError(
            f"{dataset}: corrupted sample count={rows.shape[0]}, "
            f"expected={expected_total}"
        )
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError(f"duplicate sample ids in {path}")
    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError(f"sample id out of range in {path}")
    known_ids = set(CORRUPTION_ID_TO_NAME)
    if not set(np.unique(type_ids).tolist()).issubset(known_ids):
        raise ValueError(f"unknown corruption type id in {path}")

    corruption_types = np.full(num_samples, -1, dtype=np.int16)
    corruption_types[sample_ids] = type_ids.astype(np.int16)
    counts = {
        CORRUPTION_ID_TO_NAME[type_id]: int(np.sum(type_ids == type_id))
        for type_id in CORRUPTION_ID_TO_NAME
    }
    if any(count != expected_per_type for count in counts.values()):
        raise ValueError(
            f"each corruption type should have {expected_per_type} samples; "
            f"got {counts}"
        )

    return {
        "path": path,
        "hash": sha1_file(path),
        "num_samples": num_samples,
        "corruption_types": corruption_types,
        "counts": counts,
    }


def load_learned_weights(dataset: str, seed: int) -> dict[str, Any]:
    data = load_json_object(WEIGHTS_PATH)
    dataset_entry = data.get(dataset)
    if not isinstance(dataset_entry, dict):
        raise KeyError(f"missing weight dataset entry: {dataset}")
    entry = dataset_entry.get(str(seed))
    if not isinstance(entry, dict):
        raise KeyError(f"missing learned weights for dataset={dataset}, seed={seed}")

    weights: dict[str, float] = {}
    for key in METRIC_KEYS:
        try:
            value = float(entry[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid learned weight: {dataset}/{seed}/{key}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"weight must be finite and positive: {key}={value}")
        weights[key] = value

    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"weights do not sum to 1: sum={total}, weights={weights}")

    context = entry.get("corruption_context")
    return {
        "weights": weights,
        "context": context if isinstance(context, dict) else {},
    }


def directories_with_cache_triplet(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    if all((root / filename).is_file() for filename in CACHE_FILENAMES.values()):
        return [root]

    candidates: set[Path] = set()
    for sa_path in root.rglob(CACHE_FILENAMES["sa"]):
        parent = sa_path.parent
        if all((parent / filename).is_file() for filename in CACHE_FILENAMES.values()):
            candidates.add(parent)
    return sorted(candidates)


def load_metric_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"scores", "labels", "indices", "meta"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        scores = np.asarray(archive["scores"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        indices = np.asarray(archive["indices"], dtype=np.int64)
        meta_text = str(np.asarray(archive["meta"]).item())

    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata JSON in {path}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"cache metadata is not an object: {path}")
    return {"scores": scores, "labels": labels, "indices": indices, "meta": meta}


def load_cache_directory(
    cache_dir: Path,
    dataset: str,
    seed: int,
    num_samples: int,
) -> dict[str, Any]:
    loaded = {
        key: load_metric_cache(cache_dir / filename)
        for key, filename in CACHE_FILENAMES.items()
    }
    reference = loaded["sa"]
    expected_indices = np.arange(num_samples, dtype=np.int64)

    for key, item in loaded.items():
        if item["scores"].shape != (num_samples,):
            raise ValueError(f"{key} score shape mismatch in {cache_dir}")
        if item["labels"].shape != (num_samples,):
            raise ValueError(f"{key} label shape mismatch in {cache_dir}")
        if not np.array_equal(item["indices"], expected_indices):
            raise ValueError(f"{key} indices are not 0..N-1 in {cache_dir}")
        if not np.array_equal(item["labels"], reference["labels"]):
            raise ValueError(f"labels differ across metrics in {cache_dir}")
        if item["meta"] != reference["meta"]:
            raise ValueError(f"metadata differs across metrics in {cache_dir}")

        meta = item["meta"]
        if meta.get("dataset") != dataset:
            raise ValueError(f"{key} cache dataset mismatch in {cache_dir}")
        if int(meta.get("seed", -1)) != seed:
            raise ValueError(f"{key} cache seed mismatch in {cache_dir}")
        if int(meta.get("num_samples", -1)) != num_samples:
            raise ValueError(f"{key} cache sample count mismatch in {cache_dir}")

    return {
        "cache_dir": cache_dir,
        "labels": reference["labels"],
        "meta": reference["meta"],
        "scores": {key: loaded[key]["scores"] for key in METRIC_KEYS},
    }


def cache_matches_weight_context(
    cache_meta: dict[str, Any], weight_context: dict[str, Any]
) -> bool:
    """Match fields shared by the learned-weight context and static cache metadata."""
    shared_keys = (
        "clip_model",
        "adapter_image_sha1",
        "adapter_text_sha1",
    )
    compared = False
    for key in shared_keys:
        expected = weight_context.get(key)
        if expected in (None, ""):
            continue
        compared = True
        if cache_meta.get(key) != expected:
            return False
    return compared


def resolve_static_cache(
    dataset: str,
    seed: int,
    manifest_hash: str,
    num_samples: int,
    weight_context: dict[str, Any],
    override: Path | None,
) -> dict[str, Any]:
    if override is None:
        search_root = STATIC_SCORE_ROOT / dataset / str(seed)
        marker = f"corruption_{manifest_hash[:12]}"
        candidates = [
            path
            for path in directories_with_cache_triplet(search_root)
            if marker in path.parts
        ]
    else:
        search_root = override
        candidates = directories_with_cache_triplet(search_root)

    if not candidates:
        raise FileNotFoundError(
            f"no cached SA/Div/DDS triplet found for {dataset}, seed={seed} "
            f"under {search_root}. zzz.py will not recompute static scores."
        )

    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            valid.append(load_cache_directory(candidate, dataset, seed, num_samples))
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if not valid:
        detail = "\n".join(f"  - {message}" for message in errors)
        raise RuntimeError(f"all discovered cache directories are invalid:\n{detail}")

    context_matches = [
        item
        for item in valid
        if cache_matches_weight_context(item["meta"], weight_context)
    ]
    if len(context_matches) == 1:
        return context_matches[0]
    if len(valid) == 1:
        return valid[0]

    choices = "\n".join(f"  - {item['cache_dir']}" for item in valid)
    raise RuntimeError(
        f"multiple valid static-score cache directories found for {dataset}, seed={seed}:\n"
        f"{choices}\nUse --cache-dir {dataset}=PATH to choose one explicitly."
    )


def summarize_dataset(
    dataset: str,
    seed: int,
    ddof: int,
    cache_override: Path | None,
) -> dict[str, Any]:
    manifest = load_corruption_manifest(dataset, seed)
    weight_info = load_learned_weights(dataset, seed)
    weights = weight_info["weights"]
    weight_context = weight_info["context"]

    context_hash = weight_context.get("corruption_list_hash")
    if context_hash is not None and str(context_hash) != manifest["hash"]:
        raise ValueError(
            f"learned-weight corruption hash mismatch for {dataset}/{seed}: "
            f"weights={context_hash}, manifest={manifest['hash']}"
        )

    cache = resolve_static_cache(
        dataset=dataset,
        seed=seed,
        manifest_hash=manifest["hash"],
        num_samples=manifest["num_samples"],
        weight_context=weight_context,
        override=cache_override,
    )

    labels = cache["labels"]
    raw_scores = cache["scores"]
    zscores = {
        key: standard_zscore_by_class(raw_scores[key], labels).astype(np.float64)
        for key in METRIC_KEYS
    }
    contributions = {key: weights[key] * zscores[key] for key in METRIC_KEYS}
    total_scores = sum(contributions.values())

    rows: list[dict[str, Any]] = []
    corruption_types = manifest["corruption_types"]
    for type_id, corruption_name in CORRUPTION_ID_TO_NAME.items():
        mask = corruption_types == type_id
        count = int(mask.sum())
        if count <= ddof:
            raise ValueError(
                f"not enough samples for variance: {dataset}/{corruption_name}, "
                f"count={count}, ddof={ddof}"
            )

        means = {
            key: float(np.mean(zscores[key][mask], dtype=np.float64))
            for key in METRIC_KEYS
        }
        variances = {
            key: float(np.var(zscores[key][mask], ddof=ddof, dtype=np.float64))
            for key in METRIC_KEYS
        }
        contribution_means = {
            key: float(np.mean(contributions[key][mask], dtype=np.float64))
            for key in METRIC_KEYS
        }
        dominant_raw = max(METRIC_KEYS, key=lambda key: abs(means[key]))
        dominant_weighted = max(
            METRIC_KEYS, key=lambda key: abs(contribution_means[key])
        )

        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "corruption_id": type_id,
                "corruption": corruption_name,
                "num_samples": count,
                "sa_mean": means["sa"],
                "sa_var": variances["sa"],
                "div_mean": means["div"],
                "div_var": variances["div"],
                "dds_mean": means["dds"],
                "dds_var": variances["dds"],
                "total_mean": float(
                    np.mean(total_scores[mask], dtype=np.float64)
                ),
                "total_var": float(
                    np.var(total_scores[mask], ddof=ddof, dtype=np.float64)
                ),
                "sa_weighted_mean": contribution_means["sa"],
                "div_weighted_mean": contribution_means["div"],
                "dds_weighted_mean": contribution_means["dds"],
                "dominant_metric_by_abs_z_mean": dominant_raw.upper(),
                "dominant_metric_by_abs_weighted_mean": dominant_weighted.upper(),
            }
        )

    return {
        "dataset": dataset,
        "seed": seed,
        "manifest_path": str(manifest["path"]),
        "manifest_hash": manifest["hash"],
        "cache_dir": str(cache["cache_dir"]),
        "cache_meta": cache["meta"],
        "weights": weights,
        "standardization": (
            "class-wise standard z-score over the complete training set"
        ),
        "variance_ddof": ddof,
        "rows": rows,
    }


def print_dataset_summary(result: dict[str, Any]) -> None:
    weights = result["weights"]
    print("\n" + "=" * 154)
    print(f"Dataset: {result['dataset']} | seed={result['seed']}")
    print(f"Cache:   {result['cache_dir']}")
    print(
        "Weights: "
        f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
        f"DDS={weights['dds']:.6f}"
    )
    print("Standardization: class-wise z-score over all training samples")
    print(f"Variance ddof: {result['variance_ddof']}")

    header = (
        f"{'corruption':<24} {'n':>7} "
        f"{'SA mean':>10} {'SA var':>10} "
        f"{'Div mean':>10} {'Div var':>10} "
        f"{'DDS mean':>10} {'DDS var':>10} "
        f"{'Total mean':>11} {'Total var':>11} "
        f"{'raw shift':>10} {'weighted':>10}"
    )
    print(header)
    print("-" * 154)
    for row in result["rows"]:
        print(
            f"{row['corruption']:<24} {row['num_samples']:>7d} "
            f"{row['sa_mean']:>10.6f} {row['sa_var']:>10.6f} "
            f"{row['div_mean']:>10.6f} {row['div_var']:>10.6f} "
            f"{row['dds_mean']:>10.6f} {row['dds_var']:>10.6f} "
            f"{row['total_mean']:>11.6f} {row['total_var']:>11.6f} "
            f"{row['dominant_metric_by_abs_z_mean']:>10} "
            f"{row['dominant_metric_by_abs_weighted_mean']:>10}"
        )


def save_results(
    results: list[dict[str, Any]], output_dir: Path, seed: int
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"corruption_static_metric_summary_seed{seed}.csv"
    json_path = output_dir / f"corruption_static_metric_summary_seed{seed}.json"

    flat_rows = [row for result in results for row in result["rows"]]
    fieldnames = [
        "dataset",
        "seed",
        "corruption_id",
        "corruption",
        "num_samples",
        "sa_mean",
        "sa_var",
        "div_mean",
        "div_var",
        "dds_mean",
        "dds_var",
        "total_mean",
        "total_var",
        "sa_weighted_mean",
        "div_weighted_mean",
        "dds_weighted_mean",
        "dominant_metric_by_abs_z_mean",
        "dominant_metric_by_abs_weighted_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    payload = {
        "seed": seed,
        "standardization": (
            "class-wise standard z-score over the complete training set"
        ),
        "variance_ddof": results[0]["variance_ddof"] if results else 0,
        "datasets": {result["dataset"]: result for result in results},
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.datasets)
    cache_overrides = parse_cache_overrides(args.cache_dir)
    unused = sorted(set(cache_overrides) - set(datasets))
    if unused:
        raise ValueError(f"cache overrides supplied for unselected datasets: {unused}")

    results: list[dict[str, Any]] = []
    for dataset in datasets:
        result = summarize_dataset(
            dataset=dataset,
            seed=args.seed,
            ddof=args.ddof,
            cache_override=cache_overrides.get(dataset),
        )
        results.append(result)
        print_dataset_summary(result)

    if not args.no_save:
        csv_path, json_path = save_results(results, args.output_dir, args.seed)
        print(f"\nCSV saved to:  {csv_path}")
        print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()