#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure corruption sensitivity of dynamic components with AUROC.

This script reuses yyy/try_1.py for corruption metadata, cache paths, --keep
semantics, and truncated/full-run dynamic definitions. It does not fit static
weights and does not compute selection masks.

Positive class: corrupted image.

Expected corruption-detection directions:
- A, C, T: lower score means more likely corrupted -> evaluate -score
- final_risk: higher score means more likely corrupted -> evaluate +score
- noise_gate: lower value means more likely corrupted -> evaluate -gate
- pseudo_label: lower value means more likely corrupted -> evaluate -score

For every metric, the script reports:
- expected_auc: AUROC using the expected direction above
- separability_auc: max(expected_auc, 1-expected_auc), which ignores direction
- observed_relation:
    expected: the expected direction is correct
    reversed: the metric separates the groups in the opposite direction
    none: AUROC is approximately 0.5

Comparisons:
1. all corrupted samples vs all clean samples
2. each corruption type vs clean samples; other corruption types are excluded

Examples:
    python yyy/auc.py
    python yyy/auc.py --dataset cifar100
    python yyy/auc.py --keep A
    python yyy/auc.py --keep A,C,T,noise_gate
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
YYY_ROOT = THIS_FILE.parent
PROJECT_ROOT = YYY_ROOT.parent

for path in (YYY_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import yyy.try_2 as trial
except ImportError as exc:
    raise ImportError("yyy/auc.py requires yyy/try_1.py.") from exc


METRICS = (
    ("A", "lower=>corrupted", "Absorption Gain"),
    ("C", "lower=>corrupted", "Confusion Complementarity"),
    ("T", "lower=>corrupted", "Transferability Alignment"),
    ("final_risk", "higher=>corrupted", "Raw final risk"),
    ("noise_gate", "lower=>corrupted", "Noise gate"),
    ("pseudo_label", "lower=>corrupted", "Final gated pseudo-label"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute corruption-detection AUROC for dynamic components."
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", *trial.SUPPORTED_DATASETS),
    )
    parser.add_argument("--proxy-model", default=trial.PROXY_MODEL)
    parser.add_argument(
        "--keep",
        default="",
        help="Original full-run metrics to keep: A,C,T,noise_gate.",
    )
    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--special-gate-low", type=float, default=0.2)
    parser.add_argument("--special-gate-high", type=float, default=0.95)

    # Required by try_1.gate_thresholds, although the normal pair is unused.
    parser.add_argument("--normal-gate-low", type=float, default=0.1)
    parser.add_argument("--normal-gate-high", type=float, default=0.9)

    parser.add_argument("--force-dynamic", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=YYY_ROOT / "auc_results",
    )
    parser.add_argument("--no-save-sample-scores", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.keep_items = trial.parse_keep_items(args.keep)
    if args.learn_window <= 0:
        raise ValueError("--learn-window must be positive.")
    if not 0 < args.learn_min_correct <= args.learn_window:
        raise ValueError("--learn-min-correct must be in [1, learn-window].")
    for low, high, name in (
        (args.special_gate_low, args.special_gate_high, "special"),
        (args.normal_gate_low, args.normal_gate_high, "normal"),
    ):
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(
                f"{name} gate thresholds must satisfy 0 <= low < high <= 1."
            )


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with exact tie handling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Rank values must be a finite one-dimensional array.")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end + 1)
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware AUROC using the Mann-Whitney rank statistic."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors.")
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinity.")

    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = average_ranks(scores)
    rank_sum = float(ranks[positive].sum(dtype=np.float64))
    u_value = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(np.clip(u_value / (n_pos * n_neg), 0.0, 1.0))


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def detection_score(raw: np.ndarray, direction: str) -> np.ndarray:
    if direction == "higher=>corrupted":
        return np.asarray(raw, dtype=np.float64)
    if direction == "lower=>corrupted":
        return -np.asarray(raw, dtype=np.float64)
    raise ValueError(f"Unknown expected direction: {direction}")


def source_definition(metric: str, keep: frozenset[str]) -> str:
    if metric in {"A", "C", "T"}:
        return "original_full_run" if metric in keep else "truncated"
    if metric in {"final_risk", "noise_gate"}:
        return "original_full_run" if "noise_gate" in keep else "truncated"

    parts = [
        f"{name}={'full' if name in keep else 'truncated'}"
        for name in trial.COMPONENT_NAMES
    ]
    parts.append(
        f"gate={'full' if 'noise_gate' in keep else 'truncated'}"
    )
    states = {
        "full" if name in keep else "truncated"
        for name in (*trial.COMPONENT_NAMES, "noise_gate")
    }
    if len(states) == 1:
        return "original_full_run" if "full" in states else "truncated"
    return "mixed:" + ",".join(parts)


def build_metric_values(
    components: dict[str, Any],
    gate_data: dict[str, np.ndarray],
    pseudo_label: np.ndarray,
) -> dict[str, np.ndarray]:
    values = {
        "A": np.asarray(components["A"].final_normalized, dtype=np.float64),
        "C": np.asarray(components["C"].final_normalized, dtype=np.float64),
        "T": np.asarray(components["T"].final_normalized, dtype=np.float64),
        "final_risk": np.asarray(gate_data["final_risk"], dtype=np.float64),
        "noise_gate": np.asarray(gate_data["gate"], dtype=np.float64),
        "pseudo_label": np.asarray(pseudo_label, dtype=np.float64),
    }
    lengths = {name: array.size for name, array in values.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Metric lengths differ: {lengths}")
    for name, array in values.items():
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} is not a finite vector: {array.shape}")
    return values


def make_row(
    *,
    dataset: str,
    comparison: str,
    corruption_type: str,
    labels: np.ndarray,
    raw: np.ndarray,
    metric: str,
    direction: str,
    description: str,
    keep: frozenset[str],
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8)
    raw = np.asarray(raw, dtype=np.float64)
    auc = roc_auc(labels, detection_score(raw, direction))
    separation = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")

    if not np.isfinite(auc):
        relation = "undefined"
    elif auc > 0.5 + 1e-12:
        relation = "expected"
    elif auc < 0.5 - 1e-12:
        relation = "reversed"
    else:
        relation = "none"

    clean_stats = describe(raw[labels == 0])
    corrupt_stats = describe(raw[labels == 1])

    return {
        "dataset": dataset,
        "comparison": comparison,
        "corruption_type": corruption_type,
        "metric": metric,
        "description": description,
        "source_definition": source_definition(metric, keep),
        "expected_direction": direction,
        "observed_relation": relation,
        "n_clean": int(np.sum(labels == 0)),
        "n_corrupted": int(np.sum(labels == 1)),
        "expected_auc": auc,
        "separability_auc": separation,
        "raw_clean_mean": clean_stats["mean"],
        "raw_corrupted_mean": corrupt_stats["mean"],
        "raw_mean_difference": corrupt_stats["mean"] - clean_stats["mean"],
        "raw_clean_std": clean_stats["std"],
        "raw_corrupted_std": corrupt_stats["std"],
        "raw_clean_median": clean_stats["median"],
        "raw_corrupted_median": corrupt_stats["median"],
        "raw_clean_q25": clean_stats["q25"],
        "raw_clean_q75": clean_stats["q75"],
        "raw_corrupted_q25": corrupt_stats["q25"],
        "raw_corrupted_q75": corrupt_stats["q75"],
    }


def evaluate_dataset(
    dataset: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    ctx = trial.build_context("corruption", dataset)
    info = ctx.corruption_info
    if info is None:
        raise RuntimeError("Missing corruption metadata.")

    components, gate_data, pseudo_label = (
        trial.load_or_compute_dynamic_supervision(ctx, args)
    )
    values = build_metric_values(components, gate_data, pseudo_label)

    corruption_types = np.asarray(info.corruption_types, dtype=np.int16)
    is_corrupted = np.asarray(info.is_corrupted, dtype=bool)
    if corruption_types.shape != (ctx.labels.size,):
        raise ValueError("Corruption metadata and dynamic scores do not align.")
    if not np.array_equal(is_corrupted, corruption_types >= 0):
        raise ValueError("Corruption type flags are inconsistent.")

    type_names = dict(
        trial.corruption_mod.corruption_opt.CORRUPTION_ID_TO_NAME
    )
    rows: list[dict[str, Any]] = []

    overall_labels = is_corrupted.astype(np.uint8)
    for metric, direction, description in METRICS:
        rows.append(
            make_row(
                dataset=dataset,
                comparison="all_corrupted_vs_clean",
                corruption_type="all",
                labels=overall_labels,
                raw=values[metric],
                metric=metric,
                direction=direction,
                description=description,
                keep=args.keep_items,
            )
        )

    for type_id in sorted(type_names):
        selected = (corruption_types < 0) | (corruption_types == type_id)
        labels = (corruption_types[selected] == type_id).astype(np.uint8)
        type_name = str(type_names[type_id])
        for metric, direction, description in METRICS:
            rows.append(
                make_row(
                    dataset=dataset,
                    comparison="single_type_vs_clean",
                    corruption_type=type_name,
                    labels=labels,
                    raw=values[metric][selected],
                    metric=metric,
                    direction=direction,
                    description=description,
                    keep=args.keep_items,
                )
            )

    sample_arrays = {
        "labels": np.asarray(ctx.labels, dtype=np.int64),
        "is_corrupted": is_corrupted.astype(np.uint8),
        "corruption_types": corruption_types,
        **{
            name: np.asarray(array, dtype=np.float32)
            for name, array in values.items()
        },
    }
    metadata = {
        "dataset": dataset,
        "experiment": "corruption",
        "seed": int(trial.SEED),
        "proxy_model": args.proxy_model,
        "truncated_epochs": int(trial.DYNAMIC_EPOCHS[dataset]),
        "full_run_epochs": int(trial.SOURCE_PROXY_EPOCHS[dataset]),
        "keep_original_metrics": sorted(args.keep_items),
        "learn_window": int(args.learn_window),
        "learn_min_correct": int(args.learn_min_correct),
        "special_gate_low": float(args.special_gate_low),
        "special_gate_high": float(args.special_gate_high),
        "num_samples": int(ctx.labels.size),
        "num_clean": int((~is_corrupted).sum()),
        "num_corrupted": int(is_corrupted.sum()),
        "corruption_type_counts": {
            str(type_names[type_id]): int(np.sum(corruption_types == type_id))
            for type_id in sorted(type_names)
        },
    }
    return rows, sample_arrays, metadata


def keep_tag(keep: frozenset[str]) -> str:
    ordered = [
        name
        for name in (*trial.COMPONENT_NAMES, "noise_gate")
        if name in keep
    ]
    return "keep_none" if not ordered else "keep_" + "-".join(ordered)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
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


def print_tables(rows: list[dict[str, Any]], dataset: str) -> None:
    overall = [
        row for row in rows
        if row["comparison"] == "all_corrupted_vs_clean"
    ]
    print(f"\n[{dataset}] all corrupted vs clean")
    print(
        f"{'metric':<14}{'source':<20}{'expected AUC':>14}"
        f"{'separate AUC':>15}{'relation':>12}"
        f"{'clean mean':>14}{'corrupt mean':>16}"
    )
    print("-" * 105)
    for row in overall:
        print(
            f"{row['metric']:<14}"
            f"{row['source_definition'][:19]:<20}"
            f"{row['expected_auc']:>14.6f}"
            f"{row['separability_auc']:>15.6f}"
            f"{row['observed_relation']:>12}"
            f"{row['raw_clean_mean']:>14.6f}"
            f"{row['raw_corrupted_mean']:>16.6f}"
        )

    per_type = [
        row for row in rows
        if row["comparison"] == "single_type_vs_clean"
    ]
    names: list[str] = []
    for row in per_type:
        name = str(row["corruption_type"])
        if name not in names:
            names.append(name)

    metric_names = [item[0] for item in METRICS]
    lookup = {
        (row["corruption_type"], row["metric"]): row["expected_auc"]
        for row in per_type
    }
    print(f"\n[{dataset}] each corruption type vs clean: expected AUROC")
    header = f"{'corruption type':<26}" + "".join(
        f"{metric:>14}" for metric in metric_names
    )
    print(header)
    print("-" * len(header))
    for name in names:
        line = f"{name:<26}"
        for metric in metric_names:
            line += f"{lookup[(name, metric)]:>14.6f}"
        print(line)


def main() -> None:
    args = parse_args()
    validate_args(args)
    datasets = (
        list(trial.SUPPORTED_DATASETS)
        if args.dataset == "all"
        else [args.dataset]
    )

    print("=" * 108)
    print(
        f"Corruption AUROC | seed={trial.SEED} | "
        f"proxy_model={args.proxy_model}"
    )
    print(f"datasets={datasets}")
    print(f"keep original metrics={sorted(args.keep_items)}")
    print("Positive class=corrupted image.")
    print("=" * 108)

    all_rows: list[dict[str, Any]] = []
    all_metadata: dict[str, Any] = {}

    for dataset in tqdm(datasets, desc="AUC datasets", unit="dataset"):
        rows, arrays, metadata = evaluate_dataset(dataset, args)
        all_rows.extend(rows)
        all_metadata[dataset] = metadata

        out_dir = (
            args.output_root
            / dataset
            / str(trial.DYNAMIC_EPOCHS[dataset])
            / keep_tag(args.keep_items)
        )
        csv_path = out_dir / "auc_results.csv"
        json_path = out_dir / "auc_results.json"
        npz_path = out_dir / "sample_scores.npz"

        write_csv(csv_path, rows)
        json_path.write_text(
            json.dumps(
                json_safe({"metadata": metadata, "results": rows}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not args.no_save_sample_scores:
            np.savez_compressed(npz_path, **arrays)

        print_tables(rows, dataset)
        print(f"\n[saved] {csv_path}")
        print(f"[saved] {json_path}")
        if not args.no_save_sample_scores:
            print(f"[saved] {npz_path}")

    combined_dir = args.output_root / "combined" / keep_tag(args.keep_items)
    combined_csv = combined_dir / "auc_results.csv"
    combined_json = combined_dir / "auc_results.json"
    write_csv(combined_csv, all_rows)
    combined_json.write_text(
        json.dumps(
            json_safe({"metadata": all_metadata, "results": all_rows}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 108)
    print(f"Combined CSV: {combined_csv}")
    print(f"Combined JSON: {combined_json}")
    print("No scoring weights or selection masks were computed.")
    print("=" * 108)


if __name__ == "__main__":
    main()