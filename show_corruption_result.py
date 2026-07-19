#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show numerical tables for fixed image-corruption experiments.

Run from the project root:

    python show_corruption_result.py

The script prints one terminal table containing:
- CIFAR-100 and Tiny-ImageNet test accuracy at kr=30/50/70;
- the percentage of corrupted samples in each selected subset.

Accuracy is aggregated across seeds as mean±std. The script first uses the mean
of the last 10 values in ``accuracy_samples`` and otherwise falls back to a
recognized scalar accuracy field.

The corrupted-sample ratio is recomputed from the actual mask and:

    corruption_data/<dataset>/corruption_list_<seed>.txt

Expected result path:

    <result_root>/<method>/<dataset>/<model>/<seed>/result_<kr>.json

Expected mask path:

    <mask_root>/<method>/<dataset>/<seed>/mask_<kr>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Sequence

import numpy as np


DEFAULT_DATASETS = ["cifar100", "tiny-imagenet"]
DEFAULT_SEEDS = [22, 42, 96]
DEFAULT_KR = [30, 50, 70]
DEFAULT_MODEL = "resnet50"
DEFAULT_RESULT_ROOTS = ["result", "corruption_exp/result", "result_corruption"]
DEFAULT_MASK_ROOTS = ["mask", "corruption_exp/mask"]

PREFERRED_METHOD_ORDER = [
    "corruption_random",
    "corruption_EL2N",
    "corruption_Forgetting",
    "corruption_GraNd",
    "corruption_herding",
    "corruption_Herding",
    "corruption_MDS",
    "corruption_MoSo",
    "corruption_yangclip",
    "corruption_YangCLIP",
    "corruption_RLSelector",
    "corruption_naive_group",
    "corruption_learned_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print fixed image-corruption experiment result tables."
    )
    parser.add_argument(
        "--result-roots",
        default=",".join(DEFAULT_RESULT_ROOTS),
        help=(
            "Comma-separated result roots searched in order. "
            "Default: result,corruption_exp/result,result_corruption"
        ),
    )
    parser.add_argument(
        "--mask-roots",
        default=",".join(DEFAULT_MASK_ROOTS),
        help=(
            "Comma-separated mask roots searched in order. "
            "Default: mask,corruption_exp/mask"
        ),
    )
    parser.add_argument(
        "--corruption-root",
        default="corruption_data",
        help="Root containing corruption_list files. Default: corruption_data",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets. Default: cifar100,tiny-imagenet",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 22,42,96",
    )
    parser.add_argument(
        "--kr",
        default=",".join(str(kr) for kr in DEFAULT_KR),
        help="Comma-separated keep ratios. Default: 30,50,70",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name used in result paths. Default: resnet50",
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Optional comma-separated method directory names. When omitted, "
            "all corruption_* directories are discovered."
        ),
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Display method names with the corruption_ prefix.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error when an expected result, mask, or list is missing.",
    )
    return parser.parse_args()


def parse_csv_str(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_int(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Integer list argument cannot be empty.")
    return values


def display_method_name(method: str, keep_prefix: bool = False) -> str:
    if keep_prefix:
        return method
    prefix = "corruption_"
    return method[len(prefix) :] if method.startswith(prefix) else method


def sort_methods(methods: Sequence[str]) -> list[str]:
    order = {method: index for index, method in enumerate(PREFERRED_METHOD_ORDER)}
    return sorted(
        dict.fromkeys(methods),
        key=lambda method: (order.get(method, 10_000), method.lower()),
    )


def discover_methods(
    result_roots: Sequence[Path],
    mask_roots: Sequence[Path],
    explicit_methods: Sequence[str],
) -> list[str]:
    if explicit_methods:
        return sort_methods(explicit_methods)

    methods: set[str] = set()
    for root in [*result_roots, *mask_roots]:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("corruption_"):
                methods.add(path.name)
    return sort_methods(methods)


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    return next((path for path in paths if path.is_file()), None)


def result_candidates(
    roots: Sequence[Path],
    method: str,
    dataset: str,
    model: str,
    seed: int,
    kr: int,
) -> list[Path]:
    return [
        root / method / dataset / model / str(seed) / f"result_{int(kr)}.json"
        for root in roots
    ]


def mask_candidates(
    roots: Sequence[Path],
    method: str,
    dataset: str,
    seed: int,
    kr: int,
) -> list[Path]:
    return [
        root / method / dataset / str(seed) / f"mask_{int(kr)}.npz"
        for root in roots
    ]


def load_accuracy(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[WARN] failed to read result JSON: {path} ({exc})")
        return None

    samples = payload.get("accuracy_samples")
    if isinstance(samples, list) and samples:
        try:
            values = [float(value) for value in samples[-10:]]
            if np.all(np.isfinite(values)):
                return float(mean(values))
        except (TypeError, ValueError):
            pass

    for key in (
        "accuracy",
        "accuracy_mean",
        "test_acc",
        "test_accuracy",
        "last10_mean_acc",
        "accuracy_mean_last10",
    ):
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value

    print(f"[WARN] no recognizable finite accuracy field in {path}")
    return None


def aggregate_accuracy(
    roots: Sequence[Path],
    method: str,
    dataset: str,
    model: str,
    seeds: Sequence[int],
    kr: int,
    strict: bool,
) -> Optional[tuple[float, float, int]]:
    values: list[float] = []
    for seed in seeds:
        candidates = result_candidates(roots, method, dataset, model, seed, kr)
        path = first_existing(candidates)
        if path is None:
            message = (
                f"result missing: method={method}, dataset={dataset}, "
                f"seed={seed}, kr={kr}"
            )
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}")
            continue
        value = load_accuracy(path)
        if value is not None:
            values.append(value)
        elif strict:
            raise ValueError(f"Invalid result JSON: {path}")

    if not values:
        return None
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)


def load_mask(path: Path) -> Optional[np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "mask" in data.files:
                mask = np.asarray(data["mask"]).reshape(-1)
            elif len(data.files) == 1:
                mask = np.asarray(data[data.files[0]]).reshape(-1)
            else:
                print(f"[WARN] no unique mask array in {path}")
                return None
    except Exception as exc:
        print(f"[WARN] failed to read mask: {path} ({exc})")
        return None

    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1, False, True}):
        print(f"[WARN] non-binary mask: {path}, values={sorted(values)}")
        return None
    return mask.astype(bool, copy=False)


def read_corrupted_ids(root: Path, dataset: str, seed: int) -> Optional[np.ndarray]:
    path = root / dataset / f"corruption_list_{int(seed)}.txt"
    if not path.is_file():
        print(f"[WARN] corruption list not found: {path}")
        return None
    try:
        rows = np.loadtxt(path, dtype=np.int64)
    except Exception as exc:
        print(f"[WARN] failed to read corruption list: {path} ({exc})")
        return None

    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 2:
        print(f"[WARN] corruption list must have two columns: {path}")
        return None

    sample_ids = rows[:, 0].astype(np.int64)
    type_ids = rows[:, 1].astype(np.int64)
    if np.unique(sample_ids).size != sample_ids.size:
        print(f"[WARN] duplicate sample IDs in {path}")
        return None
    if np.any(type_ids < 0) or np.any(type_ids > 4):
        print(f"[WARN] invalid corruption type in {path}")
        return None
    return sample_ids


def corruption_ratio_for_mask(
    mask_roots: Sequence[Path],
    corruption_root: Path,
    method: str,
    dataset: str,
    seed: int,
    kr: int,
    strict: bool,
) -> Optional[float]:
    candidates = mask_candidates(mask_roots, method, dataset, seed, kr)
    mask_path = first_existing(candidates)
    if mask_path is None:
        message = (
            f"mask missing: method={method}, dataset={dataset}, "
            f"seed={seed}, kr={kr}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")
        return None

    mask = load_mask(mask_path)
    if mask is None:
        if strict:
            raise ValueError(f"Invalid mask: {mask_path}")
        return None

    expected = int(round(mask.size * kr / 100.0))
    if int(mask.sum()) != expected:
        message = (
            f"mask selected count mismatch: {mask_path}, "
            f"actual={int(mask.sum())}, expected={expected}"
        )
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}")
        return None

    ids = read_corrupted_ids(corruption_root, dataset, seed)
    if ids is None:
        if strict:
            raise ValueError(f"Invalid corruption list: dataset={dataset}, seed={seed}")
        return None
    if np.any(ids < 0) or np.any(ids >= mask.size):
        message = f"corruption IDs out of range for {mask_path}"
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}")
        return None

    is_corrupted = np.zeros(mask.size, dtype=bool)
    is_corrupted[ids] = True
    return float(is_corrupted[mask].mean() * 100.0)


def aggregate_corruption_ratio(
    mask_roots: Sequence[Path],
    corruption_root: Path,
    method: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    kr: int,
    strict: bool,
) -> Optional[tuple[float, float, int]]:
    values: list[float] = []
    for dataset in datasets:
        for seed in seeds:
            value = corruption_ratio_for_mask(
                mask_roots,
                corruption_root,
                method,
                dataset,
                seed,
                kr,
                strict,
            )
            if value is not None:
                values.append(value)
    if not values:
        return None
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)


def format_cell(
    stats: Optional[tuple[float, float, int]],
    expected_count: int,
    decimals: int,
) -> str:
    if stats is None:
        return "-"
    mean_value, std_value, count = stats
    cell = f"{mean_value:.{decimals}f}±{std_value:.{decimals}f}"
    if count < expected_count:
        cell += f"({count})"
    return cell


def build_rows(
    result_roots: Sequence[Path],
    mask_roots: Sequence[Path],
    corruption_root: Path,
    methods: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    keep_ratios: Sequence[int],
    model: str,
    keep_prefix: bool,
    strict: bool,
) -> tuple[list[list[str]], list[str], list[str]]:
    header1 = ["method"]
    header2 = [""]
    for dataset in datasets:
        for kr in keep_ratios:
            header1.append(dataset)
            header2.append(f"kr={kr}")
    for kr in keep_ratios:
        header1.append("corruption ratio (%)")
        header2.append(f"kr={kr}")

    rows: list[list[str]] = []
    expected_ratio_count = len(datasets) * len(seeds)
    for method in methods:
        row = [display_method_name(method, keep_prefix)]
        for dataset in datasets:
            for kr in keep_ratios:
                stats = aggregate_accuracy(
                    result_roots, method, dataset, model, seeds, kr, strict
                )
                row.append(format_cell(stats, len(seeds), 4))
        for kr in keep_ratios:
            stats = aggregate_corruption_ratio(
                mask_roots,
                corruption_root,
                method,
                datasets,
                seeds,
                kr,
                strict,
            )
            row.append(format_cell(stats, expected_ratio_count, 2))
        rows.append(row)
    return rows, header1, header2


def print_table(header1: list[str], header2: list[str], rows: list[list[str]]) -> None:
    table = [header1, header2, *rows]
    widths = [0] * len(header1)
    for row in table:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_row(row: list[str]) -> str:
        parts: list[str] = []
        for index, cell in enumerate(row):
            text = str(cell)
            parts.append(text.ljust(widths[index]) if index == 0 else text.rjust(widths[index]))
        return "  ".join(parts)

    print(format_row(header1))
    print(format_row(header2))
    print("-" * len(format_row(header1)))
    for row in rows:
        print(format_row(row))


def main() -> None:
    args = parse_args()
    result_roots = [Path(item) for item in parse_csv_str(args.result_roots)]
    mask_roots = [Path(item) for item in parse_csv_str(args.mask_roots)]
    corruption_root = Path(args.corruption_root)
    datasets = parse_csv_str(args.datasets)
    seeds = parse_csv_int(args.seeds)
    keep_ratios = parse_csv_int(args.kr)

    methods = discover_methods(
        result_roots,
        mask_roots,
        parse_csv_str(args.methods),
    )
    if not methods:
        print("[INFO] No corruption_* methods found.")
        print(f"       result_roots={[str(path) for path in result_roots]}")
        print(f"       mask_roots={[str(path) for path in mask_roots]}")
        return

    print("[INFO] show fixed image-corruption experiment table")
    print(f"       result_roots={[str(path) for path in result_roots]}")
    print(f"       mask_roots={[str(path) for path in mask_roots]}")
    print(f"       corruption_root={corruption_root}")
    print(f"       datasets={datasets}")
    print(f"       seeds={seeds}")
    print(f"       keep_ratios={keep_ratios}")
    print(f"       model={args.model}")
    print(f"       methods={methods}")
    print()

    rows, header1, header2 = build_rows(
        result_roots,
        mask_roots,
        corruption_root,
        methods,
        datasets,
        seeds,
        keep_ratios,
        args.model,
        args.keep_prefix,
        args.strict,
    )
    print_table(header1, header2, rows)


if __name__ == "__main__":
    main()
