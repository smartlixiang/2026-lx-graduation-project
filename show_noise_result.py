#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show numerical tables for label-noise experiments.

Run from the project root of 2026-lx-graduation-project:

    python show_noise_result.py

This script does not draw figures. It prints one terminal table containing:
    - CIFAR-10 accuracy at kr=30/50
    - CIFAR-100 accuracy at kr=30/50
    - selected noisy-label sample ratio at kr=30/50

Accuracy is aggregated across seeds as mean±std, following draw_acc_curve.py:
    - use mean of the last 10 entries in "accuracy_samples" if available;
    - otherwise use the scalar field "accuracy".

Accuracy is displayed as a percentage with two decimal places. Noise ratios are
already computed in percentage units and are displayed with two decimal places.

Noise ratio is computed from:
    - noise_mask/[noise_method]/[dataset]/[seed]/mask_[kr].npz
    - noise/[dataset]/noise_list_[seed].txt

For the noise-ratio block, each keep ratio aggregates 6 values:
    2 datasets × 3 seeds.

The random baseline is re-sampled once per requested dataset/seed/keep ratio
with the same class-stratified selection function used by training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Sequence

import numpy as np


DEFAULT_DATASETS = ["cifar10", "cifar100"]
DEFAULT_SEEDS = [22, 42, 96]
DEFAULT_KR = [30, 50]
DEFAULT_MODEL = "resnet50"

PREFERRED_METHOD_ORDER = [
    "noise_random",
    "noise_EL2N",
    "noise_Forgetting",
    "noise_GraNd",
    "noise_herding",
    "noise_Herding",
    "noise_MDS",
    "noise_MoSo",
    "noise_yangclip",
    "noise_YangCLIP",
    "noise_learned_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print label-noise experiment result table without plotting."
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="noise_result",
        help="Comma-separated result roots. Default: noise_result",
    )
    parser.add_argument(
        "--mask-roots",
        type=str,
        default="noise_mask",
        help=(
            "Comma-separated mask roots. The first existing matching mask is used. "
            "Default: noise_mask"
        ),
    )
    parser.add_argument(
        "--noise-root",
        type=str,
        default="noise",
        help="Root directory for noise_list files. Default: noise",
    )
    parser.add_argument("--data-root", default="data", help="Dataset root. Default: data")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets. Default: cifar10,cifar100",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 22,42,96",
    )
    parser.add_argument(
        "--kr",
        type=str,
        default=",".join(str(k) for k in DEFAULT_KR),
        help="Comma-separated keep ratios. Default: 30,50",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name used in result path. Default: resnet50",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help=(
            "Optional comma-separated method directory names. "
            "If omitted, all result/mask directories starting with noise_ are detected."
        ),
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Display method names with the noise_ prefix.",
    )
    return parser.parse_args()


def parse_csv_str(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_int(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def normalize_noise_method(method: str) -> str:
    return strip_repeated_prefix(method, "noise_")


def strip_repeated_prefix(method: str, prefix: str) -> str:
    while method.startswith(prefix):
        method = method[len(prefix):]
    return method


def noise_method_candidates(base_method: str) -> list[str]:
    return [f"noise_{base_method}", base_method]


def display_method_name(method: str, keep_prefix: bool = False) -> str:
    base = normalize_noise_method(method)
    return f"noise_{base}" if keep_prefix else base


def is_random_method(method: str) -> bool:
    return normalize_noise_method(method).lower() == "random"


def sort_methods(methods: Sequence[str]) -> list[str]:
    order = {normalize_noise_method(m): i for i, m in enumerate(PREFERRED_METHOD_ORDER)}
    bases = dict.fromkeys(normalize_noise_method(m) for m in methods)
    return sorted(bases, key=lambda m: (order.get(m, 10000), m.lower()))


def discover_noise_methods(
    result_roots: Sequence[Path],
    mask_roots: Sequence[Path],
    explicit_methods: Sequence[str],
) -> list[str]:
    if explicit_methods:
        return sort_methods(explicit_methods)
    methods = set()
    for root in [*result_roots, *mask_roots]:
        if root.is_dir():
            methods.update(
                normalize_noise_method(path.name)
                for path in root.iterdir()
                if path.is_dir()
            )
    return sort_methods(methods)


def load_accuracy_from_json(path: Path) -> Optional[float]:
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[WARN] failed to read JSON: {path} ({exc})")
        return None

    acc_samples = payload.get("accuracy_samples")
    if isinstance(acc_samples, list) and acc_samples:
        values = [float(value) for value in acc_samples[-10:]]
        return mean(values)

    for key in ("accuracy", "test_acc", "test_accuracy", "last10_mean_acc"):
        if key in payload:
            try:
                return float(payload[key])
            except Exception:
                pass

    print(f"[WARN] no recognizable accuracy field in {path}")
    return None


def accuracy_result_path(
    result_root: Path,
    method: str,
    dataset: str,
    model: str,
    seed: int,
    kr: int,
) -> Path:
    return result_root / method / dataset / model / str(seed) / f"result_{int(kr)}.json"


def aggregate_accuracy(
    result_root: Sequence[Path],
    method: str,
    dataset: str,
    model: str,
    seeds: Sequence[int],
    kr: int,
) -> Optional[tuple[float, float, int]]:
    values: list[float] = []
    for seed in seeds:
        paths = [
            root / candidate / dataset / model / str(seed) / f"result_{int(kr)}.json"
            for candidate in noise_method_candidates(method)
            for root in result_root
        ]
        path = next((item for item in paths if item.is_file()), paths[0])
        value = load_accuracy_from_json(path)
        if value is not None:
            values.append(value)

    if not values:
        return None

    std_val = stdev(values) if len(values) > 1 else 0.0
    return mean(values), std_val, len(values)


def load_mask(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.is_file():
        return None

    try:
        with np.load(mask_path, allow_pickle=True) as data:
            if "mask" in data.files:
                mask = data["mask"]
            elif len(data.files) == 1:
                mask = data[data.files[0]]
            else:
                print(f"[WARN] mask file has no 'mask' key and multiple arrays: {mask_path}")
                return None
    except Exception as exc:
        print(f"[WARN] failed to read mask: {mask_path} ({exc})")
        return None

    mask = np.asarray(mask).reshape(-1)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    return mask


def find_mask_path(
    mask_roots: Sequence[Path],
    method: str,
    dataset: str,
    seed: int,
    kr: int,
) -> Optional[Path]:
    for candidate in noise_method_candidates(method):
        for root in mask_roots:
            path = root / candidate / dataset / str(seed) / f"mask_{int(kr)}.npz"
            if path.is_file():
                return path
    return None


def read_noisy_sample_ids(
    noise_root: Path,
    dataset: str,
    seed: int,
) -> Optional[np.ndarray]:
    path = noise_root / dataset / f"noise_list_{int(seed)}.txt"
    if not path.is_file():
        print(f"[WARN] noise list not found: {path}")
        return None

    try:
        arr = np.loadtxt(path, dtype=np.int64)
    except Exception as exc:
        print(f"[WARN] failed to read noise list: {path} ({exc})")
        return None

    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] < 1:
        print(f"[WARN] invalid noise list shape={arr.shape}: {path}")
        return None

    return arr[:, 0].astype(np.int64)


def compute_noise_ratio_for_mask(
    mask_roots: Sequence[Path],
    noise_root: Path,
    method: str,
    dataset: str,
    seed: int,
    kr: int,
) -> Optional[float]:
    mask_path = find_mask_path(mask_roots, method, dataset, seed, kr)
    if mask_path is None:
        return None

    mask = load_mask(mask_path)
    if mask is None:
        return None

    selected_total = int(mask.sum())
    if selected_total <= 0:
        print(f"[WARN] empty selected subset: {mask_path}")
        return None

    noisy_ids = read_noisy_sample_ids(noise_root, dataset, seed)
    if noisy_ids is None:
        return None

    if np.any(noisy_ids < 0) or np.any(noisy_ids >= len(mask)):
        print(
            f"[WARN] noise ids out of range for mask length {len(mask)}: "
            f"dataset={dataset}, seed={seed}, method={method}, kr={kr}"
        )
        return None

    is_noisy = np.zeros(len(mask), dtype=bool)
    is_noisy[noisy_ids] = True

    return float(is_noisy[mask].mean() * 100.0)


_LABEL_CACHE: dict[tuple[str, int, str], tuple[np.ndarray, int]] = {}


def compute_random_noise_ratio(
    data_root: Path,
    noise_root: Path,
    dataset: str,
    seed: int,
    kr: int,
) -> Optional[float]:
    try:
        from dataset.dataset import BaseDataLoader
        from train_after_selection import _extract_labels, read_noise_list
        from train_after_selection import select_random_indices_by_class

        key = (dataset, seed, str(data_root))
        if key not in _LABEL_CACHE:
            loader = BaseDataLoader(
                dataset,
                data_path=data_root,
                batch_size=1,
                num_workers=0,
                val_split=0.0,
                seed=seed,
            )
            train_loader, _, _ = loader.load()
            labels = _extract_labels(train_loader.dataset).astype(np.int64)
            _LABEL_CACHE[key] = (labels, loader.num_classes)
        clean_labels, num_classes = _LABEL_CACHE[key]

        mapping = read_noise_list(
            noise_root / dataset / f"noise_list_{seed}.txt"
        )
        noisy_labels = clean_labels.copy()
        noisy_labels[mapping[:, 0].astype(np.int64)] = mapping[:, 1].astype(np.int64)
        selected = select_random_indices_by_class(
            noisy_labels,
            num_classes,
            kr,
            seed,
        )
        return float(np.isin(selected, mapping[:, 0]).mean() * 100.0)
    except Exception as exc:
        print(
            f"[WARN] failed random noise ratio: dataset={dataset}, "
            f"seed={seed}, kr={kr} ({exc})"
        )
        return None


def aggregate_noise_ratio(
    mask_roots: Sequence[Path],
    noise_root: Path,
    method: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    kr: int,
    data_root: Path,
) -> Optional[tuple[float, float, int]]:
    values: list[float] = []
    for dataset in datasets:
        for seed in seeds:
            value = (
                compute_random_noise_ratio(
                    data_root,
                    noise_root,
                    dataset,
                    seed,
                    kr,
                )
                if is_random_method(method)
                else compute_noise_ratio_for_mask(
                    mask_roots=mask_roots,
                    noise_root=noise_root,
                    method=method,
                    dataset=dataset,
                    seed=seed,
                    kr=kr,
                )
            )
            if value is not None:
                values.append(value)

    if not values:
        return None

    std_val = stdev(values) if len(values) > 1 else 0.0
    return mean(values), std_val, len(values)


def format_acc_cell(
    stats: Optional[tuple[float, float, int]],
    expected_count: int,
) -> str:
    """Format raw 0-1 accuracy as percentage values without a percent sign."""
    if stats is None:
        return "-"
    mean_val, std_val, count = stats
    cell = f"{mean_val * 100.0:.2f}±{std_val * 100.0:05.2f}"
    if count < expected_count:
        cell += f"({count})"
    return cell


def format_noise_cell(
    stats: Optional[tuple[float, float, int]],
    expected_count: int,
) -> str:
    """Format values that are already expressed on a 0-100 percentage scale."""
    if stats is None:
        return "-"
    mean_val, std_val, count = stats
    cell = f"{mean_val:.2f}±{std_val:05.2f}"
    if count < expected_count:
        cell += f"({count})"
    return cell


def build_table_rows(
    result_root: Sequence[Path],
    mask_roots: Sequence[Path],
    noise_root: Path,
    methods: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    keep_ratios: Sequence[int],
    model: str,
    keep_prefix: bool,
    data_root: Path,
) -> tuple[list[list[str]], list[str], list[str]]:
    header1 = ["method"]
    header2 = [""]

    for dataset in datasets:
        for kr in keep_ratios:
            header1.append(dataset)
            header2.append(f"kr={kr}")

    for kr in keep_ratios:
        header1.append("noise ratio (%)")
        header2.append(f"kr={kr}")

    rows: list[list[str]] = []
    expected_noise_count = len(datasets) * len(seeds)

    for method in methods:
        row = [display_method_name(method, keep_prefix=keep_prefix)]

        for dataset in datasets:
            for kr in keep_ratios:
                acc_stats = aggregate_accuracy(
                    result_root=result_root,
                    method=method,
                    dataset=dataset,
                    model=model,
                    seeds=seeds,
                    kr=kr,
                )
                row.append(
                    format_acc_cell(
                        acc_stats,
                        expected_count=len(seeds),
                    )
                )

        for kr in keep_ratios:
            noise_stats = aggregate_noise_ratio(
                mask_roots=mask_roots,
                noise_root=noise_root,
                method=method,
                datasets=datasets,
                seeds=seeds,
                kr=kr,
                data_root=data_root,
            )
            row.append(
                format_noise_cell(
                    noise_stats,
                    expected_count=expected_noise_count,
                )
            )

        rows.append(row)

    return rows, header1, header2


def print_table(
    header1: list[str],
    header2: list[str],
    rows: list[list[str]],
) -> None:
    table = [header1, header2] + rows
    ncols = len(header1)

    widths = [0] * ncols
    for row in table:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def fmt_row(row: list[str]) -> str:
        parts = []
        for index, cell in enumerate(row):
            text = str(cell)
            if index == 0:
                parts.append(text.ljust(widths[index]))
            else:
                parts.append(text.rjust(widths[index]))
        return "  ".join(parts)

    print(fmt_row(header1))
    print(fmt_row(header2))
    print("-" * len(fmt_row(header1)))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    args = parse_args()

    result_root = [
        Path(item)
        for item in parse_csv_str(args.result_root)
    ]
    mask_roots = [
        Path(item)
        for item in parse_csv_str(args.mask_roots)
    ]
    noise_root = Path(args.noise_root)

    datasets = parse_csv_str(args.datasets)
    seeds = parse_csv_int(args.seeds)
    keep_ratios = parse_csv_int(args.kr)

    explicit_methods = parse_csv_str(args.methods)
    methods = discover_noise_methods(
        result_roots=result_root,
        mask_roots=mask_roots,
        explicit_methods=explicit_methods,
    )

    if not methods:
        print("[INFO] No noise_* methods found.")
        print(f"       result_root={result_root}")
        print(f"       mask_roots={[str(path) for path in mask_roots]}")
        return

    print("[INFO] show label-noise experiment table")
    print(f"       result_root={result_root}")
    print(f"       mask_roots={[str(path) for path in mask_roots]}")
    print(f"       noise_root={noise_root}")
    print(f"       datasets={datasets}")
    print(f"       seeds={seeds}")
    print(f"       keep_ratios={keep_ratios}")
    print(f"       model={args.model}")
    print(f"       methods={methods}")
    print()

    rows, header1, header2 = build_table_rows(
        result_root=result_root,
        mask_roots=mask_roots,
        noise_root=noise_root,
        methods=methods,
        datasets=datasets,
        seeds=seeds,
        keep_ratios=keep_ratios,
        model=args.model,
        keep_prefix=args.keep_prefix,
        data_root=Path(args.data_root),
    )

    print_table(header1, header2, rows)


if __name__ == "__main__":
    main()