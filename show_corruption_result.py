#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show numerical tables for fixed image-corruption experiments.

Run from the project root:

    python show_corruption_result.py

The script prints one terminal table containing:
- test accuracy for the requested datasets and retention ratios;
- the percentage of corrupted samples in each selected subset.

Accuracy is aggregated across seeds as mean±std. The script first uses the mean
of the last 10 values in ``accuracy_samples`` and otherwise falls back to a
recognized scalar accuracy field. Accuracy is displayed as a percentage with
two decimal places. Corruption ratios are already computed in percentage
units and are displayed with two decimal places.

For non-random methods, the corrupted-sample ratio is recomputed from the mask
and ``corruption_data/<dataset>/corruption_list_<seed>.txt``. For ``random``,
the script repeats the same class-stratified selection used by
``train_after_selection.py`` and does not read a mask.

Expected result path candidates, in priority order:

    <result_root>/corruption_<method>/<dataset>/<model>/<seed>/result_<kr>.json
    <result_root>/<method>/<dataset>/<model>/<seed>/result_<kr>.json

Expected mask path candidates, in priority order:

    <mask_root>/corruption_<method>/<dataset>/<seed>/mask_<kr>.npz
    <mask_root>/<method>/<dataset>/<seed>/mask_<kr>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Optional, Sequence

import numpy as np


DEFAULT_DATASET_ARGUMENT = "tiny-imagenet"
DEFAULT_SEEDS = [22, 42, 96]
DEFAULT_KR = [30, 50, 70]
DEFAULT_MODEL = "resnet50"
DEFAULT_RESULT_ROOTS = ["corruption_result"]
DEFAULT_MASK_ROOTS = ["corruption_mask"]
CORRUPTION_PREFIX = "corruption_"

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


Stats = tuple[float, float, int]
_LABEL_CACHE: dict[tuple[str, int, str], tuple[np.ndarray, int]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print fixed image-corruption experiment result tables."
    )
    parser.add_argument(
        "--result-roots",
        "--result-root",
        dest="result_roots",
        default=",".join(DEFAULT_RESULT_ROOTS),
        help=(
            "Comma-separated result roots searched in order. "
            "Default: corruption_result"
        ),
    )
    parser.add_argument(
        "--mask-roots",
        "--mask-root",
        dest="mask_roots",
        default=",".join(DEFAULT_MASK_ROOTS),
        help=(
            "Comma-separated mask roots searched in order. "
            "Default: corruption_mask"
        ),
    )
    parser.add_argument(
        "--corruption-root",
        default="corruption_data",
        help="Root containing corruption_list files. Default: corruption_data",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset root used only to reproduce random selection. Default: data",
    )
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        default=DEFAULT_DATASET_ARGUMENT,
        help=(
            "Comma-separated datasets. "
            f"Default: {DEFAULT_DATASET_ARGUMENT}"
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds. Default: 22,42,96",
    )
    parser.add_argument(
        "--kr",
        default=",".join(str(kr) for kr in DEFAULT_KR),
        help="Comma-separated retention ratios. Default: 30,50,70",
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
            "Optional comma-separated method directory names. Prefixed and "
            "unprefixed forms are treated as the same logical method."
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
        help="Raise an error when an expected result, mask, dataset, or list is missing.",
    )
    return parser.parse_args()


def parse_csv_str(raw: str, *, argument_name: str = "value") -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{argument_name} cannot be empty.")
    return values


def parse_csv_int(raw: str, *, argument_name: str = "integer list") -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{argument_name} cannot be empty.")
    return values


def strip_repeated_prefix(method: str, prefix: str) -> str:
    normalized = method.strip()
    while normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    if not normalized:
        raise ValueError(f"Invalid method name: {method!r}")
    return normalized


def normalize_corruption_method(method: str) -> str:
    return strip_repeated_prefix(method, CORRUPTION_PREFIX)


def corruption_method_candidates(method: str) -> list[str]:
    base_method = normalize_corruption_method(method)
    return [f"{CORRUPTION_PREFIX}{base_method}", base_method]


def display_method_name(method: str, keep_prefix: bool = False) -> str:
    base_method = normalize_corruption_method(method)
    return f"{CORRUPTION_PREFIX}{base_method}" if keep_prefix else base_method


def is_random_method(method: str) -> bool:
    return normalize_corruption_method(method).lower() == "random"


def sort_methods(methods: Sequence[str]) -> list[str]:
    preferred_order = {
        normalize_corruption_method(method): index
        for index, method in enumerate(PREFERRED_METHOD_ORDER)
    }
    base_methods = list(
        dict.fromkeys(normalize_corruption_method(method) for method in methods)
    )
    return sorted(
        base_methods,
        key=lambda method: (preferred_order.get(method, 10_000), method.lower()),
    )


def discover_methods(
    result_roots: Sequence[Path],
    mask_roots: Sequence[Path],
    explicit_methods: Sequence[str],
) -> list[str]:
    if explicit_methods:
        return sort_methods(explicit_methods)

    discovered: set[str] = set()
    for root in [*result_roots, *mask_roots]:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir():
                try:
                    discovered.add(normalize_corruption_method(path.name))
                except ValueError:
                    continue
    return sort_methods(discovered)


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
    # Prefix priority is global: search all roots for corruption_<method> first,
    # then search all roots for the unprefixed fallback.
    return [
        root
        / candidate
        / dataset
        / model
        / str(seed)
        / f"result_{int(kr)}.json"
        for candidate in corruption_method_candidates(method)
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
        root / candidate / dataset / str(seed) / f"mask_{int(kr)}.npz"
        for candidate in corruption_method_candidates(method)
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
        except (TypeError, ValueError):
            values = []
        if values and np.all(np.isfinite(values)):
            return float(mean(values))

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
) -> Optional[Stats]:
    values: list[float] = []
    for seed in seeds:
        candidates = result_candidates(roots, method, dataset, model, seed, kr)
        path = first_existing(candidates)
        if path is None:
            message = (
                f"result missing: method={method}, dataset={dataset}, "
                f"seed={seed}, kr={kr}; tried: "
                + ", ".join(str(candidate) for candidate in candidates)
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

    if mask.size == 0:
        print(f"[WARN] empty mask array: {path}")
        return None

    unique_values = set(np.unique(mask).tolist())
    if not unique_values.issubset({0, 1, False, True}):
        print(f"[WARN] non-binary mask: {path}, values={sorted(unique_values)}")
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

    if rows.size == 0:
        print(f"[WARN] empty corruption list: {path}")
        return None
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 2:
        print(f"[WARN] corruption list must have exactly two columns: {path}")
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
            f"seed={seed}, kr={kr}; tried: "
            + ", ".join(str(candidate) for candidate in candidates)
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

    selected_count = int(mask.sum())
    if selected_count <= 0:
        message = f"empty selected subset: {mask_path}"
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}")
        return None

    expected_count = int(round(mask.size * kr / 100.0))
    if selected_count != expected_count:
        message = (
            f"mask selected count mismatch: {mask_path}, "
            f"actual={selected_count}, expected={expected_count}"
        )
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}")
        return None

    corrupted_ids = read_corrupted_ids(corruption_root, dataset, seed)
    if corrupted_ids is None:
        if strict:
            raise ValueError(
                f"Invalid corruption list: dataset={dataset}, seed={seed}"
            )
        return None
    if np.any(corrupted_ids < 0) or np.any(corrupted_ids >= mask.size):
        message = f"corruption IDs out of range for {mask_path}"
        if strict:
            raise ValueError(message)
        print(f"[WARN] {message}")
        return None

    is_corrupted = np.zeros(mask.size, dtype=bool)
    is_corrupted[corrupted_ids] = True
    return float(is_corrupted[mask].mean() * 100.0)


def load_training_labels(
    data_root: Path,
    dataset: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Load only the full training split needed for random re-sampling.

    Calling ``BaseDataLoader.load()`` would also construct the test split. That
    is unnecessary here and can fail or mutate the Tiny-ImageNet validation
    layout even though this script only needs training labels.
    """
    from dataset.dataset import BaseDataLoader
    from train_after_selection import _extract_labels

    cache_key = (dataset, int(seed), str(data_root.resolve()))
    if cache_key not in _LABEL_CACHE:
        loader = BaseDataLoader(
            dataset,
            data_path=data_root,
            batch_size=1,
            num_workers=0,
            val_split=0.0,
            download=False,
            augment=False,
            normalize=False,
            seed=seed,
        )
        train_dataset = loader.dataset._build_train_set()
        labels = _extract_labels(train_dataset).astype(np.int64, copy=False)
        _LABEL_CACHE[cache_key] = (labels, loader.num_classes)
    return _LABEL_CACHE[cache_key]


def compute_random_corruption_ratio(
    data_root: Path,
    corruption_root: Path,
    dataset: str,
    seed: int,
    kr: int,
    strict: bool,
) -> Optional[float]:
    try:
        from train_after_selection import select_random_indices_by_class

        labels, num_classes = load_training_labels(data_root, dataset, seed)
        corrupted_ids = read_corrupted_ids(corruption_root, dataset, seed)
        if corrupted_ids is None:
            raise FileNotFoundError(
                f"invalid corruption list: dataset={dataset}, seed={seed}"
            )
        if np.any(corrupted_ids < 0) or np.any(corrupted_ids >= labels.size):
            raise ValueError(
                f"corruption IDs out of range: dataset={dataset}, seed={seed}, "
                f"train_size={labels.size}"
            )

        selected_indices = select_random_indices_by_class(
            labels,
            num_classes,
            kr,
            seed,
        )
        if selected_indices.size == 0:
            raise ValueError(
                f"random selection is empty: dataset={dataset}, seed={seed}, kr={kr}"
            )

        is_corrupted = np.zeros(labels.size, dtype=bool)
        is_corrupted[corrupted_ids] = True
        return float(is_corrupted[selected_indices].mean() * 100.0)
    except Exception as exc:
        if strict:
            raise
        print(
            f"[WARN] failed random corruption ratio: dataset={dataset}, "
            f"seed={seed}, kr={kr} ({exc})"
        )
        return None


def aggregate_corruption_ratio(
    mask_roots: Sequence[Path],
    corruption_root: Path,
    method: str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    kr: int,
    strict: bool,
    data_root: Path,
) -> Optional[Stats]:
    values: list[float] = []
    for dataset_name in datasets:
        for seed in seeds:
            if is_random_method(method):
                value = compute_random_corruption_ratio(
                    data_root=data_root,
                    corruption_root=corruption_root,
                    dataset=dataset_name,
                    seed=seed,
                    kr=kr,
                    strict=strict,
                )
            else:
                value = corruption_ratio_for_mask(
                    mask_roots=mask_roots,
                    corruption_root=corruption_root,
                    method=method,
                    dataset=dataset_name,
                    seed=seed,
                    kr=kr,
                    strict=strict,
                )
            if value is not None:
                values.append(value)

    if not values:
        return None
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)


def format_accuracy_cell(
    stats: Optional[Stats],
    expected_count: int,
) -> str:
    """Format raw 0-1 accuracy as percentage values without a percent sign."""
    if stats is None:
        return "-"
    mean_value, std_value, count = stats
    cell = f"{mean_value * 100.0:.2f}±{std_value * 100.0:05.2f}"
    if count < expected_count:
        cell += f"({count})"
    return cell


def format_percentage_cell(
    stats: Optional[Stats],
    expected_count: int,
) -> str:
    """Format values that are already expressed on a 0-100 percentage scale."""
    if stats is None:
        return "-"
    mean_value, std_value, count = stats
    cell = f"{mean_value:.2f}±{std_value:05.2f}"
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
    data_root: Path,
) -> tuple[list[list[str]], list[str], list[str]]:
    header1 = ["method"]
    header2 = [""]

    for dataset_name in datasets:
        for kr in keep_ratios:
            header1.append(dataset_name)
            header2.append(f"kr={kr}")
    for kr in keep_ratios:
        header1.append("corruption ratio (%)")
        header2.append(f"kr={kr}")

    rows: list[list[str]] = []
    expected_ratio_count = len(datasets) * len(seeds)

    for method in methods:
        row = [display_method_name(method, keep_prefix)]
        for dataset_name in datasets:
            for kr in keep_ratios:
                stats = aggregate_accuracy(
                    roots=result_roots,
                    method=method,
                    dataset=dataset_name,
                    model=model,
                    seeds=seeds,
                    kr=kr,
                    strict=strict,
                )
                row.append(format_accuracy_cell(stats, len(seeds)))

        for kr in keep_ratios:
            stats = aggregate_corruption_ratio(
                mask_roots=mask_roots,
                corruption_root=corruption_root,
                method=method,
                datasets=datasets,
                seeds=seeds,
                kr=kr,
                strict=strict,
                data_root=data_root,
            )
            row.append(format_percentage_cell(stats, expected_ratio_count))
        rows.append(row)

    return rows, header1, header2


def print_table(
    header1: list[str],
    header2: list[str],
    rows: list[list[str]],
) -> None:
    table = [header1, header2, *rows]
    widths = [0] * len(header1)
    for row in table:
        if len(row) != len(header1):
            raise ValueError(
                f"table row has {len(row)} columns; expected {len(header1)}: {row}"
            )
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def format_row(row: list[str]) -> str:
        parts: list[str] = []
        for index, cell in enumerate(row):
            text = str(cell)
            parts.append(
                text.ljust(widths[index])
                if index == 0
                else text.rjust(widths[index])
            )
        return "  ".join(parts)

    print(format_row(header1))
    print(format_row(header2))
    print("-" * len(format_row(header1)))
    for row in rows:
        print(format_row(row))


def main() -> None:
    args = parse_args()
    result_roots = [
        Path(item)
        for item in parse_csv_str(args.result_roots, argument_name="result roots")
    ]
    mask_roots = [
        Path(item)
        for item in parse_csv_str(args.mask_roots, argument_name="mask roots")
    ]
    corruption_root = Path(args.corruption_root)
    data_root = Path(args.data_root)
    datasets = parse_csv_str(args.datasets, argument_name="datasets")
    seeds = parse_csv_int(args.seeds, argument_name="seeds")
    keep_ratios = parse_csv_int(args.kr, argument_name="retention ratios")

    explicit_methods = (
        parse_csv_str(args.methods, argument_name="methods")
        if args.methods.strip()
        else []
    )
    methods = discover_methods(
        result_roots=result_roots,
        mask_roots=mask_roots,
        explicit_methods=explicit_methods,
    )
    if not methods:
        print("[INFO] No corruption experiment methods found.")
        print(f"       result_roots={[str(path) for path in result_roots]}")
        print(f"       mask_roots={[str(path) for path in mask_roots]}")
        return

    print("[INFO] show fixed image-corruption experiment table")
    print(f"       result_roots={[str(path) for path in result_roots]}")
    print(f"       mask_roots={[str(path) for path in mask_roots]}")
    print(f"       corruption_root={corruption_root}")
    print(f"       data_root={data_root}")
    print(f"       datasets={datasets}")
    print(f"       seeds={seeds}")
    print(f"       keep_ratios={keep_ratios}")
    print(f"       model={args.model}")
    print(f"       methods={methods}")
    print()

    rows, header1, header2 = build_rows(
        result_roots=result_roots,
        mask_roots=mask_roots,
        corruption_root=corruption_root,
        methods=methods,
        datasets=datasets,
        seeds=seeds,
        keep_ratios=keep_ratios,
        model=args.model,
        keep_prefix=args.keep_prefix,
        strict=args.strict,
        data_root=data_root,
    )
    print_table(header1, header2, rows)


if __name__ == "__main__":
    main()