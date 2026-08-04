#!/usr/bin/env python3
"""Compare corruption-type composition in selected masks.

This script is designed for the corruption experiment in
smartlixiang/2026-lx-graduation-project. It reads:

  corruption_data/<dataset>/corruption_list_<seed>.txt
  corruption_mask/<mode>/<dataset>/<seed>/mask_<kr>.npz

and reports, for Ours and YangCLIP by default:

1. Selected-subset share (%):
   selected samples of a corruption type / all selected samples.
   The five type shares sum to the overall corruption ratio.
2. Per-type retention rate (%):
   selected samples of a corruption type / all injected samples of that type.
3. Per-type removal rate (%): 100 - retention rate.
4. Composition among selected corrupted samples (%).

Run the script from the project root, or pass explicit --mask-root and
--corruption-root paths.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Keep these names aligned with corruption_exp/corruption_opt.py.
CORRUPTION_ID_TO_NAME: dict[int, str] = {
    0: "gaussian_noise",
    1: "partial_occlusion",
    2: "resolution_degradation",
    3: "fog",
    4: "motion_blur",
}
NUM_CORRUPTION_TYPES = len(CORRUPTION_ID_TO_NAME)

DEFAULT_METHODS: dict[str, str] = {
    "Ours": "corruption_learned_group",
    "YangCLIP": "corruption_yangclip",
}


@dataclass(frozen=True)
class Record:
    method: str
    mode: str
    dataset: str
    seed: int
    keep_ratio: int
    selected_total: int
    corrupted_selected: int
    overall_corruption_ratio: float
    selected_counts: np.ndarray
    subset_shares: np.ndarray
    retention_rates: np.ndarray
    removal_rates: np.ndarray
    corrupted_composition: np.ndarray


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("The list must contain at least one integer.")
    return values


def parse_method_specs(specs: list[str] | None) -> dict[str, str]:
    if not specs:
        return dict(DEFAULT_METHODS)

    methods: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid method specification {spec!r}. Expected LABEL=MODE, "
                "for example Ours=corruption_learned_group."
            )
        label, mode = (part.strip() for part in spec.split("=", 1))
        if not label or not mode:
            raise ValueError(f"Invalid method specification: {spec!r}")
        methods[label] = mode
    return methods


def candidate_mask_paths(
    mask_root: Path,
    mode: str,
    dataset: str,
    seed: int,
    keep_ratio: int,
) -> list[Path]:
    """Mirror the prefixed/unprefixed read compatibility used by the project."""
    mode_candidates = [mode]
    if mode.startswith("corruption_"):
        mode_candidates.append(mode.removeprefix("corruption_"))
    else:
        mode_candidates.append(f"corruption_{mode}")

    paths: list[Path] = []
    for candidate in dict.fromkeys(mode_candidates):
        paths.append(
            mask_root
            / candidate
            / dataset
            / str(seed)
            / f"mask_{keep_ratio}.npz"
        )
    return paths


def resolve_mask_path(
    mask_root: Path,
    mode: str,
    dataset: str,
    seed: int,
    keep_ratio: int,
) -> Path:
    candidates = candidate_mask_paths(mask_root, mode, dataset, seed, keep_ratio)
    for path in candidates:
        if path.exists():
            return path
    attempted = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Mask not found for method={mode}, seed={seed}, kr={keep_ratio}. "
        f"Attempted:\n  - {attempted}"
    )


def load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "mask" in data.files:
            mask = data["mask"]
        elif len(data.files) == 1:
            mask = data[data.files[0]]
        else:
            raise ValueError(
                f"Cannot identify mask array in {path}; arrays={data.files}."
            )

    mask = np.asarray(mask).squeeze()
    if mask.ndim != 1:
        raise ValueError(f"Mask must be one-dimensional: {path}, shape={mask.shape}")

    if mask.dtype == np.bool_:
        return mask.copy()

    unique = np.unique(mask)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"Mask must contain only 0/1 values: {path}, unique={unique[:20]}"
        )
    return mask.astype(bool, copy=False)


def load_corruption_types(path: Path, num_samples: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Corruption list not found: {path}")

    mapping = np.loadtxt(path, dtype=np.int64)
    if mapping.ndim == 1:
        mapping = mapping.reshape(1, 2)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError(
            f"Corruption list must contain two columns: {path}, shape={mapping.shape}"
        )

    sample_ids = mapping[:, 0]
    type_ids = mapping[:, 1]
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate sample IDs in {path}")
    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError(
            f"Out-of-range sample ID in {path}; mask length={num_samples}."
        )
    if np.any(type_ids < 0) or np.any(type_ids >= NUM_CORRUPTION_TYPES):
        raise ValueError(
            f"Invalid corruption type in {path}; valid IDs are "
            f"0..{NUM_CORRUPTION_TYPES - 1}."
        )

    corruption_types = np.full(num_samples, -1, dtype=np.int16)
    corruption_types[sample_ids] = type_ids.astype(np.int16)
    return corruption_types


def safe_percentage(numerator: np.ndarray | float, denominator: float) -> np.ndarray:
    values = np.asarray(numerator, dtype=np.float64)
    if denominator <= 0:
        return np.full(values.shape, np.nan, dtype=np.float64)
    return values / denominator * 100.0


def analyze_one(
    method: str,
    mode: str,
    dataset: str,
    seed: int,
    keep_ratio: int,
    mask_root: Path,
    corruption_root: Path,
    strict: bool,
) -> Record:
    mask_path = resolve_mask_path(mask_root, mode, dataset, seed, keep_ratio)
    mask = load_mask(mask_path)

    list_path = corruption_root / dataset / f"corruption_list_{seed}.txt"
    corruption_types = load_corruption_types(list_path, num_samples=mask.size)

    selected_total = int(mask.sum())
    if selected_total == 0:
        raise ValueError(f"Mask selects no samples: {mask_path}")

    available_counts = np.asarray(
        [np.sum(corruption_types == type_id) for type_id in range(NUM_CORRUPTION_TYPES)],
        dtype=np.int64,
    )
    selected_counts = np.asarray(
        [np.sum(mask & (corruption_types == type_id)) for type_id in range(NUM_CORRUPTION_TYPES)],
        dtype=np.int64,
    )
    corrupted_selected = int(selected_counts.sum())

    if strict:
        expected_selected = int(round(mask.size * keep_ratio / 100.0))
        # Class-wise budgeting can differ by a few samples for unusual class sizes,
        # so reject only meaningful mismatches.
        tolerance = max(1, int(round(mask.size * 0.0001)))
        if abs(selected_total - expected_selected) > tolerance:
            raise ValueError(
                f"Unexpected selected count in {mask_path}: selected={selected_total}, "
                f"expected approximately {expected_selected} for kr={keep_ratio}."
            )
        if np.any(available_counts == 0):
            raise ValueError(
                f"At least one corruption type has zero injected samples in {list_path}: "
                f"counts={available_counts.tolist()}"
            )
        if not np.all(available_counts == available_counts[0]):
            raise ValueError(
                f"Corruption types are not equally represented in {list_path}: "
                f"counts={available_counts.tolist()}"
            )

    subset_shares = safe_percentage(selected_counts, selected_total)
    retention_rates = selected_counts / available_counts * 100.0
    removal_rates = 100.0 - retention_rates
    corrupted_composition = safe_percentage(selected_counts, corrupted_selected)
    overall_corruption_ratio = corrupted_selected / selected_total * 100.0

    return Record(
        method=method,
        mode=mode,
        dataset=dataset,
        seed=seed,
        keep_ratio=keep_ratio,
        selected_total=selected_total,
        corrupted_selected=corrupted_selected,
        overall_corruption_ratio=overall_corruption_ratio,
        selected_counts=selected_counts,
        subset_shares=subset_shares,
        retention_rates=retention_rates,
        removal_rates=removal_rates,
        corrupted_composition=corrupted_composition,
    )


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)), float(np.nanstd(array, ddof=0))


def fmt(values: Iterable[float]) -> str:
    mean, std = mean_std(values)
    return f"{mean:.2f}±{std:.2f}"


def records_for(records: list[Record], method: str, keep_ratio: int) -> list[Record]:
    result = [
        record
        for record in records
        if record.method == method and record.keep_ratio == keep_ratio
    ]
    return sorted(result, key=lambda record: record.seed)


def print_summary_table(
    title: str,
    records: list[Record],
    methods: dict[str, str],
    keep_ratios: list[int],
    attribute: str,
    include_total: bool,
) -> None:
    names = [CORRUPTION_ID_TO_NAME[i] for i in range(NUM_CORRUPTION_TYPES)]
    headers = ["Method", "KR"]
    if include_total:
        headers.append("all_corruptions")
    headers.extend(names)

    rows: list[list[str]] = []
    for method in methods:
        for keep_ratio in keep_ratios:
            group = records_for(records, method, keep_ratio)
            row = [method, f"{keep_ratio}%"]
            if include_total:
                row.append(fmt(record.overall_corruption_ratio for record in group))
            for type_id in range(NUM_CORRUPTION_TYPES):
                row.append(fmt(getattr(record, attribute)[type_id] for record in group))
            rows.append(row)

    widths = [
        max(len(headers[col]), max(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print(f"\n{title}")
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def write_long_csv(path: Path, records: list[Record]) -> None:
    names = [CORRUPTION_ID_TO_NAME[i] for i in range(NUM_CORRUPTION_TYPES)]
    fields = [
        "method",
        "mode",
        "dataset",
        "seed",
        "keep_ratio",
        "selected_total",
        "corrupted_selected",
        "overall_corruption_ratio_pct",
    ]
    for name in names:
        fields.extend(
            [
                f"{name}_selected_count",
                f"{name}_subset_share_pct",
                f"{name}_retention_rate_pct",
                f"{name}_removal_rate_pct",
                f"{name}_composition_among_corrupted_pct",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "method": record.method,
                "mode": record.mode,
                "dataset": record.dataset,
                "seed": record.seed,
                "keep_ratio": record.keep_ratio,
                "selected_total": record.selected_total,
                "corrupted_selected": record.corrupted_selected,
                "overall_corruption_ratio_pct": record.overall_corruption_ratio,
            }
            for type_id, name in enumerate(names):
                row[f"{name}_selected_count"] = int(record.selected_counts[type_id])
                row[f"{name}_subset_share_pct"] = float(record.subset_shares[type_id])
                row[f"{name}_retention_rate_pct"] = float(record.retention_rates[type_id])
                row[f"{name}_removal_rate_pct"] = float(record.removal_rates[type_id])
                row[f"{name}_composition_among_corrupted_pct"] = float(
                    record.corrupted_composition[type_id]
                )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze five corruption types retained by selection masks."
    )
    parser.add_argument("--dataset", default="tiny-imagenet")
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("22,42,96"))
    parser.add_argument("--kr", type=parse_int_list, default=parse_int_list("30,50,70"))
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        metavar="LABEL=MODE",
        help=(
            "Method label and mask directory name. Repeat for multiple methods. "
            "Defaults: Ours=corruption_learned_group and "
            "YangCLIP=corruption_yangclip."
        ),
    )
    parser.add_argument("--mask-root", type=Path, default=Path("corruption_mask"))
    parser.add_argument(
        "--corruption-root", type=Path, default=Path("corruption_data")
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("corruption_type_mask_analysis.csv"),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate keep-ratio counts and equal corruption-type injection counts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        methods = parse_method_specs(args.method)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    records: list[Record] = []
    try:
        for method, mode in methods.items():
            for seed in args.seeds:
                for keep_ratio in args.kr:
                    records.append(
                        analyze_one(
                            method=method,
                            mode=mode,
                            dataset=args.dataset,
                            seed=seed,
                            keep_ratio=keep_ratio,
                            mask_root=args.mask_root,
                            corruption_root=args.corruption_root,
                            strict=args.strict,
                        )
                    )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1

    # Primary result requested for the paper analysis. The five columns sum to
    # all_corruptions for each method/KR.
    print_summary_table(
        title="Selected-subset share (%) — mean±std over seeds",
        records=records,
        methods=methods,
        keep_ratios=args.kr,
        attribute="subset_shares",
        include_total=True,
    )

    # This table most directly tests whether one method removes Gaussian noise
    # more aggressively than the other.
    print_summary_table(
        title="Per-type retention rate (%) — lower means stronger removal",
        records=records,
        methods=methods,
        keep_ratios=args.kr,
        attribute="retention_rates",
        include_total=False,
    )

    print_summary_table(
        title="Per-type removal rate (%) — higher means stronger removal",
        records=records,
        methods=methods,
        keep_ratios=args.kr,
        attribute="removal_rates",
        include_total=False,
    )

    print_summary_table(
        title=(
            "Composition among selected corrupted samples (%) — rows sum to 100%"
        ),
        records=records,
        methods=methods,
        keep_ratios=args.kr,
        attribute="corrupted_composition",
        include_total=False,
    )

    write_long_csv(args.output_csv, records)
    print(f"\nPer-seed details saved to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())