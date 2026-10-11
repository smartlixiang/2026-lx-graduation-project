#!/usr/bin/env python3
"""Print human-weight ablation results without generating plots.

Uses the same accuracy convention as draw_acc_curve.py: average the last ten
accuracy_samples for each seed, then report the mean and sample standard
deviation across seeds. Stored accuracy is used when samples are unavailable.
The final column averages (method accuracy - Ours accuracy) across the requested
subset retention ratios, in percentage points, following Table 6's sign convention.

Example:
    python show_weights_ablation.py
    python show_weights_ablation.py --model resnet50 --seed 22,42,96
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parent
METHODS = ("Ours", "naive_group", "high_sa", "high_div", "high_sv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print accuracy tables for the human-weight ablation."
    )
    parser.add_argument(
        "--result-dir", type=Path, default=PROJECT_ROOT / "result",
        help="Root containing <method>/<dataset>/<model>/<seed>/result_*.json.",
    )
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--kr", default="30,50,70", help="Comma-separated keep ratios.")
    parser.add_argument(
        "--seed", default=None,
        help="Comma-separated seeds; by default aggregate all saved numeric seed directories.",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser


def parse_kr_list(raw: str) -> list[int]:
    ratios = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not ratios or any(not 0 < ratio <= 100 for ratio in ratios):
        raise ValueError("--kr must contain percentages in [1, 100].")
    return ratios


def load_seed_results(seed_dir: Path, keep_ratios: list[int]) -> dict[int, float]:
    results = {}
    for keep_ratio in keep_ratios:
        path = seed_dir / f"result_{keep_ratio}.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        stored_ratio = int(metadata.get("keep_ratio", metadata.get("cut_ratio", keep_ratio)))
        if stored_ratio != keep_ratio:
            raise ValueError(f"{path}: metadata keep ratio is {stored_ratio}, expected {keep_ratio}.")
        samples = payload.get("accuracy_samples")
        if samples:
            accuracy = mean(float(value) for value in samples[-10:])
        elif "accuracy" in payload:
            accuracy = float(payload["accuracy"])
        else:
            raise ValueError(f"{path}: missing accuracy_samples and accuracy.")
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"{path}: expected finite accuracy in [0, 1], got {accuracy}.")
        results[keep_ratio] = accuracy
    return results


def collect_method_stats(
    method_root: Path, keep_ratios: list[int], seeds: list[int] | None,
) -> tuple[dict[int, tuple[float, float]], dict[int, list[int]], list[int]]:
    if seeds is None:
        seeds = sorted(
            int(path.name) for path in method_root.glob("*")
            if path.is_dir() and path.name.isdecimal()
        )
    results = {
        seed: load_seed_results(method_root / str(seed), keep_ratios)
        for seed in seeds
    }
    stats = {}
    used_seeds = {}
    for ratio in keep_ratios:
        used_seeds[ratio] = [seed for seed in seeds if ratio in results[seed]]
        values = [results[seed][ratio] for seed in used_seeds[ratio]]
        if values:
            stats[ratio] = (mean(values), stdev(values) if len(values) > 1 else 0.0)
    return stats, used_seeds, seeds


def print_table(
    methods: list[str], keep_ratios: list[int],
    method_stats: dict[str, dict[int, tuple[float, float]]],
) -> None:
    rank_samples: dict[str, list[float]] = {method: [] for method in methods}
    best_by_ratio = {}
    for ratio in keep_ratios:
        present = [
            (method, method_stats[method][ratio][0])
            for method in methods if ratio in method_stats[method]
        ]
        if not present or ratio == 100:
            continue
        present.sort(key=lambda item: item[1], reverse=True)
        best_by_ratio[ratio] = present[0][1]
        # Use average ranks for ties, avoiding dependence on method order.
        start = 0
        while start < len(present):
            end = start + 1
            while end < len(present) and present[end][1] == present[start][1]:
                end += 1
            rank = (start + 1 + end) / 2.0
            for method, _ in present[start:end]:
                rank_samples[method].append(rank)
            start = end
    average_ranks = {
        method: mean(values) for method, values in rank_samples.items() if values
    }
    best_rank = min(average_ranks.values()) if average_ranks else None
    header = [
        "method", *(str(ratio) for ratio in keep_ratios), "avg_rank",
        "avg_Δacc_vs_Ours(pp)",
    ]
    comparison_ratios = [ratio for ratio in keep_ratios if ratio != 100]
    ours_stats = method_stats.get("Ours", {})
    rows = []
    for method in methods:
        row = [method]
        for ratio in keep_ratios:
            stats = method_stats[method].get(ratio)
            if stats is None:
                row.append("-")
                continue
            mean_value, std_value = stats
            cell = f"{mean_value * 100:.2f}±{std_value * 100:05.2f}"
            if ratio in best_by_ratio and math.isclose(
                mean_value, best_by_ratio[ratio], rel_tol=1e-12, abs_tol=1e-12,
            ):
                cell = f"**{cell}**"
            row.append(cell)
        rank = average_ranks.get(method)
        cell = "-" if rank is None else f"{rank:.4f}"
        if rank is not None and math.isclose(rank, best_rank, rel_tol=1e-12, abs_tol=1e-12):
            cell = f"**{cell}**"
        row.append(cell)
        # Require the full requested comparison range so every delta uses the
        # same retention ratios; incomplete results are shown as missing.
        if comparison_ratios and all(
            ratio in method_stats[method] and ratio in ours_stats
            for ratio in comparison_ratios
        ):
            difference = mean(
                (method_stats[method][ratio][0] - ours_stats[ratio][0]) * 100.0
                for ratio in comparison_ratios
            )
            row.append("0.00" if method == "Ours" else f"{difference:+.2f}")
        else:
            row.append("-")
        rows.append(row)
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print(row[0].ljust(widths[0]) + "  " + "  ".join(
            cell.rjust(widths[i]) for i, cell in enumerate(row[1:], start=1)
        ))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        keep_ratios = parse_kr_list(args.kr)
        seeds = None
        if args.seed is not None:
            seeds = sorted({int(item.strip()) for item in args.seed.split(",") if item.strip()})
            if not seeds:
                raise ValueError("--seed must not be empty.")
    except ValueError as error:
        parser.error(str(error))
    methods = list(dict.fromkeys(args.methods))
    method_stats = {}
    notes = []
    # Ours remains the delta reference even when --methods hides its row.
    for method in dict.fromkeys([*methods, "Ours"]):
        method_root = args.result_dir / method / args.dataset / args.model
        if method == "Ours" and not method_root.is_dir():
            fallback = args.result_dir / "learned_group" / args.dataset / args.model
            if fallback.is_dir():
                method_root = fallback
                notes.append(f"[INFO] Ours results loaded from {fallback}")
        stats, used, expected = collect_method_stats(method_root, keep_ratios, seeds)
        method_stats[method] = stats
        if not expected:
            notes.append(f"[WARN] No seed results found: {method_root}")
        for ratio in keep_ratios:
            missing = sorted(set(expected) - set(used[ratio]))
            if missing:
                notes.append(f"[WARN] {method}, kr={ratio}: missing seeds {missing}.")

    print(f"Human-weight ablation | dataset={args.dataset} | model={args.model}")
    print("Accuracy (%): last-10 mean per seed, then mean ± sample std across seeds.\n")
    print_table(methods, keep_ratios, method_stats)
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
