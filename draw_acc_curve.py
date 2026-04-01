#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw accuracy curves averaged across seeds for multiple selection methods"
        )
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Root directory that stores result/<method>/<dataset>/<model>/<seed>/result_*.json",
    )
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "herding", "E2LN", "GraNd", "Forgetting", "MoSo",
                 "yangclip", "naive_topk", "learned_topk", "naive_group", "learned_group"],
        help="Selection methods to compare",
    )
    parser.add_argument(
        "--kr",
        default="20,30,40,50,60,70,80,90",
        help="Keep ratio list, e.g. '20,30,40,50,60,70,80,90'",
    )
    parser.add_argument("--output", default=None, help="Output image path")
    return parser.parse_args()


def parse_kr_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--kr cannot be empty")
    return sorted({int(v) for v in values})


def load_seed_results(seed_dir: Path) -> dict[int, float]:
    results: dict[int, float] = {}
    for path in seed_dir.glob("result_*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        keep_ratio = int(
            metadata.get("keep_ratio", metadata.get("cut_ratio", path.stem.split("_")[-1]))
        )
        acc_samples = payload.get("accuracy_samples")
        if acc_samples:
            acc_value = mean(acc_samples[-10:])
        else:
            acc_value = float(payload.get("accuracy", 0.0))
        results[keep_ratio] = acc_value
    return results


def main() -> None:
    args = parse_args()
    keep_ratios = parse_kr_list(args.kr)
    result_root = Path(args.result_dir)
    methods = args.methods
    color_map = {
        "random": "black",
        "my_learned": "red",
        "my_naive": "blue",
    }

    output_name = args.output or f"{args.dataset}_{args.model}.png"
    output_path = Path("picture") / Path(output_name).name

    plt.figure(figsize=(8, 5))
    missing_methods: list[str] = []
    valid_methods: list[str] = []
    method_to_mean: dict[str, dict[int, float]] = {}

    for method in methods:
        method_root = result_root / method / args.dataset / args.model
        if not method_root.exists():
            missing_methods.append(method)
            continue
        seed_dirs = sorted(path for path in method_root.iterdir() if path.is_dir())
        if not seed_dirs:
            missing_methods.append(method)
            continue
        seed_results = []
        for seed_dir in seed_dirs:
            seed_results.append(load_seed_results(seed_dir))

        avg_by_kr: dict[int, float] = {}
        for keep_ratio in keep_ratios:
            values = [result[keep_ratio] for result in seed_results if keep_ratio in result]
            if values:
                avg_by_kr[keep_ratio] = mean(values)
        method_to_mean[method] = avg_by_kr
        valid_methods.append(method)

        x_values = [kr for kr in keep_ratios if kr in avg_by_kr]
        y_values = [avg_by_kr[kr] for kr in x_values]
        if not x_values:
            print(f"[WARN] method={method} has no results for requested keep ratios: {keep_ratios}")
            continue

        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            color=color_map.get(method),
            label=method,
        )

    ranking_sum = {method: 0.0 for method in valid_methods}
    ranking_count = {method: 0 for method in valid_methods}
    for kr in keep_ratios:
        present = [(method, method_to_mean.get(method, {}).get(kr)) for method in valid_methods]
        present = [(m, v) for m, v in present if v is not None]
        present.sort(key=lambda item: item[1], reverse=True)
        for rank, (method, _) in enumerate(present, start=1):
            ranking_sum[method] += rank
            ranking_count[method] += 1

    avg_rank_map: dict[str, float] = {}
    for method in valid_methods:
        if ranking_count[method] > 0:
            avg_rank_map[method] = ranking_sum[method] / ranking_count[method]

    print("\nMean accuracy by keep ratio (4 decimal places):")
    header = ["method"] + [str(kr) for kr in keep_ratios] + ["avg_rank"]

    best_by_kr: dict[int, float] = {}
    for kr in keep_ratios:
        vals = [method_to_mean.get(method, {}).get(kr) for method in valid_methods]
        vals = [v for v in vals if v is not None]
        if vals:
            best_by_kr[kr] = max(vals)
    best_avg_rank = min(avg_rank_map.values()) if avg_rank_map else None

    table_rows = []
    for method in valid_methods:
        kr_to_value = method_to_mean.get(method, {})
        row = [method]
        for kr in keep_ratios:
            val = kr_to_value.get(kr)
            if val is None:
                row.append("-")
            else:
                cell = f"{val:.4f}"
                if kr in best_by_kr and math.isclose(val, best_by_kr[kr], rel_tol=1e-12, abs_tol=1e-12):
                    cell = f"**{cell}**"
                row.append(cell)
        if method in avg_rank_map:
            avg_rank_cell = f"{avg_rank_map[method]:.4f}"
            if best_avg_rank is not None and math.isclose(avg_rank_map[method], best_avg_rank, rel_tol=1e-12, abs_tol=1e-12):
                avg_rank_cell = f"**{avg_rank_cell}**"
            row.append(avg_rank_cell)
        else:
            row.append("-")
        table_rows.append(row)

    cols = [header] + table_rows
    col_count = len(header)
    widths = [0] * col_count
    for r in cols:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = (
        f"{header[0].ljust(widths[0])}  "
        + "  ".join(header[i].rjust(widths[i]) for i in range(1, col_count))
    )
    print(header_line)

    for row in table_rows:
        line = f"{row[0].ljust(widths[0])}  " + "  ".join(row[i].rjust(widths[i]) for i in range(1, col_count))
        print(line)

    plt.xlabel("Keep Ratio (kr)")
    plt.ylabel("Accuracy (mean of last 10 epochs)")
    plt.title(f"{args.dataset.upper()} {args.model} - Mean Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(keep_ratios)
    if valid_methods:
        plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved figure to {output_path}")
    if missing_methods:
        print(f"未保存结果的方法（已忽略）: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()
