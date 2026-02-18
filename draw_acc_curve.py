#!/usr/bin/env python3
import argparse
import json
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
        default=["random", "learned_topk", "naive_topk", "learned_group", "naive_group"],
        help="Selection methods to compare",
    )
    parser.add_argument(
        "--output",
        default="acc_curve_cifar10_resnet50_methods.png",
        help="Output image path",
    )
    return parser.parse_args()


def load_seed_results(seed_dir: Path) -> dict[int, float]:
    results: dict[int, float] = {}
    for path in seed_dir.glob("result_*.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        cut_ratio = int(metadata.get("cut_ratio", path.stem.split("_")[-1]))
        acc_samples = payload.get("accuracy_samples")
        if acc_samples:
            acc_value = mean(acc_samples[-10:])
        else:
            acc_value = float(payload.get("accuracy", 0.0))
        results[cut_ratio] = acc_value
    return results


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_dir)
    methods = args.methods
    color_map = {
        "random": "black",
        "my_learned": "red",
        "my_naive": "blue",
    }

    plt.figure(figsize=(8, 5))
    all_cut_ratios: set[int] = set()
    for method in methods:
        method_root = result_root / method / args.dataset / args.model
        if not method_root.exists():
            raise FileNotFoundError(f"Missing method directory: {method_root}")
        seed_dirs = sorted(path for path in method_root.iterdir() if path.is_dir())
        if not seed_dirs:
            raise FileNotFoundError(f"No seed directories under: {method_root}")
        seed_results = []
        for seed_dir in seed_dirs:
            seed_results.append(load_seed_results(seed_dir))

        common_cut_ratios = sorted(
            set.intersection(*(set(r.keys()) for r in seed_results))
        )
        if not common_cut_ratios:
            raise ValueError(
                f"No common cut ratios found across seeds for method: {method}"
            )

        avg_acc = []
        for cut_ratio in common_cut_ratios:
            values = [r[cut_ratio] for r in seed_results]
            avg_acc.append(mean(values))
        all_cut_ratios.update(common_cut_ratios)

        plt.plot(
            common_cut_ratios,
            avg_acc,
            marker="o",
            linewidth=2,
            color=color_map.get(method),
            label=method,
        )

    plt.xlabel("Cut Ratio (cr)")
    plt.ylabel("Accuracy (mean of last 10 epochs)")
    plt.title(f"{args.dataset.upper()} {args.model} - Mean Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    if all_cut_ratios:
        plt.xticks(sorted(all_cut_ratios))
    plt.legend()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
