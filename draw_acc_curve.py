#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw accuracy curve for random selection on CIFAR-10 with ResNet-50"
        )
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Root directory that stores result/<dataset>/<model>/<seed>/*.json",
    )
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument("--method", default="random", help="Selection method")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[22, 42, 96],
        help="Seeds to average",
    )
    parser.add_argument(
        "--output",
        default="acc_curve_cifar10_resnet50_random.png",
        help="Output image path",
    )
    return parser.parse_args()


def load_seed_results(seed_dir: Path, method: str) -> dict[int, float]:
    results: dict[int, float] = {}
    for path in seed_dir.glob(f"result_*_{method}.json"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        cut_ratio = int(metadata.get("cut_ratio", path.stem.split("_")[1]))
        acc_samples = payload.get("accuracy_samples")
        if acc_samples:
            acc_value = mean(acc_samples[-10:])
        else:
            acc_value = float(payload.get("accuracy", 0.0))
        results[cut_ratio] = acc_value
    return results


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_dir) / args.dataset / args.model
    seed_results = []
    for seed in args.seeds:
        seed_dir = result_root / str(seed)
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
        seed_results.append(load_seed_results(seed_dir, args.method))

    common_cut_ratios = sorted(set.intersection(*(set(r.keys()) for r in seed_results)))
    if not common_cut_ratios:
        raise ValueError("No common cut ratios found across seeds.")

    avg_acc = []
    for cut_ratio in common_cut_ratios:
        values = [r[cut_ratio] for r in seed_results]
        avg_acc.append(mean(values))

    plt.figure(figsize=(8, 5))
    plt.plot(common_cut_ratios, avg_acc, marker="o", linewidth=2)
    plt.xlabel("Cut Ratio (cr)")
    plt.ylabel("Accuracy (mean of last 10 epochs)")
    plt.title(
        f"{args.dataset.upper()} {args.model} ({args.method}) - Mean Accuracy"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(common_cut_ratios)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
