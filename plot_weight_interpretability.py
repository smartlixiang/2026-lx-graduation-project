#!/usr/bin/env python3
"""
Plot interpretable naive-vs-learned scoring weight comparison.

Designed for the repository:
smartlixiang/2026-lx-graduation-project

Expected input format (same as weights/scoring_weights.json):
{
  "cifar10": {
    "naive": {"sa": 0.3333, "div": 0.3333, "dds": 0.3333},
    "22": {"sa": ..., "div": ..., "dds": ...},
    "42": {"sa": ..., "div": ..., "dds": ...},
    "96": {"sa": ..., "div": ..., "dds": ...}
  },
  ...
}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

COMPONENTS = ["sa", "div", "dds"]
LABEL_MAP = {"sa": "SA", "div": "Div", "dds": "DDS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw naive vs learned scoring-weight comparison."
    )
    parser.add_argument(
        "--weight-json",
        type=str,
        default="weights/scoring_weights.json",
        help="Path to weights/scoring_weights.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets to draw; default = all datasets in JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="picture/naive_vs_learned_weights.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="Annotate learned-naive deltas beside the learned bars",
    )
    return parser.parse_args()


def load_weight_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_triplet(d: dict) -> np.ndarray:
    arr = np.array([float(d[k]) for k in COMPONENTS], dtype=np.float64)
    s = arr.sum()
    if s <= 0:
        raise ValueError(f"Invalid non-positive weight sum: {d}")
    return arr / s


def mean_learned_weights(dataset_entry: dict) -> np.ndarray:
    learned_keys = [k for k in dataset_entry.keys() if k != "naive" and not k.endswith("_meta")]
    if not learned_keys:
        raise ValueError("No learned seeds found in dataset entry.")
    arrs = []
    for key in learned_keys:
        item = dataset_entry[key]
        arrs.append(normalize_triplet(item))
    return np.mean(np.stack(arrs, axis=0), axis=0)


def main() -> None:
    args = parse_args()
    weight_path = Path(args.weight_json)
    data = load_weight_json(weight_path)

    datasets = args.datasets if args.datasets else list(data.keys())
    datasets = [d for d in datasets if d in data]
    if not datasets:
        raise ValueError("No valid datasets selected.")

    naive = []
    learned = []
    for dataset in datasets:
        entry = data[dataset]
        naive.append(normalize_triplet(entry["naive"]))
        learned.append(mean_learned_weights(entry))
    naive = np.stack(naive, axis=0)
    learned = np.stack(learned, axis=0)

    fig, ax = plt.subplots(figsize=(2.8 * len(datasets) + 2.0, 5.2))
    x = np.arange(len(datasets), dtype=float)
    width = 0.32
    x_naive = x - width / 1.6
    x_learned = x + width / 1.6

    bottoms_naive = np.zeros(len(datasets), dtype=np.float64)
    bottoms_learned = np.zeros(len(datasets), dtype=np.float64)

    for idx, comp in enumerate(COMPONENTS):
        vals_naive = naive[:, idx]
        vals_learned = learned[:, idx]

        bars_n = ax.bar(
            x_naive, vals_naive, width=width, bottom=bottoms_naive, label=LABEL_MAP[comp] if idx == 0 else None
        )
        bars_l = ax.bar(
            x_learned, vals_learned, width=width, bottom=bottoms_learned
        )

        for bars, vals, bottoms in [(bars_n, vals_naive, bottoms_naive), (bars_l, vals_learned, bottoms_learned)]:
            for rect, v, b in zip(bars, vals, bottoms):
                if v >= 0.08:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        b + v / 2.0,
                        f"{LABEL_MAP[comp]}\n{v:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        bottoms_naive += vals_naive
        bottoms_learned += vals_learned

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Weight proportion")
    ax.set_title("Naive vs Learned scoring weights")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=10)

    for xi_n, xi_l in zip(x_naive, x_learned):
        ax.text(xi_n, 1.02, "Naive", ha="center", va="bottom", fontsize=10)
        ax.text(xi_l, 1.02, "Learned", ha="center", va="bottom", fontsize=10)

    if args.show_delta:
        for i, dataset in enumerate(datasets):
            delta = learned[i] - naive[i]
            delta_text = "\n".join(
                f"{LABEL_MAP[c]} {'+' if d >= 0 else ''}{d:.3f}"
                for c, d in zip(COMPONENTS, delta)
            )
            ax.text(
                x_learned[i] + 0.28, 0.52, delta_text,
                ha="left", va="center", fontsize=8
            )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
