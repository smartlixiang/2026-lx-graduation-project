from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.global_config import CONFIG
from weights.dynamic_v2_utils import default_dynamic_v2_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dynamic component distributions (A/C/D).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--seed", type=int, default=CONFIG.exp_seeds[0], help="Random seed.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional direct path to dynamic cache npz.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: result/{dataset}_dynamic_components_v3_hist.png",
    )
    return parser.parse_args()


def resolve_input_path(dataset: str, seed: int, input_path: str | None) -> Path:
    del seed  # dynamic cache is seed-free.
    if input_path is not None:
        return Path(input_path)
    return default_dynamic_v2_cache_path(dataset)


def main() -> None:
    args = parse_args()
    cache_path = resolve_input_path(args.dataset, args.seed, args.input)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dynamic cache not found: {cache_path}. "
            "Please run learn_scoring_weights.py first to generate the dynamic cache npz."
        )

    data = np.load(cache_path, allow_pickle=True)
    required = ["A_norm2", "C_norm2", "D_norm2"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in cache {cache_path}: {missing}")

    components = {
        "A: Absorbable Stability": data["A_norm2"].astype(np.float32),
        "C: Dynamic Class Complementarity": data["C_norm2"].astype(np.float32),
        "D: Validation Learnable Boundary Value": data["D_norm2"].astype(np.float32),
    }

    bins = np.arange(0.0, 1.0001, 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, (title, values) in zip(axes, components.items()):
        if values.ndim != 1:
            raise ValueError(f"{title} must be 1D, got shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{title} contains NaN/inf values")
        ax.hist(values, bins=bins, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Normalized score")
        ax.set_ylabel("Count")

    fig.suptitle(f"Dynamic v3 components histogram | dataset={args.dataset}, seed={args.seed}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("result") / f"{args.dataset}_dynamic_components_v3_hist.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved histogram figure to: {output_path}")


if __name__ == "__main__":
    main()
