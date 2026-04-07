from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BINS = np.arange(0.0, 1.0001, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A/C/D dynamic component distributions.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=100)
    parser.add_argument("--cache", type=str, default=None, help="Optional explicit .npz cache path.")
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    return parser.parse_args()


def default_cache_path(dataset: str, proxy_model: str, proxy_epochs: int) -> Path:
    return Path("weights") / "dynamic_cache" / dataset / proxy_model / str(proxy_epochs) / "dynamic_components.npz"


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else default_cache_path(args.dataset, args.proxy_model, args.proxy_epochs)
    if not cache_path.exists():
        raise FileNotFoundError(f"Dynamic cache file not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    components = [
        ("A_norm2", "A: Absorbable Stability"),
        ("C_norm2", "C: Dynamic Class Complementarity"),
        ("D_norm2", "D: Validation Learnable Boundary"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (key, title) in zip(axes, components):
        if key not in data:
            raise KeyError(f"Missing key in cache: {key}")
        vals = data[key].astype(np.float32)
        ax.hist(vals, bins=BINS, color="#4472c4", edgecolor="white", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"{args.dataset} A/C/D distribution (seed={args.seed})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output = Path(args.output) if args.output else Path("result") / f"{args.dataset}_dynamic_components_hist.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
