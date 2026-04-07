"""Visualize dynamic component distributions before summation into u_raw."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weights.dynamic_utils import default_dynamic_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A/C/D/E histograms from dynamic cache.")
    parser.add_argument("--cache", type=str, default=None, help="Path to dynamic_components.npz")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name for default cache path")
    parser.add_argument("--proxy-model", type=str, default="resnet18", help="Proxy model for default cache path")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch tag for default cache path")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: picture/<dataset>_dynamic_components_hist.png)",
    )
    return parser.parse_args()


def _resolve_component(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    key = f"{name}_final_normalized"
    if key in data:
        values = data[key].astype(np.float32)
    else:
        legacy = f"{name}_norm2"
        if legacy not in data:
            raise KeyError(f"Neither '{key}' nor '{legacy}' found in cache.")
        values = data[legacy].astype(np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Component {name} has no finite values.")
    return values


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache) if args.cache else default_dynamic_cache_path(
        args.dataset,
        proxy_model=args.proxy_model,
        epochs=args.epochs,
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Dynamic cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    components = ["A", "C", "D", "E"]
    values = {name: _resolve_component(data, name) for name in components}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, name in zip(axes, components):
        ax.hist(values[name], bins=args.bins, color="#4C78A8", alpha=0.85, edgecolor="white")
        ax.set_title(f"{name} final_normalized")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle(f"Dynamic components before summation (cache: {cache_path.name})", fontsize=14)

    output_path = Path(args.output) if args.output else Path("picture") / f"{args.dataset}_dynamic_components_hist.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved histogram figure to: {output_path}")


if __name__ == "__main__":
    main()
