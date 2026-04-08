"""Visualize dynamic component distributions before summation into u_raw."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weights.dynamic_utils import default_dynamic_cache_path


COMPONENT_TITLES = {
    "A": "A / Absorption Gain",
    "C": "C / Confusion Complementarity",
    "D": "D / Transferability Alignment",
    "E": "E / Persistent Difficulty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A/C/D/E histograms from dynamic cache.")
    parser.add_argument("--cache", type=str, default=None, help="Path to dynamic_components.npz")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name for default cache path")
    parser.add_argument("--proxy-model", type=str, default="resnet18", help="Proxy model for default cache path")
    parser.add_argument("--epochs", type=int, default=200, help="Epoch tag for default cache path")
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
    if key not in data:
        raise KeyError(f"Missing cache field: '{key}'.")
    values = data[key].astype(np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Component {name} has no finite values.")
    return values


def _resolve_final_utility(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    if "u_norm" in data:
        values = data["u_norm"].astype(np.float32)
        title = "Final utility label used for regression"
    elif "u_raw" in data:
        values = data["u_raw"].astype(np.float32)
        title = "Final utility label before normalization"
    else:
        raise KeyError("Missing cache field: 'u_norm' or 'u_raw'.")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Final utility label has no finite values.")
    return values, title


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
        ax.hist(values[name], bins=args.bins, color="#4C78A8", alpha=0.85, edgecolor="white", label=COMPONENT_TITLES[name])
        ax.set_title(COMPONENT_TITLES[name])
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(loc="best")
        ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle(f"Dynamic components before summation (cache: {cache_path.name})", fontsize=14)

    output_path = Path(args.output) if args.output else Path("picture") / f"{args.dataset}_dynamic_components_hist.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Saved histogram figure to: {output_path}")

    final_values, final_title = _resolve_final_utility(data)
    fig_final, ax_final = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax_final.hist(final_values, bins=args.bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax_final.set_title(final_title)
    ax_final.set_xlabel("Score")
    ax_final.set_ylabel("Count")
    ax_final.grid(alpha=0.2, linestyle="--")

    final_output_path = output_path.with_name(f"{output_path.stem}_final_utility_hist{output_path.suffix}")
    fig_final.savefig(final_output_path, dpi=180)
    print(f"Saved final utility histogram figure to: {final_output_path}")


if __name__ == "__main__":
    main()
