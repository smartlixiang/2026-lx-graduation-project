from __future__ import annotations

"""Visualize ablation results for CIFAR-10 / CIFAR-100.

This script reads per-seed result JSON files produced by the project and
plots the ablation curves for two datasets in one figure.

Key features:
- Two side-by-side subplots: CIFAR-10 and CIFAR-100.
- Default kr values: [20, 30, 50, 60, 80, 90].
- Default methods:
  ["random", "naive_group", "ablation_dds", "ablation_sa", "ablation_div", "learned_group"].
- The last method uses a red five-point star marker.
- Ranking / ordering uses mean accuracy descending; when means tie, smaller
  standard deviation ranks ahead.

The script is intentionally robust to slightly different result JSON layouts.
It tries several common accuracy field names and nested structures.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    from utils.path_rules import resolve_result_path
except Exception:  # pragma: no cover - fallback for running outside repo root
    def resolve_result_path(
        mode: str,
        dataset: str,
        model: str,
        seed: int,
        keep_ratio: int,
        *,
        root: Path | str | None = None,
    ) -> Path:
        base = Path(root) if root is not None else Path("result")
        return base / mode / dataset / model / str(seed) / f"result_{int(keep_ratio)}.json"


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ["cifar10", "cifar100"]
DEFAULT_METHODS = [
    "random",
    "naive_group",
    "ablation_dds",
    "ablation_sa",
    "ablation_div",
    "learned_group",
]
DEFAULT_KR = [20, 30, 50, 60, 80, 90]
DEFAULT_SEEDS = [22, 42, 96]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show ablation curves.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to plot. Default: cifar10 cifar100.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to plot. Default matches the ablation setup.",
    )
    parser.add_argument(
        "--kr",
        nargs="+",
        type=int,
        default=DEFAULT_KR,
        help="Keep ratios to visualize.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds used to aggregate results.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="Model name used in the result directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "figures" / "show_ablation.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    return parser.parse_args()


def _flatten_numeric_values(obj: Any) -> list[float]:
    values: list[float] = []
    if obj is None:
        return values
    if isinstance(obj, (int, float, np.integer, np.floating)):
        if np.isfinite(obj):
            values.append(float(obj))
        return values
    if isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            values.extend(_flatten_numeric_values(item))
        return values
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(_flatten_numeric_values(v))
        return values
    return values


def _extract_accuracy_from_mapping(data: dict[str, Any]) -> float | None:
    # Preferred keys first.
    preferred_keys = [
        "test_acc",
        "test_accuracy",
        "accuracy",
        "acc",
        "best_acc",
        "best_accuracy",
        "final_acc",
        "final_accuracy",
        "top1_acc",
        "top1_accuracy",
    ]
    for key in preferred_keys:
        if key in data:
            vals = _flatten_numeric_values(data[key])
            if vals:
                return vals[-1]

    # Then search nested dict/list values whose key suggests accuracy.
    for key, value in data.items():
        if isinstance(key, str) and ("acc" in key.lower() or "accuracy" in key.lower()):
            vals = _flatten_numeric_values(value)
            if vals:
                return vals[-1]

    # Generic recursive search.
    for value in data.values():
        if isinstance(value, dict):
            nested = _extract_accuracy_from_mapping(value)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested = _extract_accuracy_from_mapping(item)
                    if nested is not None:
                        return nested
    return None


def read_result_accuracy(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The project may store a dict, list of dicts, or a raw scalar.
    if isinstance(data, (int, float, np.integer, np.floating)):
        value = float(data)
    elif isinstance(data, list):
        value = None
        # Try the last dict / scalar first.
        for item in reversed(data):
            if isinstance(item, (int, float, np.integer, np.floating)):
                value = float(item)
                break
            if isinstance(item, dict):
                value = _extract_accuracy_from_mapping(item)
                if value is not None:
                    break
        if value is None:
            flat = _flatten_numeric_values(data)
            if not flat:
                raise ValueError(f"No numeric accuracy found in {path}")
            value = flat[-1]
    elif isinstance(data, dict):
        value = _extract_accuracy_from_mapping(data)
        if value is None:
            flat = _flatten_numeric_values(data)
            if not flat:
                raise ValueError(f"No numeric accuracy found in {path}")
            value = flat[-1]
    else:
        raise TypeError(f"Unsupported JSON type in {path}: {type(data)!r}")

    return float(value)


def normalize_accuracy(values: np.ndarray) -> np.ndarray:
    """Convert accuracy to percent if it looks like a ratio in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    # If the median is at most 1.5, treat as a ratio.
    if np.median(finite) <= 1.5:
        return values * 100.0
    return values


def collect_statistics(
    dataset: str,
    method: str,
    krs: Iterable[int],
    seeds: Iterable[int],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[float]]]:
    means: list[float] = []
    stds: list[float] = []
    per_kr: dict[int, list[float]] = {}

    for kr in krs:
        seed_values: list[float] = []
        for seed in seeds:
            result_path = resolve_result_path(
                mode=method,
                dataset=dataset,
                model=model_name,
                seed=seed,
                keep_ratio=kr,
            )
            acc = read_result_accuracy(result_path)
            seed_values.append(acc)
        per_kr[int(kr)] = seed_values
        arr = normalize_accuracy(np.asarray(seed_values, dtype=np.float64))
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=0)))

    return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64), per_kr


def method_ranking(
    methods: list[str],
    means_by_method: dict[str, np.ndarray],
    stds_by_method: dict[str, np.ndarray],
) -> list[str]:
    # Rank by average mean descending; if tied, smaller average std first.
    items = []
    for idx, method in enumerate(methods):
        mean_val = float(np.mean(means_by_method[method]))
        std_val = float(np.mean(stds_by_method[method]))
        items.append((method, mean_val, std_val, idx))
    items.sort(key=lambda x: (-x[1], x[2], x[3]))
    return [m for m, *_ in items]


def _style_for_method(method: str, is_last: bool) -> dict[str, Any]:
    # Keep the last method as a red five-point star, consistent with draw_acc_curve.py.
    if is_last:
        return {
            "color": "red",
            "marker": "*",
            "markersize": 12,
            "linewidth": 2.2,
            "linestyle": "-",
            "zorder": 6,
        }

    palette = {
        "random": "#7f7f7f",
        "naive_group": "#1f77b4",
        "ablation_dds": "#ff7f0e",
        "ablation_sa": "#2ca02c",
        "ablation_div": "#9467bd",
    }
    markers = {
        "random": "o",
        "naive_group": "s",
        "ablation_dds": "^",
        "ablation_sa": "D",
        "ablation_div": "v",
    }
    return {
        "color": palette.get(method, "#333333"),
        "marker": markers.get(method, "o"),
        "markersize": 7,
        "linewidth": 2.0,
        "linestyle": "-",
        "zorder": 4,
    }


def add_table(
    ax: plt.Axes,
    methods_order: list[str],
    krs: list[int],
    means_by_method: dict[str, np.ndarray],
    stds_by_method: dict[str, np.ndarray],
) -> None:
    cell_text: list[list[str]] = []
    row_labels: list[str] = []
    for method in methods_order:
        row_labels.append(method)
        row = []
        for i in range(len(krs)):
            row.append(f"{means_by_method[method][i]:.2f}±{stds_by_method[method][i]:.2f}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[str(kr) for kr in krs],
        loc="bottom",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, -0.60, 1.0, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Make the table more compact and readable.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == -1:
            cell.set_text_props(weight="bold")


def plot_dataset_panel(
    ax: plt.Axes,
    dataset: str,
    methods: list[str],
    krs: list[int],
    seeds: list[int],
    model_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    means_by_method: dict[str, np.ndarray] = {}
    stds_by_method: dict[str, np.ndarray] = {}

    for method in methods:
        means, stds, _ = collect_statistics(dataset, method, krs, seeds, model_name)
        means_by_method[method] = means
        stds_by_method[method] = stds

    ranked_methods = method_ranking(methods, means_by_method, stds_by_method)

    for idx, method in enumerate(methods):
        style = _style_for_method(method, is_last=(idx == len(methods) - 1))
        ax.errorbar(
            krs,
            means_by_method[method],
            yerr=stds_by_method[method],
            label=method,
            capsize=3,
            **style,
        )

    ax.set_title(dataset.upper(), fontsize=15, pad=12)
    ax.set_xlabel("kr (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(krs)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="both", labelsize=10)

    # Keep the y-range visually stable across datasets.
    all_vals = []
    for method in methods:
        all_vals.extend(means_by_method[method].tolist())
        all_vals.extend((means_by_method[method] + stds_by_method[method]).tolist())
        all_vals.extend((means_by_method[method] - stds_by_method[method]).tolist())
    finite_vals = np.asarray([v for v in all_vals if np.isfinite(v)], dtype=np.float64)
    if finite_vals.size:
        ymin = max(0.0, float(np.min(finite_vals)) - 1.0)
        ymax = min(100.0, float(np.max(finite_vals)) + 1.0)
        if ymax <= ymin:
            ymax = ymin + 1.0
        ax.set_ylim(ymin, ymax)

    add_table(ax, ranked_methods, krs, means_by_method, stds_by_method)
    return means_by_method, stds_by_method


def main() -> None:
    args = parse_args()

    datasets = [str(x).strip().lower() for x in args.datasets]
    methods = [str(x).strip() for x in args.methods]
    krs = [int(x) for x in args.kr]
    seeds = [int(x) for x in args.seeds]

    if len(datasets) != 2:
        raise ValueError("This script is designed for exactly two datasets: cifar10 and cifar100.")
    if datasets != ["cifar10", "cifar100"]:
        # Keep the figure layout fixed as requested; only these two datasets are expected.
        raise ValueError("datasets must be exactly ['cifar10', 'cifar100'].")
    if len(methods) < 2:
        raise ValueError("At least two methods are required.")
    if not krs:
        raise ValueError("kr cannot be empty.")
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    # Fixed and reasonable overall figure size, independent of the number of subplots.
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 8.4), constrained_layout=False)
    plt.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.30, wspace=0.24)

    for ax, dataset in zip(axes, datasets, strict=True):
        plot_dataset_panel(ax, dataset, methods, krs, seeds, args.model_name)

    # A compact shared legend. The last method is the red star.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(labels)),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.965),
    )

    # Add a subtle figure title for context.
    fig.suptitle(
        "Ablation Results on CIFAR-10 / CIFAR-100",
        fontsize=16,
        y=0.985,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
