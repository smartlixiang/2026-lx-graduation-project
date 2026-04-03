#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


METHODS = ["random", "naive_topk", "learned_topk", "naive_group", "learned_group"]
DATASETS = ["cifar10", "cifar100", "tiny-imagenet"]
KEEP_RATIOS = [20, 30, 40, 60, 80]
TARGET_MODEL = "resnet50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show ablation-study curves for three datasets in one combined figure."
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Root directory that stores result/<method>/<dataset>/<model>/<seed>/result_*.json",
    )
    parser.add_argument(
        "--model",
        default=TARGET_MODEL,
        help="Model name used in the saved result directories.",
    )
    parser.add_argument(
        "--output",
        default="picture/ablation_result.png",
        help="Output image path.",
    )
    return parser.parse_args()


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


def load_method_mean_results(
    result_root: Path,
    method: str,
    dataset: str,
    model: str,
    keep_ratios: list[int],
) -> dict[int, float]:
    method_root = result_root / method / dataset / model
    if not method_root.exists():
        return {}

    seed_dirs = sorted(path for path in method_root.iterdir() if path.is_dir())
    if not seed_dirs:
        return {}

    seed_results = [load_seed_results(seed_dir) for seed_dir in seed_dirs]

    avg_by_kr: dict[int, float] = {}
    for keep_ratio in keep_ratios:
        values = [result[keep_ratio] for result in seed_results if keep_ratio in result]
        if values:
            avg_by_kr[keep_ratio] = mean(values)

    return avg_by_kr


def collect_all_results(
    result_root: Path,
    datasets: list[str],
    methods: list[str],
    model: str,
    keep_ratios: list[int],
) -> dict[str, dict[str, dict[int, float]]]:
    all_results: dict[str, dict[str, dict[int, float]]] = {}
    for dataset in datasets:
        all_results[dataset] = {}
        for method in methods:
            all_results[dataset][method] = load_method_mean_results(
                result_root=result_root,
                method=method,
                dataset=dataset,
                model=model,
                keep_ratios=keep_ratios,
            )
    return all_results


def compute_avg_improvement_percent(
    dataset_results: dict[str, dict[int, float]],
    baseline_method: str,
    target_method: str = "learned_group",
    keep_ratios: list[int] = KEEP_RATIOS,
) -> float | None:
    target = dataset_results.get(target_method, {})
    baseline = dataset_results.get(baseline_method, {})

    diffs = []
    for kr in keep_ratios:
        if kr in target and kr in baseline:
            diffs.append((target[kr] - baseline[kr]) * 100.0)

    if not diffs:
        return None
    return mean(diffs)


def format_improvement(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def get_style_map() -> dict[str, dict]:
    return {
        "random": {
            "color": "#1f77b4",
            "marker": "o",
            "linewidth": 2.1,
            "markersize": 5.8,
            "zorder": 3,
        },
        "naive_topk": {
            "color": "#ff7f0e",
            "marker": "s",
            "linewidth": 2.1,
            "markersize": 5.8,
            "zorder": 3,
        },
        "learned_topk": {
            "color": "#2ca02c",
            "marker": "^",
            "linewidth": 2.15,
            "markersize": 6.2,
            "zorder": 4,
        },
        "naive_group": {
            "color": "#d62728",
            "marker": "v",
            "linewidth": 2.2,
            "markersize": 6.2,
            "zorder": 4,
        },
        "learned_group": {
            "color": "red",
            "marker": "*",
            "linewidth": 2.8,
            "markersize": 10.0,
            "zorder": 6,
        },
    }


def configure_axis(ax, dataset: str, dataset_results: dict[str, dict[int, float]]) -> None:
    ax.set_title(f"{dataset.upper()} {TARGET_MODEL}", fontsize=14, pad=8)
    ax.set_xlabel("Keep Ratio (kr)", fontsize=11)
    ax.set_ylabel("Accuracy (mean of last 10 epochs)", fontsize=11)

    x_pos = list(range(len(KEEP_RATIOS)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(kr) for kr in KEEP_RATIOS])

    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.8)

    all_y = []
    for method in METHODS:
        for kr in KEEP_RATIOS:
            val = dataset_results.get(method, {}).get(kr)
            if val is not None:
                all_y.append(val)

    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        span = ymax - ymin

        if span == 0:
            pad = 0.0008
        else:
            pad = max(0.0006, 0.045 * span)

        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlim(-0.15, len(KEEP_RATIOS) - 1 + 0.15)


def draw_dataset_panel(
    ax,
    dataset: str,
    dataset_results: dict[str, dict[int, float]],
    style_map: dict[str, dict],
) -> None:
    x_pos_map = {kr: idx for idx, kr in enumerate(KEEP_RATIOS)}

    for method in METHODS:
        avg_by_kr = dataset_results.get(method, {})
        valid_krs = [kr for kr in KEEP_RATIOS if kr in avg_by_kr]
        x_values = [x_pos_map[kr] for kr in valid_krs]
        y_values = [avg_by_kr[kr] for kr in valid_krs]
        if not x_values:
            continue

        style = style_map[method]
        ax.plot(
            x_values,
            y_values,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markeredgewidth=0.8,
            alpha=0.98,
            zorder=style["zorder"],
        )

    configure_axis(ax, dataset, dataset_results)

    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        fontsize=9.0,
        ncol=1,
        handlelength=1.9,
        borderpad=0.42,
        labelspacing=0.35,
    )


def draw_summary_table(
    ax,
    all_results: dict[str, dict[str, dict[int, float]]],
) -> None:
    ax.axis("off")

    title_text = (
        "Average accuracy gains of learned_group over other ablation variants\n"
        "computed on the shown keep ratios (20, 30, 40, 60, 80)"
    )
    ax.text(
        0.5,
        0.92,
        title_text,
        ha="center",
        va="center",
        fontsize=12.5,
        transform=ax.transAxes,
    )

    baseline_methods = ["random", "naive_topk", "learned_topk", "naive_group"]

    row_labels = []
    cell_text = []

    for dataset in DATASETS:
        row_labels.append(dataset)
        row = []
        per_dataset_values = []
        for baseline in baseline_methods:
            value = compute_avg_improvement_percent(
                dataset_results=all_results[dataset],
                baseline_method=baseline,
                target_method="learned_group",
                keep_ratios=KEEP_RATIOS,
            )
            row.append(format_improvement(value))
            if value is not None:
                per_dataset_values.append(value)

        row.append(format_improvement(mean(per_dataset_values) if per_dataset_values else None))
        cell_text.append(row)

    # 缩短列表头，避免重叠
    col_labels = ["random", "naive_topk", "learned_topk", "naive_group", "mean"]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        bbox=[0.08, 0.18, 0.86, 0.52],
        cellLoc="center",
        rowLoc="center",
        colWidths=[0.18, 0.20, 0.20, 0.20, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
        if col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f7f7f7")


def print_terminal_summary(all_results: dict[str, dict[str, dict[int, float]]]) -> None:
    print("\nAblation mean accuracy by dataset and keep ratio:")
    for dataset in DATASETS:
        print(f"\n[{dataset}]")
        header = ["method"] + [str(kr) for kr in KEEP_RATIOS]
        print("  ".join(h.rjust(14) for h in header))
        for method in METHODS:
            row = [method]
            for kr in KEEP_RATIOS:
                value = all_results[dataset].get(method, {}).get(kr)
                row.append("-" if value is None else f"{value:.4f}")
            print("  ".join(str(x).rjust(14) for x in row))

    print("\nAverage improvement of learned_group:")
    baseline_methods = ["random", "naive_topk", "learned_topk", "naive_group"]
    for dataset in DATASETS:
        pieces = []
        values = []
        for baseline in baseline_methods:
            improvement = compute_avg_improvement_percent(all_results[dataset], baseline)
            pieces.append(f"vs {baseline}: {format_improvement(improvement)}")
            if improvement is not None:
                values.append(improvement)
        overall = mean(values) if values else None
        pieces.append(f"avg vs all: {format_improvement(overall)}")
        print(f"  {dataset}: " + " | ".join(pieces))


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_dir)
    output_path = Path(args.output)

    all_results = collect_all_results(
        result_root=result_root,
        datasets=DATASETS,
        methods=METHODS,
        model=args.model,
        keep_ratios=KEEP_RATIOS,
    )

    print_terminal_summary(all_results)

    style_map = get_style_map()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_cifar10 = axes[0, 0]
    ax_cifar100 = axes[0, 1]
    ax_tiny = axes[1, 0]
    ax_table = axes[1, 1]

    draw_dataset_panel(
        ax=ax_cifar10,
        dataset="cifar10",
        dataset_results=all_results["cifar10"],
        style_map=style_map,
    )
    draw_dataset_panel(
        ax=ax_cifar100,
        dataset="cifar100",
        dataset_results=all_results["cifar100"],
        style_map=style_map,
    )
    draw_dataset_panel(
        ax=ax_tiny,
        dataset="tiny-imagenet",
        dataset_results=all_results["tiny-imagenet"],
        style_map=style_map,
    )
    draw_summary_table(ax=ax_table, all_results=all_results)

    fig.suptitle(
        "Ablation Study of Static Scoring and Group Selection",
        fontsize=19,
        y=0.975,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    print(f"\nSaved figure to {output_path}")


if __name__ == "__main__":
    main()
