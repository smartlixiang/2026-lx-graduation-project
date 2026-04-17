#!/usr/bin/env python3
"""One-shot script: draw CIFAR10/CIFAR100 accuracy curves for DenseNet121 side-by-side."""

import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


RESULT_DIR = Path("result")
OUTPUT_PATH = Path("picture") / "cifar10_cifar100_densenet121_dual.png"
DATASETS = ["cifar10", "cifar100"]
MODEL = "densenet121"
METHODS = [
    "random", "herding", "E2LN", "GraNd", "Forgetting", "MoSo",
    "yangclip", "learned_group"
]
KEEP_RATIOS = [20, 30, 40, 50, 60, 70, 80, 90, 100]
DATASET_YMIN = {
    "cifar10": 0.88,
    "cifar100": 0.57,
}


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


def load_method_seed_stats(
    method_root: Path,
    keep_ratios: list[int],
) -> dict[int, tuple[float, float]]:
    if not method_root.exists():
        return {}

    seed_dirs = sorted(path for path in method_root.iterdir() if path.is_dir())
    if not seed_dirs:
        return {}

    seed_results = [load_seed_results(seed_dir) for seed_dir in seed_dirs]
    stats_by_kr: dict[int, tuple[float, float]] = {}

    for keep_ratio in keep_ratios:
        values = [result[keep_ratio] for result in seed_results if keep_ratio in result]
        if values:
            mean_val = mean(values)
            std_val = stdev(values) if len(values) > 1 else 0.0
            stats_by_kr[keep_ratio] = (mean_val, std_val)

    return stats_by_kr


def build_marker_map(methods: list[str]) -> dict[str, str]:
    marker_pool = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "H", "d", "8"]
    if len(methods) - 1 > len(marker_pool):
        raise ValueError(
            f"当前方法数为 {len(methods)}，普通 marker 数量不足，请自行扩充 marker_pool。"
        )

    marker_map: dict[str, str] = {}
    for i, method in enumerate(methods[:-1]):
        marker_map[method] = marker_pool[i]

    if methods:
        marker_map[methods[-1]] = "*"

    return marker_map


def build_style_map(methods: list[str]) -> dict[str, dict]:
    style_map: dict[str, dict] = {}

    for method in methods[:-1]:
        style_map[method] = {
            "linewidth": 1.1,
            "markersize": 3.1,
            "markeredgewidth": 0.55,
            "alpha": 0.95,
            "color": None,
            "zorder": 2,
        }

    if methods:
        style_map[methods[-1]] = {
            "linewidth": 1.6,
            "markersize": 5.3,
            "markeredgewidth": 0.75,
            "alpha": 1.0,
            "color": "red",
            "zorder": 5,
        }

    return style_map


def inject_kr100_from_random(
    method_to_stats: dict[str, dict[int, tuple[float, float]]],
    methods: list[str],
) -> None:
    if "random" not in method_to_stats or 100 not in method_to_stats["random"]:
        return

    random_kr100_stats = method_to_stats["random"][100]
    for method in methods:
        if method not in method_to_stats:
            continue
        method_to_stats[method][100] = random_kr100_stats


def summarize_dataset(dataset: str) -> tuple[list[str], dict[str, dict[int, tuple[float, float]]], dict[str, dict[int, float]], list[str]]:
    missing_methods: list[str] = []
    valid_methods: list[str] = []
    method_to_stats: dict[str, dict[int, tuple[float, float]]] = {}

    for method in METHODS:
        method_root = RESULT_DIR / method / dataset / MODEL
        stats_by_kr = load_method_seed_stats(method_root, KEEP_RATIOS)

        if not stats_by_kr:
            missing_methods.append(method)
            continue

        method_to_stats[method] = stats_by_kr
        valid_methods.append(method)

    if 100 in KEEP_RATIOS:
        inject_kr100_from_random(method_to_stats, valid_methods)

    method_to_mean: dict[str, dict[int, float]] = {
        method: {kr: stats[0] for kr, stats in kr_to_stats.items()}
        for method, kr_to_stats in method_to_stats.items()
    }

    return valid_methods, method_to_stats, method_to_mean, missing_methods


def print_table(dataset: str, valid_methods: list[str], method_to_stats: dict[str, dict[int, tuple[float, float]]], method_to_mean: dict[str, dict[int, float]]) -> None:
    ranking_keep_ratios = [kr for kr in KEEP_RATIOS if kr != 100]

    ranking_sum = {method: 0.0 for method in valid_methods}
    ranking_count = {method: 0 for method in valid_methods}
    for kr in ranking_keep_ratios:
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

    print(f"\n[{dataset}] Mean accuracy by keep ratio (4 decimal places):")
    header = ["method"] + [str(kr) for kr in KEEP_RATIOS] + ["avg_rank"]

    bold_keep_ratios = [kr for kr in KEEP_RATIOS if kr != 100]
    best_by_kr: dict[int, float] = {}
    for kr in bold_keep_ratios:
        vals = [method_to_mean.get(method, {}).get(kr) for method in valid_methods]
        vals = [v for v in vals if v is not None]
        if vals:
            best_by_kr[kr] = max(vals)

    best_avg_rank = min(avg_rank_map.values()) if avg_rank_map else None

    table_rows = []
    for method in valid_methods:
        kr_to_stats = method_to_stats.get(method, {})
        row = [method]
        for kr in KEEP_RATIOS:
            stats = kr_to_stats.get(kr)
            if stats is None:
                row.append("-")
            else:
                mean_val, std_val = stats
                cell = f"{mean_val:.4f}±{std_val:.4f}"
                if (
                    kr != 100
                    and kr in best_by_kr
                    and math.isclose(mean_val, best_by_kr[kr], rel_tol=1e-12, abs_tol=1e-12)
                ):
                    cell = f"**{cell}**"
                row.append(cell)

        if method in avg_rank_map:
            avg_rank_cell = f"{avg_rank_map[method]:.4f}"
            if best_avg_rank is not None and math.isclose(
                avg_rank_map[method], best_avg_rank, rel_tol=1e-12, abs_tol=1e-12
            ):
                avg_rank_cell = f"**{avg_rank_cell}**"
            row.append(avg_rank_cell)
        else:
            row.append("-")

        table_rows.append(row)

    cols = [header] + table_rows
    widths = [0] * len(header)
    for r in cols:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = (
        f"{header[0].ljust(widths[0])}  "
        + "  ".join(header[i].rjust(widths[i]) for i in range(1, len(header)))
    )
    print(header_line)

    for row in table_rows:
        line = (
            f"{row[0].ljust(widths[0])}  "
            + "  ".join(row[i].rjust(widths[i]) for i in range(1, len(header)))
        )
        print(line)


def main() -> None:
    marker_map = build_marker_map(METHODS)
    style_map = build_style_map(METHODS)

    # 提高整体高度，缓解左右子图并排造成的纵向压缩。
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 10.8), sharex=True)

    for ax, dataset in zip(axes, DATASETS):
        valid_methods, method_to_stats, method_to_mean, missing_methods = summarize_dataset(dataset)

        for method in valid_methods:
            mean_by_kr = method_to_mean.get(method, {})
            x_values = [kr for kr in KEEP_RATIOS if kr in mean_by_kr]
            y_values = [mean_by_kr[kr] for kr in x_values]
            if not x_values:
                continue

            plot_kwargs = {
                "marker": marker_map[method],
                "linewidth": style_map[method]["linewidth"],
                "markersize": style_map[method]["markersize"],
                "markeredgewidth": style_map[method]["markeredgewidth"],
                "alpha": style_map[method]["alpha"],
                "label": method,
                "zorder": style_map[method]["zorder"],
            }
            if style_map[method]["color"] is not None:
                plot_kwargs["color"] = style_map[method]["color"]

            ax.plot(x_values, y_values, **plot_kwargs)

        ax.set_xlabel("Keep Ratio (kr)")
        ax.set_ylabel("Accuracy (mean of last 10 epochs)")
        ax.set_title(f"{dataset.upper()} {MODEL} - Mean Accuracy")
        ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.7)
        ax.set_xticks(KEEP_RATIOS)

        all_y = [v for row in method_to_mean.values() for v in row.values()]
        dataset_ymin = DATASET_YMIN[dataset]
        if all_y:
            ymax = max(all_y)
            eps = max(0.0003, 0.012 * (ymax - dataset_ymin if ymax > dataset_ymin else 0.001))
            ax.set_ylim(dataset_ymin, ymax + eps)
        else:
            ax.set_ylim(dataset_ymin, dataset_ymin + 0.05)

        if valid_methods:
            ax.legend(
                loc="lower right",
                frameon=True,
                fancybox=True,
                framealpha=0.90,
                fontsize=10.0,
                ncol=1,
                handlelength=1.8,
                markerscale=1.0,
                borderpad=0.5,
                labelspacing=0.45,
            )

        if 100 in KEEP_RATIOS and ("random" not in method_to_stats or 100 not in method_to_stats["random"]):
            print(
                f"[WARN][{dataset}] 请求绘制 kr=100，但未找到 random 的 kr=100 合法结果，"
                "因此无法为所有方法补齐 kr=100 节点。"
            )
        if missing_methods:
            print(f"[WARN][{dataset}] 未保存结果的方法（已忽略）: {', '.join(missing_methods)}")

        print_table(dataset, valid_methods, method_to_stats, method_to_mean)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight")
    print(f"\nSaved dual figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
