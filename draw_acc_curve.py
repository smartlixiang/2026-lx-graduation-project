#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator


# 固定颜色顺序，保证不同运行之间的方法配色一致。
# 最后一个方法仍作为重点方法单独使用深红色。
COLOR_POOL = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#4c78a8",  # muted blue
    "#f58518",  # muted orange
    "#54a24b",  # muted green
]


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
    parser.add_argument("--dataset", default="cifar100", help="Dataset name")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "random", "herding", "EL2N", "GraNd", "Forgetting", "MDS",
            "MoSo", "yangclip", "RLSelector", "ours"
        ],
        # default=["random", "ours", "unseen_learned_group"],
        help="Selection methods to compare",
    )
    parser.add_argument(
        "--kr",
        default="20,30,40,50,60,70,80,90,100",
        help="retention ratio list, e.g. '20,30,40,50,60,70,80,90,100'",
    )
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Output image resolution (default: 320)",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    """配置更接近论文实验图的清晰、紧凑绘图风格。"""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.0,
            "axes.labelsize": 16.2,
            "axes.linewidth": 1.18,
            "xtick.labelsize": 13.8,
            "ytick.labelsize": 13.8,
            "legend.fontsize": 14.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "axes.unicode_minus": False,
        }
    )


def parse_kr_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--kr cannot be empty")
    return sorted({int(v) for v in values})


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


def build_marker_map(methods: list[str]) -> dict[str, str]:
    """
    给每个方法分配不同 marker。
    最后一个方法默认视为重点方法，单独使用五角星 marker。
    """
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
    """
    放大折线和数据点，并通过固定颜色、白色描边和更高层级减少重叠。
    最后一个方法默认作为重点方法突出显示。
    """
    if len(methods) - 1 > len(COLOR_POOL):
        raise ValueError(
            f"当前方法数为 {len(methods)}，普通颜色数量不足，请扩充 COLOR_POOL。"
        )

    style_map: dict[str, dict] = {}
    for i, method in enumerate(methods[:-1]):
        style_map[method] = {
            "linewidth": 1.70,
            "markersize": 5.9,
            "markeredgewidth": 0.80,
            "markeredgecolor": "white",
            "alpha": 0.97,
            "color": COLOR_POOL[i],
            "zorder": 3,
        }

    if methods:
        style_map[methods[-1]] = {
            "linewidth": 2.65,
            "markersize": 10.4,
            "markeredgewidth": 0.90,
            "markeredgecolor": "#8b0000",
            "alpha": 1.0,
            "color": "#e31a1c",
            "zorder": 8,
        }

    return style_map


def load_method_seed_stats(
    method_root: Path,
    keep_ratios: list[int],
) -> dict[int, tuple[float, float]]:
    """
    读取某一方法在指定数据集和模型下、跨 seed 的聚合结果。
    返回每个 retention ratio 的 (mean, std)。
    """
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


def inject_kr100_from_random(
    method_to_stats: dict[str, dict[int, tuple[float, float]]],
    methods: list[str],
) -> None:
    """
    特殊处理 kr=100：
    - 只有 random 的 kr=100 文件是合法来源；
    - 其他方法一律复用 random 的 kr=100 结果；
    - 从而所有方法在 kr=100 处重合，但 marker 不同。
    """
    if "random" not in method_to_stats or 100 not in method_to_stats["random"]:
        return

    random_kr100_stats = method_to_stats["random"][100]
    for method in methods:
        if method not in method_to_stats:
            continue
        method_to_stats[method][100] = random_kr100_stats


def choose_y_tick_step(ymin: float, ymax: float) -> float:
    """根据百分数纵轴跨度选择较密但不拥挤的主刻度间隔。"""
    span = ymax - ymin
    if span >= 32.0:
        return 5.0
    if span >= 18.0:
        return 2.5
    if span >= 10.0:
        return 2.0
    if span >= 5.0:
        return 1.0
    if span >= 2.5:
        return 0.5
    return 0.25


def compute_y_limits(
    method_to_mean: dict[str, dict[int, float]],
    keep_ratios: list[int],
) -> tuple[float, float] | None:
    """
    纵轴下界采用“最小 kr 的所有方法均值中的最小值 × 0.8”。
    这样可避免个别极低离群点把整张图拉得过扁。
    若有点低于该下界，它会被裁到图外，只保留折线连接效果。
    """
    all_y = [
        value * 100.0
        for row in method_to_mean.values()
        for value in row.values()
    ]
    if not all_y or not keep_ratios:
        return None

    left_kr = min(keep_ratios)
    right_kr = max(keep_ratios)

    left_values = [
        row[left_kr] * 100.0
        for row in method_to_mean.values()
        if left_kr in row
    ]
    right_values = [
        row[right_kr] * 100.0
        for row in method_to_mean.values()
        if right_kr in row
    ]

    if left_values:
        ymin = mean(left_values) * 0.98
    else:
        ymin = min(all_y)

    if right_values:
        ymax_ref = max(right_values)
    else:
        ymax_ref = max(all_y)

    ymax = max(max(all_y), ymax_ref)
    upper_pad = max(0.18, 0.018 * (ymax - ymin))
    return ymin, ymax + upper_pad


def build_legend_handles(
    valid_methods: list[str],
    focus_method: str | None,
    marker_map: dict[str, str],
    style_map: dict[str, dict],
) -> tuple[list[Line2D], list[str]]:
    """
    构造独立图例句柄：
    - 除最后一个重点方法外，其余所有图例符号大小完全一致；
    - 重点方法允许更大一些。
    """
    handles: list[Line2D] = []
    labels: list[str] = []

    regular_ms = 7.4
    regular_lw = 2.2
    regular_mew = 0.95

    focus_ms = 10.2
    focus_lw = 2.9
    focus_mew = 1.00

    for method in valid_methods:
        is_focus = method == focus_method
        handles.append(
            Line2D(
                [0],
                [0],
                color=style_map[method]["color"],
                marker=marker_map[method],
                linestyle="-",
                linewidth=focus_lw if is_focus else regular_lw,
                markersize=focus_ms if is_focus else regular_ms,
                markeredgewidth=focus_mew if is_focus else regular_mew,
                markeredgecolor=style_map[method]["markeredgecolor"],
                alpha=1.0,
            )
        )
        labels.append(method)

    return handles, labels


def main() -> None:
    args = parse_args()
    configure_plot_style()

    keep_ratios = parse_kr_list(args.kr)
    result_root = Path(args.result_dir)
    methods = args.methods

    marker_map = build_marker_map(methods)
    style_map = build_style_map(methods)

    output_name = args.output or f"{args.dataset}_{args.model}.png"
    output_path = Path("picture") / Path(output_name).name

    # 取消标题后，可将更多空间分配给坐标轴和右下角图例。
    fig, ax = plt.subplots(figsize=(9.0, 7.45))
    missing_methods: list[str] = []
    valid_methods: list[str] = []
    method_to_stats: dict[str, dict[int, tuple[float, float]]] = {}

    # 先读取各方法结果
    for method in methods:
        method_root = result_root / method / args.dataset / args.model
        stats_by_kr = load_method_seed_stats(method_root, keep_ratios)

        if not stats_by_kr:
            missing_methods.append(method)
            continue

        method_to_stats[method] = stats_by_kr
        valid_methods.append(method)

    # 特殊处理 kr=100：所有方法统一复用 random 的 kr=100 结果
    if 100 in keep_ratios:
        inject_kr100_from_random(method_to_stats, valid_methods)

    method_to_mean: dict[str, dict[int, float]] = {
        method: {kr: stats[0] for kr, stats in kr_to_stats.items()}
        for method, kr_to_stats in method_to_stats.items()
    }

    # 横轴使用 retention ratio 的顺序位置，而不是数值本身。
    # 因此 20、30、50、70、90、100 等节点在图中保持等间距，
    # 但刻度标签仍显示真实 retention ratio。
    x_position_by_kr = {
        keep_ratio: position
        for position, keep_ratio in enumerate(keep_ratios)
    }

    # 绘图时只转换展示单位，内部统计和排名仍使用原始 0-1 准确率。
    for method in valid_methods:
        mean_by_kr = method_to_mean.get(method, {})
        available_keep_ratios = [
            kr for kr in keep_ratios if kr in mean_by_kr
        ]
        x_values = [x_position_by_kr[kr] for kr in available_keep_ratios]
        y_values = [mean_by_kr[kr] * 100.0 for kr in available_keep_ratios]

        if not available_keep_ratios:
            print(f"[WARN] method={method} has no results for requested retention ratios: {keep_ratios}")
            continue

        ax.plot(
            x_values,
            y_values,
            marker=marker_map[method],
            linewidth=style_map[method]["linewidth"],
            markersize=style_map[method]["markersize"],
            markeredgewidth=style_map[method]["markeredgewidth"],
            markeredgecolor=style_map[method]["markeredgecolor"],
            color=style_map[method]["color"],
            alpha=style_map[method]["alpha"],
            zorder=style_map[method]["zorder"],
            clip_on=True,
        )

    # kr=100 不参与排名
    ranking_keep_ratios = [kr for kr in keep_ratios if kr != 100]

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

    print("\nMean accuracy by retention ratio (2 decimal places):")
    header = ["method"] + [str(kr) for kr in keep_ratios] + ["avg_rank"]

    # kr=100 不参与“最优结果”加粗
    bold_keep_ratios = [kr for kr in keep_ratios if kr != 100]

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
        for kr in keep_ratios:
            stats = kr_to_stats.get(kr)
            if stats is None:
                row.append("-")
            else:
                mean_val, std_val = stats
                # 终端表格不附加百分号；标准差补零到至少两位整数宽度。
                cell = f"{mean_val * 100.0:.2f}±{std_val * 100.0:05.2f}"
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
    col_count = len(header)
    widths = [0] * col_count
    for r in cols:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    header_line = (
        f"{header[0].ljust(widths[0])}  "
        + "  ".join(header[i].rjust(widths[i]) for i in range(1, col_count))
    )
    print(header_line)

    for row in table_rows:
        line = (
            f"{row[0].ljust(widths[0])}  "
            + "  ".join(row[i].rjust(widths[i]) for i in range(1, col_count))
        )
        print(line)

    ax.set_xlabel("Retention Ratio (kr)", labelpad=8, fontweight="medium")
    ax.set_ylabel("Accuracy (%)", labelpad=8, fontweight="medium")
    # 按要求：单张图不显示标题。

    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        linestyle="-",
        color="#d9d9d9",
        alpha=0.78,
        linewidth=0.84,
    )
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        color="#ececec",
        alpha=0.52,
        linewidth=0.58,
    )

    x_tick_positions = list(range(len(keep_ratios)))
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([str(kr) for kr in keep_ratios])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", width=1.1, length=5.2, pad=5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=3.0)

    if x_tick_positions:
        # 使用序号坐标后，各 retention ratio 节点在视觉上严格等间距。
        x_margin = 0.15
        ax.set_xlim(
            x_tick_positions[0] - x_margin,
            x_tick_positions[-1] + x_margin,
        )

    y_limits = compute_y_limits(method_to_mean, keep_ratios)
    if y_limits is not None:
        ymin, ymax = y_limits
        ax.set_ylim(ymin, ymax)

        major_step = choose_y_tick_step(ymin, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if major_step < 1.0 or not float(major_step).is_integer():
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    for spine in ax.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(1.15)

    if valid_methods:
        focus_method = methods[-1] if methods and methods[-1] in valid_methods else None
        legend_handles, legend_labels = build_legend_handles(
            valid_methods=valid_methods,
            focus_method=focus_method,
            marker_map=marker_map,
            style_map=style_map,
        )

        # 右下角视作安全区域，尽量放大图例和文字。
        legend = ax.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            bbox_to_anchor=(0.996, 0.012),
            frameon=True,
            fancybox=False,
            framealpha=0.985,
            facecolor="white",
            edgecolor="#888888",
            fontsize=14.8,
            ncol=1,
            handlelength=2.70,
            handletextpad=0.82,
            borderpad=0.94,
            labelspacing=0.56,
            columnspacing=1.0,
        )
        legend.get_frame().set_linewidth(1.08)

    # 手动留白比 tight_layout 更稳定，避免大图例和大字号挤压坐标轴。
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.108, top=0.985)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    if 100 in keep_ratios and ("random" not in method_to_stats or 100 not in method_to_stats["random"]):
        print("[WARN] 请求绘制 kr=100，但未找到 random 的 kr=100 合法结果，因此无法为所有方法补齐 kr=100 节点。")

    if missing_methods:
        print(f"未保存结果的方法（已忽略）: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()