#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


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
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument(
        "--methods",
        nargs="+",
        # default=[
        #     "random", "herding", "E2LN", "GraNd", "Forgetting", "MoSo",
        #     "yangclip", "learned_group"
        # ],
        default=[
            "random", "naive_topk", "learned_topk", "naive_group", "learned_group"
        ],
        help="Selection methods to compare",
    )
    parser.add_argument(
        "--kr",
        default="20,30,40,60,80",
        help="Keep ratio list, e.g. '20,30,40,50,60,70,80,90,100'",
    )
    parser.add_argument("--output", default=None, help="Output image path")
    return parser.parse_args()


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
    最后一个方法默认突出显示：
    - 更鲜艳的颜色
    - 略粗的线
    - 略大的五角星节点
    """
    style_map: dict[str, dict] = {}

    for method in methods[:-1]:
        style_map[method] = {
            "linewidth": 0.9,
            "markersize": 2.6,
            "markeredgewidth": 0.5,
            "alpha": 0.9,
            "color": None,
            "zorder": 2,
        }

    if methods:
        style_map[methods[-1]] = {
            "linewidth": 1.3,
            "markersize": 4.8,
            "markeredgewidth": 0.7,
            "alpha": 1.0,
            "color": "red",
            "zorder": 5,
        }

    return style_map


def load_method_mean_results(
    method_root: Path,
    keep_ratios: list[int],
) -> dict[int, float]:
    """
    读取某一方法在指定数据集和模型下、跨 seed 的平均结果。
    """
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


def inject_kr100_from_random(
    method_to_mean: dict[str, dict[int, float]],
    methods: list[str],
) -> None:
    """
    特殊处理 kr=100：
    - 只有 random 的 kr=100 文件是合法来源；
    - 其他方法一律复用 random 的 kr=100 结果；
    - 从而所有方法在 kr=100 处重合，但 marker 不同。
    """
    if "random" not in method_to_mean or 100 not in method_to_mean["random"]:
        return

    random_kr100 = method_to_mean["random"][100]
    for method in methods:
        if method not in method_to_mean:
            continue
        method_to_mean[method][100] = random_kr100


def main() -> None:
    args = parse_args()
    keep_ratios = parse_kr_list(args.kr)
    result_root = Path(args.result_dir)
    methods = args.methods

    marker_map = build_marker_map(methods)
    style_map = build_style_map(methods)

    output_name = args.output or f"{args.dataset}_{args.model}.png"
    output_path = Path("picture") / Path(output_name).name

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    missing_methods: list[str] = []
    valid_methods: list[str] = []
    method_to_mean: dict[str, dict[int, float]] = {}

    # 先读取各方法结果
    for method in methods:
        method_root = result_root / method / args.dataset / args.model
        avg_by_kr = load_method_mean_results(method_root, keep_ratios)

        if not avg_by_kr:
            missing_methods.append(method)
            continue

        method_to_mean[method] = avg_by_kr
        valid_methods.append(method)

    # 特殊处理 kr=100：所有方法统一复用 random 的 kr=100 结果
    if 100 in keep_ratios:
        inject_kr100_from_random(method_to_mean, valid_methods)

    # 再开始画图
    for method in valid_methods:
        avg_by_kr = method_to_mean.get(method, {})
        x_values = [kr for kr in keep_ratios if kr in avg_by_kr]
        y_values = [avg_by_kr[kr] for kr in x_values]

        if not x_values:
            print(f"[WARN] method={method} has no results for requested keep ratios: {keep_ratios}")
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

        ax.plot(
            x_values,
            y_values,
            **plot_kwargs,
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

    print("\nMean accuracy by keep ratio (4 decimal places):")
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
        kr_to_value = method_to_mean.get(method, {})
        row = [method]
        for kr in keep_ratios:
            val = kr_to_value.get(kr)
            if val is None:
                row.append("-")
            else:
                cell = f"{val:.4f}"
                if (
                    kr != 100
                    and kr in best_by_kr
                    and math.isclose(val, best_by_kr[kr], rel_tol=1e-12, abs_tol=1e-12)
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

    ax.set_xlabel("Keep Ratio (kr)")
    ax.set_ylabel("Accuracy (mean of last 10 epochs)")
    ax.set_title(f"{args.dataset.upper()} {args.model} - Mean Accuracy")

    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.7)
    ax.set_xticks(keep_ratios)

    all_y = [v for row in method_to_mean.values() for v in row.values()]
    if all_y:
        ymin = min(all_y)
        ymax = max(all_y)
        eps = max(0.0003, 0.015 * (ymax - ymin))
        ax.set_ylim(ymin - eps, ymax + eps)

    if valid_methods:
        # 将图例放入坐标轴内部右下区域，利用折线图右下角天然空白
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    if 100 in keep_ratios and ("random" not in method_to_mean or 100 not in method_to_mean["random"]):
        print("[WARN] 请求绘制 kr=100，但未找到 random 的 kr=100 合法结果，因此无法为所有方法补齐 kr=100 节点。")

    if missing_methods:
        print(f"未保存结果的方法（已忽略）: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()
