from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "比较同 dataset/model/seed/cr 下 group 与 topk mask 的重合度（仅统计两者都存在的 pair）。"
        )
    )
    parser.add_argument("--mask-root", type=str, default="mask", help="mask 根目录")
    parser.add_argument("--dataset", type=str, default="cifar10", help="数据集名")
    parser.add_argument("--model", type=str, default="resnet50", help="模型名")
    parser.add_argument(
        "--weight-group",
        type=str,
        default="naive",
        help="权重组名（对应目录前缀，如 naive_topk / naive_group）",
    )
    parser.add_argument("--seed", type=int, default=None, help="仅比较指定 seed")
    parser.add_argument("--cr", type=int, default=None, help="仅比较指定 cut ratio")
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "mask" not in data:
            raise KeyError(f"{path} 中缺少 'mask' 键")
        return np.asarray(data["mask"], dtype=np.uint8)


def overlap_stats(group_mask: np.ndarray, topk_mask: np.ndarray) -> dict[str, float]:
    if group_mask.shape != topk_mask.shape:
        raise ValueError("mask shape 不一致")

    g = group_mask.astype(bool)
    t = topk_mask.astype(bool)

    g_count = int(g.sum())
    t_count = int(t.sum())
    inter = int(np.logical_and(g, t).sum())
    union = int(np.logical_or(g, t).sum())

    jaccard = (inter / union) if union else 1.0
    overlap_on_group = (inter / g_count) if g_count else 1.0
    overlap_on_topk = (inter / t_count) if t_count else 1.0

    return {
        "group_selected": g_count,
        "topk_selected": t_count,
        "intersection": inter,
        "union": union,
        "jaccard": jaccard,
        "overlap_group": overlap_on_group,
        "overlap_topk": overlap_on_topk,
    }


def format_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "seed",
        "cr",
        "group_selected",
        "topk_selected",
        "intersection",
        "union",
        "jaccard",
        "overlap/group",
        "overlap/topk",
    ]

    str_rows: list[list[str]] = []
    for row in rows:
        str_rows.append(
            [
                str(row["seed"]),
                str(row["cr"]),
                str(row["group_selected"]),
                str(row["topk_selected"]),
                str(row["intersection"]),
                str(row["union"]),
                f"{row['jaccard']:.4f}",
                f"{row['overlap_group']:.4f}",
                f"{row['overlap_topk']:.4f}",
            ]
        )

    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def join_row(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [join_row(headers), sep]
    lines.extend(join_row(r) for r in str_rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    mask_root = Path(args.mask_root)
    mode_group = f"{args.weight_group}_group"
    mode_topk = f"{args.weight_group}_topk"

    group_base = mask_root / mode_group / args.dataset / args.model
    topk_base = mask_root / mode_topk / args.dataset / args.model

    if not group_base.exists() or not topk_base.exists():
        raise FileNotFoundError(
            f"未找到对比目录: {group_base} 或 {topk_base}"
        )

    rows: list[dict[str, object]] = []
    for group_mask_path in sorted(group_base.glob("*/mask_*.npz")):
        seed_text = group_mask_path.parent.name
        try:
            seed = int(seed_text)
        except ValueError:
            continue

        if args.seed is not None and seed != args.seed:
            continue

        stem = group_mask_path.stem
        if not stem.startswith("mask_"):
            continue
        try:
            cr = int(stem.split("_", 1)[1])
        except ValueError:
            continue

        if args.cr is not None and cr != args.cr:
            continue

        topk_mask_path = topk_base / str(seed) / f"mask_{cr}.npz"
        if not topk_mask_path.exists():
            continue

        group_mask = load_mask(group_mask_path)
        topk_mask = load_mask(topk_mask_path)
        stats = overlap_stats(group_mask, topk_mask)
        rows.append({"seed": seed, "cr": cr, **stats})

    if not rows:
        print("未找到同时存在的 group/topk mask 对。")
        return

    rows.sort(key=lambda x: (int(x["seed"]), int(x["cr"])))
    print(format_table(rows))


if __name__ == "__main__":
    main()
