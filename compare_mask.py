from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MASK_ROOT = Path("mask")
MISSING = "-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较两种方法在相同 seed/kr 下子集样本重合率")
    parser.add_argument("--A", required=True, help="方法 A（对应 mask/<A>/...）")
    parser.add_argument("--B", required=True, help="方法 B（对应 mask/<B>/...）")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    return parser.parse_args()


def collect_masks(method_dir: Path) -> dict[int, dict[int, Path]]:
    """Return {kr: {seed: mask_path}}."""
    index: dict[int, dict[int, Path]] = {}
    for seed_dir in sorted(method_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name)
        except ValueError:
            continue

        for mask_path in seed_dir.glob("mask_*.npz"):
            stem = mask_path.stem
            if not stem.startswith("mask_"):
                continue
            try:
                kr = int(stem.split("_", 1)[1])
            except ValueError:
                continue
            index.setdefault(kr, {})[seed] = mask_path
    return index


def load_mask(mask_path: Path) -> np.ndarray:
    with np.load(mask_path) as data:
        if "mask" in data:
            arr = data["mask"]
        else:
            if not data.files:
                raise ValueError(f"空的 mask 文件: {mask_path}")
            arr = data[data.files[0]]
    return np.asarray(arr).astype(bool)


def overlap_rate(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"mask 形状不一致: {mask_a.shape} vs {mask_b.shape}")

    inter = int(np.logical_and(mask_a, mask_b).sum())
    a_count = int(mask_a.sum())
    b_count = int(mask_b.sum())
    denom = a_count + b_count
    if denom == 0:
        return 1.0
    return (2.0 * inter) / float(denom)


def render_table(rows: list[tuple[int, str, str]]) -> str:
    headers = ["kr", "overlap", "pairs"]
    str_rows = [[str(kr), overlap, pairs] for kr, overlap, pairs in rows]

    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def join(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [join(headers), sep]
    lines.extend(join(r) for r in str_rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    method_a_dir = MASK_ROOT / args.A / args.dataset
    method_b_dir = MASK_ROOT / args.B / args.dataset

    if not method_a_dir.exists():
        print(f"方法不存在: {args.A} ({method_a_dir})")
        return
    if not method_b_dir.exists():
        print(f"方法不存在: {args.B} ({method_b_dir})")
        return

    masks_a = collect_masks(method_a_dir)
    masks_b = collect_masks(method_b_dir)

    all_krs = sorted(set(masks_a) | set(masks_b))
    if not all_krs:
        print("未找到任何 mask 文件。")
        return

    rows: list[tuple[int, str, str]] = []
    for kr in all_krs:
        by_seed_a = masks_a.get(kr, {})
        by_seed_b = masks_b.get(kr, {})
        common_seeds = sorted(set(by_seed_a) & set(by_seed_b))

        if not common_seeds:
            rows.append((kr, MISSING, "0"))
            print(f"kr={kr}: 缺失可比较的 seed 对")
            continue

        rates: list[float] = []
        for seed in common_seeds:
            mask_a = load_mask(by_seed_a[seed])
            mask_b = load_mask(by_seed_b[seed])
            rates.append(overlap_rate(mask_a, mask_b))

        mean_rate = float(np.mean(rates))
        rows.append((kr, f"{mean_rate:.6f}", str(len(rates))))

    print(render_table(rows))


if __name__ == "__main__":
    main()
