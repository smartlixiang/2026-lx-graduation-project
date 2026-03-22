#!/usr/bin/env python3
"""将 mask/result 目录从旧结构迁移到新结构。

旧结构：
  <root>/<mode>/<dataset>/<model>/<seed>/<file>
新结构：
  <root>/<mode>/<dataset>/<seed>/<file>

默认会同时处理 `result` 和 `mask` 两个根目录。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOTS = ["result", "mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 result/mask 目录从包含 model 的层级迁移为 dataset/seed 层级。"
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help="要迁移的根目录列表，默认: result mask",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的操作，不实际移动文件。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖；默认遇到冲突时报错并跳过该文件。",
    )
    parser.add_argument(
        "--keep-empty-dirs",
        action="store_true",
        help="迁移后保留旧空目录；默认会自动删除空目录。",
    )
    return parser.parse_args()


def is_old_layout_file(path: Path, root: Path) -> bool:
    """判断文件是否匹配旧结构 <root>/<mode>/<dataset>/<model>/<seed>/<file>。"""
    rel_parts = path.relative_to(root).parts
    return len(rel_parts) >= 5 and rel_parts[3].isdigit()


def build_new_path(path: Path, root: Path) -> Path:
    """将旧结构路径映射到新结构路径。"""
    rel_parts = path.relative_to(root).parts
    mode, dataset, _model, seed = rel_parts[:4]
    suffix_parts = rel_parts[4:]
    return root / mode / dataset / seed / Path(*suffix_parts)


def remove_empty_dirs(root: Path) -> None:
    """自底向上删除空目录。"""
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda x: len(x.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            # 非空目录或权限问题，忽略
            pass


def migrate_root(root: Path, dry_run: bool, overwrite: bool, keep_empty_dirs: bool) -> tuple[int, int, int]:
    """迁移单个根目录，返回 (moved, skipped, conflicts)。"""
    if not root.exists():
        print(f"[Skip] 根目录不存在: {root}")
        return (0, 0, 0)

    files = [p for p in root.rglob("*") if p.is_file() and is_old_layout_file(p, root)]
    moved = 0
    skipped = 0
    conflicts = 0

    for src in sorted(files):
        dst = build_new_path(src, root)

        # 已经是新路径（例如脚本多次执行）
        if src == dst:
            skipped += 1
            continue

        print(f"[Move] {src} -> {dst}")
        if dry_run:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if overwrite:
                dst.unlink()
            else:
                print(f"[Conflict] 目标已存在，跳过: {dst}")
                conflicts += 1
                continue

        shutil.move(str(src), str(dst))
        moved += 1

    if not dry_run and not keep_empty_dirs:
        remove_empty_dirs(root)

    return (moved, skipped, conflicts)


def main() -> None:
    args = parse_args()

    total_moved = 0
    total_skipped = 0
    total_conflicts = 0

    for root_name in args.roots:
        root = Path(root_name)
        moved, skipped, conflicts = migrate_root(
            root=root,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            keep_empty_dirs=args.keep_empty_dirs,
        )
        total_moved += moved
        total_skipped += skipped
        total_conflicts += conflicts

    print("\n=== Summary ===")
    print(f"Moved: {total_moved}")
    print(f"Skipped(already new layout): {total_skipped}")
    print(f"Conflicts: {total_conflicts}")


if __name__ == "__main__":
    main()
