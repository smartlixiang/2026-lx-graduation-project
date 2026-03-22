"""Migrate result files to the model-aware directory layout.

Old layout:
    result/[mode]/[dataset]/[seed]/result_[kr].json

New layout:
    result/[mode]/[dataset]/[model]/[seed]/result_[kr].json

Per the current project convention, existing historical results are assumed to use
`resnet50` as the model name.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_MODEL = "resnet50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("result"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_old_layout_files(root: Path):
    for path in root.rglob("result_*.json"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(relative_parts) != 4:
            continue
        yield path, relative_parts


def remove_empty_parents(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    stop_dir = stop_dir.resolve()
    while current.exists() and current.resolve() != stop_dir:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    moved_count = 0
    skipped_count = 0
    for src_path, (mode, dataset, seed, filename) in sorted(iter_old_layout_files(root)):
        dst_path = root / mode / dataset / args.model / seed / filename
        if src_path == dst_path:
            skipped_count += 1
            continue
        if dst_path.exists():
            raise FileExistsError(f"Target already exists: {dst_path}")

        print(f"MOVE {src_path} -> {dst_path}")
        moved_count += 1
        if args.dry_run:
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        remove_empty_parents(src_path.parent, root)

    print(f"Done. moved={moved_count}, skipped={skipped_count}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
