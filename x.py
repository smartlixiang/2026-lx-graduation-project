"""Migrate result files to the new directory structure."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


RESULT_PATTERN = re.compile(r"^result_(\d+)_(.+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move result files to result/[method]/[dataset]/[model]/[seed]/.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="result",
        help="Root directory containing result files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned moves without changing files.",
    )
    return parser.parse_args()


def find_legacy_results(result_root: Path) -> list[Path]:
    legacy_files: list[Path] = []
    for path in result_root.rglob("result_*.json"):
        rel_parts = path.relative_to(result_root).parts
        if len(rel_parts) != 4:
            continue
        if RESULT_PATTERN.match(path.name):
            legacy_files.append(path)
    return legacy_files


def migrate_results(result_root: Path, dry_run: bool) -> None:
    legacy_files = find_legacy_results(result_root)
    if not legacy_files:
        print("No legacy result files found.")
        return

    for legacy_path in legacy_files:
        dataset, model, seed, filename = legacy_path.relative_to(result_root).parts
        match = RESULT_PATTERN.match(filename)
        if not match:
            print(f"Skip invalid filename: {legacy_path}")
            continue
        cut_ratio, method = match.groups()
        new_dir = result_root / method / dataset / model / seed
        new_path = new_dir / f"result_{cut_ratio}.json"
        if new_path.exists():
            print(f"Skip existing: {new_path}")
            continue
        print(f"{legacy_path} -> {new_path}")
        if not dry_run:
            new_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_path), str(new_path))


def main() -> None:
    args = parse_args()
    migrate_results(Path(args.result_root), args.dry_run)


if __name__ == "__main__":
    main()
