"""Download Tiny-ImageNet into the project's data directory using PyTorch tooling.

This script downloads and extracts tiny-imagenet-200, then converts the validation
split layout to ImageFolder-compatible class subfolders:

    data/tiny-imagenet-200/val/<class_name>/*.JPEG

Usage:
    python download.py
    python download.py --data-root ./data --force
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from torchvision.datasets.utils import download_url, extract_archive


TINY_IMAGENET_URLS = (
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
)
ARCHIVE_NAME = "tiny-imagenet-200.zip"
EXTRACTED_DIR = "tiny-imagenet-200"
MAX_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tiny-imagenet-200 to data/ directory.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory to store the downloaded dataset (default: ./data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extract by removing existing archive/folder.",
    )
    return parser.parse_args()


def _ensure_val_class_dirs(dataset_dir: Path) -> None:
    """Reorganize val split into ImageFolder layout if needed."""

    val_dir = dataset_dir / "val"
    images_dir = val_dir / "images"
    annotations_path = val_dir / "val_annotations.txt"

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Validation annotation file not found: {annotations_path}")

    # If val/images is absent, we assume dataset has already been reorganized.
    if not images_dir.exists():
        print("[info] val/images not found; assuming validation split already reorganized.")
        return

    moved_count = 0
    with annotations_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            src = images_dir / image_name
            dst_dir = val_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / image_name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                moved_count += 1

    # Clean up empty images dir to avoid confusion.
    try:
        images_dir.rmdir()
    except OSError:
        pass

    print(f"[info] Validation split reorganized for ImageFolder (moved {moved_count} images).")


def _is_valid_zip(path: Path) -> bool:
    """Return True only if file exists and is a readable zip archive."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # testzip returns first bad filename, or None if all good.
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def _download_with_retry(data_root: Path, archive_path: Path) -> None:
    """Download tiny-imagenet archive with corruption checks and retries."""

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        if archive_path.exists():
            archive_path.unlink()
            print(f"[info] Removed stale archive before retry: {archive_path}")

        print(f"[info] Downloading Tiny-ImageNet (attempt {attempt}/{MAX_RETRIES})")
        for url in TINY_IMAGENET_URLS:
            try:
                print(f"[info]   source: {url}")
                download_url(
                    url=url,
                    root=str(data_root),
                    filename=ARCHIVE_NAME,
                    max_redirect_hops=5,
                )
                if not _is_valid_zip(archive_path):
                    raise zipfile.BadZipFile(
                        "Downloaded file is not a valid zip archive "
                        "(likely interrupted download or network/proxy response issue)."
                    )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[warn]   source failed: {exc}")

    raise RuntimeError(
        "Failed to download a valid tiny-imagenet-200 archive after multiple retries. "
        "Please check network/proxy and rerun with --force."
    ) from last_error


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    archive_path = data_root / ARCHIVE_NAME
    dataset_dir = data_root / EXTRACTED_DIR

    if args.force:
        if archive_path.exists():
            archive_path.unlink()
            print(f"[info] Removed existing archive: {archive_path}")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            print(f"[info] Removed existing extracted dataset: {dataset_dir}")

    if dataset_dir.exists():
        print(f"[info] Found existing dataset directory: {dataset_dir}")
    else:
        _download_with_retry(data_root, archive_path)
        print(f"[info] Extracting archive: {archive_path}")
        extract_archive(from_path=str(archive_path), to_path=str(data_root), remove_finished=False)
        print(f"[info] Downloaded and extracted to: {dataset_dir}")

    _ensure_val_class_dirs(dataset_dir)

    print("[done] tiny-imagenet-200 is ready.")
    print(f"[done] Train path: {dataset_dir / 'train'}")
    print(f"[done] Val path:   {dataset_dir / 'val'}")


if __name__ == "__main__":
    main()
