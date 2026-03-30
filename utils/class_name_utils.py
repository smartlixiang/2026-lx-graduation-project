"""Utilities for resolving human-readable class names used in CLIP text prompts.

Tiny-ImageNet uses WNID directory names (e.g. ``n01443537``) in ImageFolder.
Those ids are not natural-language category names and should not be fed into
CLIP prompts directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from dataset.dataset_config import TINY_IMAGENET


def _load_tiny_imagenet_wnid_name_map(data_root: str | Path) -> tuple[set[str], dict[str, str]]:
    dataset_root = Path(data_root) / "tiny-imagenet-200"
    wnids_path = dataset_root / "wnids.txt"
    words_path = dataset_root / "words.txt"

    if not wnids_path.exists() or not words_path.exists():
        raise FileNotFoundError(
            "tiny-imagenet requires both wnids.txt and words.txt to build natural-language "
            f"prompts, but missing file(s) under: {dataset_root}"
        )

    wnids = {
        line.strip()
        for line in wnids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    wnid_to_name: dict[str, str] = {}
    for line in words_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "\t" not in line:
            continue
        wnid, raw_names = line.split("\t", 1)
        wnid = wnid.strip()
        if not wnid:
            continue
        # words.txt often stores synonyms as comma-separated names.
        first_name = raw_names.split(",", 1)[0].strip()
        if first_name:
            wnid_to_name[wnid] = first_name
    return wnids, wnid_to_name


def resolve_class_names_for_prompts(
    dataset_name: str,
    data_root: str | Path,
    class_names: Sequence[str],
) -> list[str]:
    """Return class names for prompt construction.

    For tiny-imagenet, map WNID folder names to natural-language English names
    via ``wnids.txt`` + ``words.txt``.
    """

    normalized_dataset_name = str(dataset_name).strip().lower()
    original_names = [str(name) for name in class_names]
    if normalized_dataset_name != TINY_IMAGENET:
        return original_names

    wnids, wnid_to_name = _load_tiny_imagenet_wnid_name_map(data_root)
    resolved: list[str] = []
    for name in original_names:
        cleaned = name.strip()
        if cleaned in wnid_to_name:
            resolved.append(wnid_to_name[cleaned])
            continue
        if cleaned in wnids and cleaned not in wnid_to_name:
            raise KeyError(
                f"WNID '{cleaned}' exists in wnids.txt but has no entry in words.txt; "
                "cannot build natural-language prompts for tiny-imagenet."
            )
        # If the input names are already human-readable names (not WNIDs),
        # keep them as-is.
        resolved.append(cleaned)
    return resolved


def build_class_prompts(
    dataset_name: str,
    data_root: str | Path,
    class_names: Sequence[str],
    prompt_template: str,
    *,
    debug: bool = False,
    debug_count: int = 5,
) -> tuple[list[str], list[str]]:
    """Resolve class names and build prompt strings from a template."""

    resolved_names = resolve_class_names_for_prompts(dataset_name, data_root, class_names)
    prompts = [prompt_template.format(name) for name in resolved_names]

    if debug and str(dataset_name).strip().lower() == TINY_IMAGENET:
        preview = prompts[: max(0, int(debug_count))]
        print(f"[Tiny-ImageNet prompt debug] first {len(preview)} prompts: {preview}")
    return resolved_names, prompts
