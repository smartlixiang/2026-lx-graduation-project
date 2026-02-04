"""Utilities for caching static scores (SA/Div/DDS) per dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in text)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_cache_path(
    cache_root: Path,
    dataset: str,
    clip_model: str,
    adapter_path: str | None,
    div_k: int,
    dds_k: float,
    prompt_template: str,
) -> Path:
    dataset_dir = cache_root / _sanitize(dataset)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    clip_tag = _sanitize(clip_model)
    prompt_tag = hashlib.sha1(prompt_template.encode("utf-8")).hexdigest()[:8]
    adapter_hash = "no-adapter"
    if adapter_path:
        adapter_hash = _hash_file(Path(adapter_path))[:12]
    filename = f"static_{clip_tag}_div{div_k}_dds{dds_k}_{prompt_tag}_{adapter_hash}.npz"
    return dataset_dir / filename


def _validate_cache(
    cache_path: Path,
    meta: dict[str, object],
    num_samples: int,
) -> dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    if "meta" not in data:
        return None
    try:
        cached_meta = json.loads(str(data["meta"]))
    except json.JSONDecodeError:
        return None
    for key, value in meta.items():
        if cached_meta.get(key) != value:
            return None
    for key in ("sa", "div", "dds", "labels"):
        if key not in data:
            return None
        if data[key].shape != (num_samples,):
            return None
    return {
        "sa": data["sa"],
        "div": data["div"],
        "dds": data["dds"],
        "labels": data["labels"],
    }


def get_or_compute_static_scores(
    *,
    cache_root: str | Path,
    dataset: str,
    clip_model: str,
    adapter_path: str | None,
    div_k: int,
    dds_k: float,
    prompt_template: str,
    num_samples: int,
    compute_fn: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    cache_root = Path(cache_root)
    cache_path = _build_cache_path(
        cache_root, dataset, clip_model, adapter_path, div_k, dds_k, prompt_template
    )
    meta = {
        "dataset": dataset,
        "clip_model": clip_model,
        "adapter_path": adapter_path or "",
        "adapter_sha1": _hash_file(Path(adapter_path)) if adapter_path else "",
        "div_k": div_k,
        "dds_k": dds_k,
        "prompt_template": prompt_template,
        "num_samples": num_samples,
    }
    cached = _validate_cache(cache_path, meta, num_samples)
    if cached is not None:
        return cached

    computed = compute_fn()
    for key in ("sa", "div", "dds", "labels"):
        if key not in computed:
            raise ValueError(f"computed static scores missing key: {key}")
        if computed[key].shape != (num_samples,):
            raise ValueError(f"computed {key} shape mismatch: {computed[key].shape}")
    np.savez_compressed(
        cache_path,
        sa=computed["sa"],
        div=computed["div"],
        dds=computed["dds"],
        labels=computed["labels"],
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return computed


__all__ = ["get_or_compute_static_scores"]
