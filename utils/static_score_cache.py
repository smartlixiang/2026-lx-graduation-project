"""Utilities for caching static scores (SA/Div/DDS) with parameter-aware paths."""
from __future__ import annotations

import hashlib
import json
import re
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


def _format_percent(value: float) -> str:
    return f"{int(round(value * 100))}%"


def _format_div_k(div_k: float) -> str:
    if float(div_k).is_integer() and div_k >= 1:
        return str(int(div_k))
    if 0 < div_k < 1:
        return _format_percent(div_k)
    raise ValueError("Div k 仅支持 (0,1) 百分比或 >=1 的正整数。")


def _build_param_dir_name(div_k: float, dds_lower: float, dds_upper: float) -> str:
    return f"Div_{_format_div_k(div_k)}_DDS_[{_format_percent(dds_lower)}-{_format_percent(dds_upper)}]"


def _parse_param_dir_name(name: str) -> dict[str, str] | None:
    pattern = re.compile(r"^Div_(?P<div>\d+%?|\d+)_DDS_\[(?P<low>\d+%)-(?P<high>\d+%)\]$")
    match = pattern.match(name)
    if not match:
        return None
    return match.groupdict()


def _build_cache_dir(
    cache_root: Path,
    dataset: str,
    seed: int,
    div_k: float,
    dds_lower: float,
    dds_upper: float,
) -> Path:
    param_dir = _build_param_dir_name(div_k, dds_lower, dds_upper)
    cache_dir = cache_root / _sanitize(dataset) / str(int(seed)) / param_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _score_file(cache_dir: Path, metric_name: str) -> Path:
    return cache_dir / f"{metric_name}_cache.npz"


def _validate_metric_cache(
    cache_dir: Path,
    metric_name: str,
    expected_meta: dict[str, object],
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    score_path = _score_file(cache_dir, metric_name)
    if not score_path.exists():
        return None

    parsed = _parse_param_dir_name(cache_dir.name)
    if parsed is None:
        return None
    if parsed["div"] != _format_div_k(float(expected_meta["div_k"])):
        return None
    if parsed["low"] != _format_percent(float(expected_meta["dds_eigval_lower_bound"])):
        return None
    if parsed["high"] != _format_percent(float(expected_meta["dds_eigval_upper_bound"])):
        return None

    data = np.load(score_path, allow_pickle=False)
    if "meta" not in data:
        return None
    try:
        cached_meta = json.loads(str(data["meta"]))
    except json.JSONDecodeError:
        return None

    for key, value in expected_meta.items():
        if cached_meta.get(key) != value:
            return None

    if "scores" not in data or "labels" not in data or "indices" not in data:
        return None
    scores = np.asarray(data["scores"])
    labels = np.asarray(data["labels"])
    indices = np.asarray(data["indices"])

    if scores.shape != (num_samples,) or labels.shape != (num_samples,) or indices.shape != (num_samples,):
        return None
    if not np.array_equal(indices, np.arange(num_samples, dtype=indices.dtype)):
        return None
    return scores, labels, indices


def get_or_compute_static_scores(
    *,
    cache_root: str | Path,
    dataset: str,
    seed: int,
    clip_model: str,
    adapter_image_path: str | None,
    adapter_text_path: str | None,
    div_k: float,
    dds_k: int,
    dds_eigval_lower_bound: float,
    dds_eigval_upper_bound: float,
    prompt_template: str,
    num_samples: int,
    compute_fn: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    cache_root = Path(cache_root)
    cache_dir = _build_cache_dir(
        cache_root,
        dataset,
        seed,
        div_k,
        dds_eigval_lower_bound,
        dds_eigval_upper_bound,
    )

    meta = {
        "dataset": dataset,
        "seed": int(seed),
        "clip_model": clip_model,
        "adapter_image_path": adapter_image_path or "",
        "adapter_text_path": adapter_text_path or "",
        "adapter_image_sha1": _hash_file(Path(adapter_image_path)) if adapter_image_path else "",
        "adapter_text_sha1": _hash_file(Path(adapter_text_path)) if adapter_text_path else "",
        "div_k": float(div_k),
        "dds_k": int(dds_k),
        "dds_eigval_lower_bound": float(dds_eigval_lower_bound),
        "dds_eigval_upper_bound": float(dds_eigval_upper_bound),
        "prompt_template": prompt_template,
        "num_samples": int(num_samples),
    }

    cached: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None
    for metric_name, metric_key in (("SA", "sa"), ("Div", "div"), ("DDS", "dds")):
        validated = _validate_metric_cache(cache_dir, metric_name, meta, num_samples)
        if validated is None:
            cached = {}
            break
        scores, labels, _ = validated
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            cached = {}
            break
        cached[metric_key] = scores
    if cached and labels_ref is not None:
        cached["labels"] = labels_ref
        return cached

    computed = compute_fn()
    for key in ("sa", "div", "dds", "labels"):
        if key not in computed:
            raise ValueError(f"computed static scores missing key: {key}")
        if np.asarray(computed[key]).shape != (num_samples,):
            raise ValueError(f"computed {key} shape mismatch: {np.asarray(computed[key]).shape}")

    indices = np.arange(num_samples, dtype=np.int64)
    metric_mapping = {
        "SA": np.asarray(computed["sa"]),
        "Div": np.asarray(computed["div"]),
        "DDS": np.asarray(computed["dds"]),
    }
    labels = np.asarray(computed["labels"])

    for metric_name, scores in metric_mapping.items():
        np.savez_compressed(
            _score_file(cache_dir, metric_name),
            scores=scores,
            labels=labels,
            indices=indices,
            meta=json.dumps(meta, ensure_ascii=False),
        )

    return {
        "sa": metric_mapping["SA"],
        "div": metric_mapping["Div"],
        "dds": metric_mapping["DDS"],
        "labels": labels,
    }


__all__ = ["get_or_compute_static_scores"]
