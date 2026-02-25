from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm


DEFAULT_M = 100
DEFAULT_R = 0.05
DEFAULT_EPS = 1e-8


def compute_balance_penalty(
    selected_mask: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    target_size: int,
) -> float:
    selected = selected_mask.astype(bool)
    counts = np.bincount(labels[selected], minlength=num_classes).astype(np.float64)
    ideal = float(target_size) / float(num_classes) if num_classes > 0 else 0.0
    return float(np.abs(counts - ideal).sum())


def _load_cache(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_cached_lambda(
    path: Path,
    dataset: str,
    seed: int,
    cr: int,
    weight_group: str,
) -> dict[str, float] | None:
    cache = _load_cache(path)
    cr_entry = cache.get(str(dataset), {}).get(str(seed), {}).get(str(cr), {})
    if isinstance(cr_entry, dict) and "lambda" in cr_entry:
        # backward compatibility for old cache layout without weight_group nesting
        record = cr_entry
    else:
        record = cr_entry.get(str(weight_group), {}) if isinstance(cr_entry, dict) else {}
    if not isinstance(record, dict):
        return None
    lam = record.get("lambda")
    try:
        lam_f = float(lam)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lam_f):
        return None
    return {k: float(v) for k, v in record.items() if isinstance(v, (int, float))}


def save_cached_lambda(
    path: Path,
    dataset: str,
    seed: int,
    cr: int,
    weight_group: str,
    record: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(path)
    ds_key, seed_key, cr_key = str(dataset), str(seed), str(cr)
    ds_entry = cache.setdefault(ds_key, {})
    if not isinstance(ds_entry, dict):
        ds_entry = {}
        cache[ds_key] = ds_entry
    seed_entry = ds_entry.setdefault(seed_key, {})
    if not isinstance(seed_entry, dict):
        seed_entry = {}
        ds_entry[seed_key] = seed_entry
    cr_entry = seed_entry.setdefault(cr_key, {})
    if not isinstance(cr_entry, dict):
        cr_entry = {}
        seed_entry[cr_key] = cr_entry
    cr_entry[str(weight_group)] = record

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def get_or_estimate_lambda(
    *,
    cache_path: Path,
    dataset: str,
    seed: int,
    cr: int,
    weight_group: str,
    n_samples: int,
    target_size: int,
    eval_score_fn: Callable[[np.ndarray], float],
    penalty_fn: Callable[[np.ndarray], float],
    M: int = DEFAULT_M,
    r: float = DEFAULT_R,
    eps: float = DEFAULT_EPS,
    tqdm_desc: str = "Estimating lambda",
) -> dict[str, float]:
    cached = load_cached_lambda(cache_path, dataset, seed, cr, weight_group)
    if cached is not None:
        return cached

    s_values: list[float] = []
    pen_values: list[float] = []
    for _ in tqdm(range(M), desc=tqdm_desc, unit="sample", leave=False):
        idx = np.random.choice(n_samples, size=target_size, replace=False)
        mask = np.zeros(n_samples, dtype=np.uint8)
        mask[idx] = 1
        s_values.append(float(eval_score_fn(mask)))
        pen_values.append(float(penalty_fn(mask)))

    s_arr = np.asarray(s_values, dtype=np.float64)
    pen_arr = np.asarray(pen_values, dtype=np.float64)
    mu_s = float(np.mean(s_arr))
    sigma_s = float(np.std(s_arr))
    mu_pen = float(np.mean(pen_arr))
    sigma_pen = float(np.std(pen_arr))
    lambda_std = float(r * (sigma_s / (sigma_pen + eps)))
    lambda_mean = float(r * (mu_s / (mu_pen + eps)))
    lam = float(min(lambda_std, lambda_mean))

    record: dict[str, float] = {
        "lambda": lam,
        "mu_S": mu_s,
        "sigma_S": sigma_s,
        "mu_pen": mu_pen,
        "sigma_pen": sigma_pen,
        "M": float(M),
        "r": float(r),
        "eps": float(eps),
    }
    save_cached_lambda(cache_path, dataset, seed, cr, weight_group, record)
    return record
