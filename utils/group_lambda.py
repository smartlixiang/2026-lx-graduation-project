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
DEFAULT_MEAN_CAP_RATIO = 0.1

MEAN_BETA_BY_DATASET = {
    "cifar10": 0.03,
    "cifar100": 0.05,
}


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
    kr: int,
    weight_group: str,
) -> dict[str, float] | None:
    cache = _load_cache(path)
    kr_entry = cache.get(str(dataset), {}).get(str(seed), {}).get(str(kr), {})
    if isinstance(kr_entry, dict) and "lambda" in kr_entry:
        # backward compatibility for old cache layout without weight_group nesting
        record = kr_entry
    else:
        record = kr_entry.get(str(weight_group), {}) if isinstance(kr_entry, dict) else {}
    if not isinstance(record, dict):
        return None
    if "lambda_cls" in record:
        lam = record.get("lambda_cls")
    else:
        lam = record.get("lambda")
    try:
        lam_f = float(lam)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lam_f):
        return None
    # Group mode now depends on std-based statistics and applies scaling in
    # calculate_my_mask.py, so these two fields must be present in cache.
    required_fields = (
        "lambda_std_cls",
        "lambda_std_mean",
    )
    for field in required_fields:
        value = record.get(field)
        if value is None:
            return None
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value_f):
            return None
    return {k: float(v) for k, v in record.items() if isinstance(v, (int, float))}


def save_cached_lambda(
    path: Path,
    dataset: str,
    seed: int,
    kr: int,
    weight_group: str,
    record: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(path)
    ds_key, seed_key, kr_key = str(dataset), str(seed), str(kr)
    ds_entry = cache.setdefault(ds_key, {})
    if not isinstance(ds_entry, dict):
        ds_entry = {}
        cache[ds_key] = ds_entry
    seed_entry = ds_entry.setdefault(seed_key, {})
    if not isinstance(seed_entry, dict):
        seed_entry = {}
        ds_entry[seed_key] = seed_entry
    kr_entry = seed_entry.setdefault(kr_key, {})
    if not isinstance(kr_entry, dict):
        kr_entry = {}
        seed_entry[kr_key] = kr_entry
    kr_entry[str(weight_group)] = record

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
    kr: int,
    weight_group: str,
    n_samples: int,
    target_size: int,
    eval_score_fn: Callable[[np.ndarray], float],
    penalty_fn: Callable[[np.ndarray], float],
    mean_penalty_fn: Callable[[np.ndarray], float],
    M: int = DEFAULT_M,
    r: float = DEFAULT_R,
    eps: float = DEFAULT_EPS,
    c: float = DEFAULT_MEAN_CAP_RATIO,
    tqdm_desc: str = "Estimating lambda",
) -> dict[str, float]:
    cached = load_cached_lambda(cache_path, dataset, seed, kr, weight_group)
    if cached is not None:
        return cached

    s_values: list[float] = []
    pen_values: list[float] = []
    mean_pen_values: list[float] = []
    base_seed = int(seed)
    for sample_idx in tqdm(range(1, M + 1), desc=tqdm_desc, unit="sample", leave=False):
        sample_seed = base_seed * sample_idx
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(n_samples, size=target_size, replace=False)
        mask = np.zeros(n_samples, dtype=np.uint8)
        mask[idx] = 1
        s_values.append(float(eval_score_fn(mask)))
        pen_values.append(float(penalty_fn(mask)))
        mean_pen_values.append(float(mean_penalty_fn(mask)))

    s_arr = np.asarray(s_values, dtype=np.float64)
    pen_arr = np.asarray(pen_values, dtype=np.float64)
    mu_s = float(np.mean(s_arr))
    sigma_s = float(np.std(s_arr))
    mu_pen = float(np.mean(pen_arr))
    sigma_pen = float(np.std(pen_arr))
    h_arr = np.asarray(mean_pen_values, dtype=np.float64)
    mu_h = float(np.mean(h_arr))
    sigma_h = float(np.std(h_arr))

    lambda_std_cls = float(r * (sigma_s / (sigma_pen + eps)))
    lambda_mean_cls = float(r * (mu_s / (mu_pen + eps)))
    lambda_cls = float(min(lambda_std_cls, lambda_mean_cls))

    beta = float(MEAN_BETA_BY_DATASET.get(str(dataset).lower(), 0.03))
    lambda_std = float(beta * sigma_s / (sigma_h + eps))
    lambda_mean_base = float(beta * abs(mu_s) / (mu_h + eps))
    lambda_mean_cap = float(c * lambda_mean_base)
    # mean-based term is treated as an auxiliary regularizer and only provides
    # an upper cap; no lower-bound lifting is applied.
    lambda_mean = float(min(lambda_std, lambda_mean_cap))

    record: dict[str, float] = {
        "lambda": lambda_cls,
        "lambda_cls": lambda_cls,
        "lambda_mean": lambda_mean,
        "mu_S": mu_s,
        "sigma_S": sigma_s,
        "mu_pen": mu_pen,
        "sigma_pen": sigma_pen,
        "mu_mean_pen": mu_h,
        "sigma_mean_pen": sigma_h,
        "lambda_std_cls": lambda_std_cls,
        "lambda_mean_cls": lambda_mean_cls,
        "lambda_std_mean": lambda_std,
        "lambda_mean_base": lambda_mean_base,
        "lambda_mean_cap": lambda_mean_cap,
        "beta": beta,
        "c": float(c),
        "M": float(M),
        "r": float(r),
        "eps": float(eps),
    }
    save_cached_lambda(cache_path, dataset, seed, kr, weight_group, record)
    return record
