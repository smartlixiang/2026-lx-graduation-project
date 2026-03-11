from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm


DEFAULT_M = 100
DEFAULT_EPS = 1e-8
DEFAULT_MEAN_LAMBDA_BASE = 0.02
DEFAULT_CLS_LAMBDA_BASE = 0.001


def compute_balance_penalty(
    selected_mask: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    target_size: int,
) -> float:
    selected = selected_mask.astype(bool)
    counts = np.bincount(labels[selected], minlength=num_classes).astype(np.float64)
    ideal = float(target_size) / float(num_classes) if num_classes > 0 else 0.0
    if num_classes <= 0:
        return 0.0
    return float(np.abs(counts - ideal).sum() / float(num_classes))


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
    if isinstance(kr_entry, dict) and "lambda_cls" in kr_entry:
        # backward compatibility for old cache layout without weight_group nesting
        record = kr_entry
    else:
        record = kr_entry.get(str(weight_group), {}) if isinstance(kr_entry, dict) else {}
    if not isinstance(record, dict):
        return None
    required_fields = (
        "lambda_cls",
        "lambda_mean",
        "raw_mean",
        "class_penalty_mean",
        "mean_penalty_mean",
        "ratio_cls_target",
        "ratio_mean_target",
        "M",
        "eps",
        "cls_lambda_base",
        "mean_lambda_base",
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


def _target_ratios(
    kr: int,
    *,
    cls_lambda_base: float,
    mean_lambda_base: float,
) -> tuple[float, float]:
    """Return target correction ratios calibrated from random baseline.

    The ratio is anchored at kr=20 (max) and linearly decays to 0 at kr=100:
    r(kr)=(100-kr)/80.
    """
    decay = max(0.0, min(1.0, (100.0 - float(kr)) / 80.0))
    return float(cls_lambda_base) * decay, float(mean_lambda_base) * decay


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
    eps: float = DEFAULT_EPS,
    cls_lambda_base: float = DEFAULT_CLS_LAMBDA_BASE,
    mean_lambda_base: float = DEFAULT_MEAN_LAMBDA_BASE,
    tqdm_desc: str = "Estimating lambda",
) -> dict[str, float]:
    cached = load_cached_lambda(cache_path, dataset, seed, kr, weight_group)
    if cached is not None:
        cached_cls_base = float(cached.get("cls_lambda_base", float("nan")))
        cached_mean_base = float(cached.get("mean_lambda_base", float("nan")))
        if np.isfinite(cached_cls_base) and np.isfinite(cached_mean_base):
            if abs(cached_cls_base - float(cls_lambda_base)) < 1e-12 and abs(cached_mean_base - float(mean_lambda_base)) < 1e-12:
                return cached
        # base ratios changed, force re-estimation and overwrite cache.

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

    raw_mean = float(np.mean(np.asarray(s_values, dtype=np.float64)))
    class_penalty_mean = float(np.mean(np.asarray(pen_values, dtype=np.float64)))
    mean_penalty_mean = float(np.mean(np.asarray(mean_pen_values, dtype=np.float64)))
    ratio_cls_target, ratio_mean_target = _target_ratios(
        kr,
        cls_lambda_base=cls_lambda_base,
        mean_lambda_base=mean_lambda_base,
    )
    lambda_cls = float((ratio_cls_target * raw_mean) / (class_penalty_mean + eps))
    lambda_mean = float((ratio_mean_target * raw_mean) / (mean_penalty_mean + eps))

    record: dict[str, float] = {
        "lambda_cls": lambda_cls,
        "lambda_mean": lambda_mean,
        "raw_mean": raw_mean,
        "class_penalty_mean": class_penalty_mean,
        "mean_penalty_mean": mean_penalty_mean,
        "ratio_cls_target": float(ratio_cls_target),
        "ratio_mean_target": float(ratio_mean_target),
        "M": float(M),
        "eps": float(eps),
        "cls_lambda_base": float(cls_lambda_base),
        "mean_lambda_base": float(mean_lambda_base),
    }
    save_cached_lambda(cache_path, dataset, seed, kr, weight_group, record)
    print(
        f"[LambdaBaseline] dataset={dataset} | seed={seed} | kr={kr} "
        f"| raw_mean={raw_mean:.8f} | class_penalty_mean={class_penalty_mean:.8f} "
        f"| mean_penalty_mean={mean_penalty_mean:.8f} | lambda_cls={lambda_cls:.8f} "
        f"| lambda_mean={lambda_mean:.8f} | cls_lambda_base={float(cls_lambda_base):.6f} "
        f"| mean_lambda_base={float(mean_lambda_base):.6f}"
    )
    return record
