"""Learn seed-specific scoring weights from proxy training dynamics.

Cache-first behavior:
1. For each requested seed, first try loading cached ACT dynamic components under
   weights/dynamic_cache/[dataset]/[proxy_model]/[seed]/[epochs].
2. A cache file is usable as long as its final outputs are self-consistent and
   finite. NaN in raw_foldwise / fold_normalized is allowed because those arrays
   may deliberately use NaN as undefined fold-slot markers.
3. If all four caches are available for that seed, continue without requiring
   original proxy logs.
4. Otherwise, load proxy logs from
   weights/proxy_logs/[dataset]/[proxy_model]/[seed]/[epochs], recompute/save
   missing components, and continue.
5. Static SA/Div/DDS scores are computed with the adapter of the same output
   seed, then fitted against that seed's dynamic target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from model.adapter import AdapterMLP, load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.proxy_log_utils import resolve_proxy_log_path
from utils.seed import parse_seed_list, set_seed
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.static_score_cache import NORMALIZATION_VERSION, get_or_compute_static_scores
from weights import (
    AbsorptionGainScore,
    ConfusionComplementarityScore,
    TransferabilityAlignmentScore,
)
from weights.dynamic_utils import DynamicComponentResult, load_cv_fold_logs

PROJECT_ROOT = Path(__file__).resolve().parent
COMPONENT_NAMES = ("A", "C", "T")


def resolve_default_proxy_epochs(dataset_name: str) -> int:
    mapping = {
        CIFAR10: 200,
        CIFAR100: 200,
        TINY_IMAGENET: 90,
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset for default proxy epochs: {dataset_name}")
    return mapping[dataset_name]


def resolve_dynamic_component_cache_dir(dataset: str, proxy_model: str, seed: int, epochs: int) -> Path:
    return Path("weights") / "dynamic_cache" / dataset / proxy_model / str(int(seed)) / str(int(epochs))


def resolve_dynamic_component_cache_path(
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    component_name: str,
) -> Path:
    return resolve_dynamic_component_cache_dir(dataset, proxy_model, seed, epochs) / f"{component_name.strip().upper()}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn seed-specific scoring weights from proxy logs.")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root path.")
    parser.add_argument("--proxy-log", type=str, default="weights/proxy_logs", help="Proxy log root path or specific log dir.")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--adapter-image-path", type=str, default=None)
    parser.add_argument("--adapter-text-path", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument("--dds-k", type=int, default=5)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.8)
    parser.add_argument("--coverage-tau-g", type=float, default=0.15)
    parser.add_argument("--coverage-s-g", type=float, default=0.07)
    parser.add_argument("--coverage-k-pct", type=float, default=0.05)
    parser.add_argument("--coverage-q-low", type=float, default=0.002)
    parser.add_argument("--coverage-q-high", type=float, default=0.998)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--output", type=str, default="weights/scoring_weights.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help=(
            "seed 列表：每个 seed 分别加载该 seed 的 proxy CV 动态轨迹，"
            "并使用该 seed 的 adapter 静态分数拟合该 seed 的权重。"
        ),
    )
    parser.add_argument(
        "--proxy-training-seed",
        type=int,
        default=None,
        help=(
            "兼容旧命令的可选参数。默认不使用；此时每个输出 seed 同时作为 proxy 训练 seed。"
            "仅建议在单 seed 调试时显式指定。"
        ),
    )
    return parser.parse_args()


def _build_dataset(dataset_name: str, data_root: str, transform) -> datasets.VisionDataset:
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_score_loader(preprocess, data_root: str, dataset_name: str, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    dataset = _build_dataset(dataset_name, data_root, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_class_names(dataset_name: str, data_root: str) -> Iterable[str]:
    dataset = _build_dataset(dataset_name, data_root, transform=None)
    return resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=data_root,
        class_names=dataset.classes,  # type: ignore[attr-defined]
    )


def load_adapters_for_seed(args: argparse.Namespace, dataset_name: str, input_dim: int, seed: int, device: torch.device) -> tuple[AdapterMLP, AdapterMLP, dict[str, Path]]:
    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=dataset_name,
        clip_model=args.clip_model,
        input_dim=input_dim,
        seed=seed,
        map_location=device,
        adapter_image_path=args.adapter_image_path,
        adapter_text_path=args.adapter_text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()
    return image_adapter, text_adapter, adapter_paths

def _softmax_simplex(theta: np.ndarray) -> np.ndarray:
    """Map unconstrained parameters to strictly positive simplex weights."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a non-empty 1D array.")

    shifted = theta - float(np.max(theta))
    shifted = np.clip(shifted, -50.0, 50.0)
    exp_theta = np.exp(shifted)
    denom = float(np.sum(exp_theta))

    if (not np.isfinite(denom)) or denom <= 0.0:
        return np.full(theta.shape, 1.0 / theta.size, dtype=np.float64)

    weights = exp_theta / denom
    return weights.astype(np.float64)

def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be a non-empty 1D array.")
    sorted_vec = np.sort(vector)[::-1]
    cumulative_sum = np.cumsum(sorted_vec)
    rho_candidates = sorted_vec - (cumulative_sum - 1) / np.arange(1, vector.size + 1)
    rho_indices = np.where(rho_candidates > 0)[0]
    if rho_indices.size == 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)
    rho = rho_indices[-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    denom = projected.sum()
    if denom <= 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)
    return (projected / denom).astype(np.float64)


def fit_ridge_regression_nonnegative(
    features: np.ndarray,
    targets: np.ndarray,
    l2_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    """Fit positive simplex weights with ridge regularization.

    The returned weights satisfy:
      - weights_i > 0
      - sum_i weights_i = 1

    We optimize unconstrained theta and set weights = softmax(theta).
    The L2 term is applied to weights. Under the simplex constraint, this
    discourages overly concentrated weights and favors a more balanced mixture.
    """
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets shape mismatch.")
    if l2_lambda < 0:
        raise ValueError("l2_lambda must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    num_samples, num_features = features.shape
    features = features.astype(np.float64, copy=False)
    targets = targets.astype(np.float64, copy=False)

    theta = np.zeros(num_features, dtype=np.float64)
    weights = _softmax_simplex(theta)
    bias = float(np.mean(targets))

    for _ in tqdm(range(max_iter), desc="Fitting positive simplex weights", unit="iter", leave=False):
        preds = features @ weights + bias
        errors = preds - targets

        grad_w = (features.T @ errors) / num_samples + l2_lambda * weights
        grad_b = float(np.mean(errors))

        # Chain rule through softmax:
        # dL/dtheta_i = w_i * (dL/dw_i - sum_j w_j * dL/dw_j)
        grad_theta = weights * (grad_w - float(np.dot(weights, grad_w)))

        next_theta = theta - learning_rate * grad_theta
        next_bias = bias - learning_rate * grad_b
        next_weights = _softmax_simplex(next_theta)

        if np.linalg.norm(next_weights - weights) < tol and abs(next_bias - bias) < tol:
            theta = next_theta
            weights = next_weights
            bias = next_bias
            break

        theta = next_theta
        weights = next_weights
        bias = next_bias

    return weights.astype(np.float64), float(bias)

def build_output_path(base_path: str) -> Path:
    return Path(base_path)


def safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return 0.0
    xx = x[finite]
    yy = y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def save_dynamic_component_cache(
    *,
    cache_path: Path,
    component_name: str,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    proxy_log_path: Path,
    labels: np.ndarray,
    result: DynamicComponentResult,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels_i64 = labels.astype(np.int64, copy=False)
    np.savez_compressed(
        cache_path,
        component_name=np.array(component_name, dtype=np.str_),
        dataset=np.array(dataset, dtype=np.str_),
        proxy_model=np.array(proxy_model, dtype=np.str_),
        proxy_training_seed=np.array(proxy_training_seed, dtype=np.int64),
        seed_free=np.array(False),
        epochs=np.array(int(epochs), dtype=np.int64),
        proxy_log_path=np.array(str(proxy_log_path), dtype=np.str_),
        normalization=np.array(NORMALIZATION_VERSION, dtype=np.str_),
        num_samples=np.array(int(labels_i64.shape[0]), dtype=np.int64),
        labels=labels_i64,
        raw_foldwise=result.raw_foldwise.astype(np.float32),
        fold_normalized=result.fold_normalized.astype(np.float32),
        aggregated=result.aggregated.astype(np.float32),
        final_normalized=result.final_normalized.astype(np.float32),
    )


def load_dynamic_component_cache_if_valid(
    *,
    cache_path: Path,
    component_name: str,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    proxy_log_path: Path,
    labels_all: np.ndarray,
) -> DynamicComponentResult | None:
    if not cache_path.is_file():
        return None
    required_keys = {
        "component_name", "dataset", "proxy_model", "proxy_training_seed", "epochs",
        "proxy_log_path", "normalization", "num_samples", "labels", "raw_foldwise", "fold_normalized",
        "aggregated", "final_normalized",
    }
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if not required_keys.issubset(set(data.files)):
                return None
            if str(data["component_name"].item()) != component_name:
                return None
            if str(data["dataset"].item()) != dataset:
                return None
            if str(data["proxy_model"].item()) != proxy_model:
                return None
            if int(data["proxy_training_seed"].item()) != int(proxy_training_seed):
                return None
            if int(data["epochs"].item()) != int(epochs):
                return None
            if str(data["proxy_log_path"].item()) != str(proxy_log_path):
                return None
            if str(data["normalization"].item()) != NORMALIZATION_VERSION:
                return None
            labels_cached = np.asarray(data["labels"], dtype=np.int64)
            if not np.array_equal(labels_cached, labels_all.astype(np.int64, copy=False)):
                return None
            raw_foldwise = np.asarray(data["raw_foldwise"], dtype=np.float32)
            fold_normalized = np.asarray(data["fold_normalized"], dtype=np.float32)
            aggregated = np.asarray(data["aggregated"], dtype=np.float32)
            final_normalized = np.asarray(data["final_normalized"], dtype=np.float32)
    except Exception:
        return None

    num_samples = labels_all.shape[0]
    if raw_foldwise.ndim != 2 or fold_normalized.ndim != 2:
        return None
    if raw_foldwise.shape != fold_normalized.shape or raw_foldwise.shape[1] != num_samples:
        return None
    if aggregated.shape != (num_samples,) or final_normalized.shape != (num_samples,):
        return None
    if not np.all(np.isfinite(aggregated)) or not np.all(np.isfinite(final_normalized)):
        return None

    return DynamicComponentResult(
        raw_foldwise=raw_foldwise,
        fold_normalized=fold_normalized,
        aggregated=aggregated,
        final_normalized=final_normalized,
    )


def _load_dynamic_component_cache_with_reason(
    *,
    cache_path: Path,
    component_name: str,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
) -> tuple[DynamicComponentResult | None, np.ndarray | None, str]:
    """Load cache with minimal necessary checks and return concrete mismatch reason."""
    if not cache_path.is_file():
        return None, None, f"cache file not found: {cache_path}"

    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception as exc:
        return None, None, f"failed to open npz: {exc}"

    required_keys = {
        "component_name", "dataset", "proxy_model", "proxy_training_seed", "epochs",
        "normalization", "num_samples", "labels", "raw_foldwise", "fold_normalized", "aggregated", "final_normalized",
    }
    files = set(data.files)
    missing_keys = sorted(required_keys - files)
    if missing_keys:
        return None, None, f"missing keys: {missing_keys}"

    try:
        cache_component = str(data["component_name"].item())
        cache_dataset = str(data["dataset"].item())
        cache_proxy_model = str(data["proxy_model"].item())
        cache_proxy_training_seed = int(data["proxy_training_seed"].item())
        cache_epochs = int(data["epochs"].item())
        cache_normalization = str(data["normalization"].item())
        declared_num_samples = int(data["num_samples"].item())

        labels = np.asarray(data["labels"], dtype=np.int64)
        raw_foldwise = np.asarray(data["raw_foldwise"], dtype=np.float32)
        fold_normalized = np.asarray(data["fold_normalized"], dtype=np.float32)
        aggregated = np.asarray(data["aggregated"], dtype=np.float32)
        final_normalized = np.asarray(data["final_normalized"], dtype=np.float32)
    except Exception as exc:
        return None, None, f"failed to parse arrays/metadata: {exc}"

    if cache_component != component_name:
        return None, None, f"component_name mismatch: cache={cache_component}, expected={component_name}"
    if cache_dataset != dataset:
        return None, None, f"dataset mismatch: cache={cache_dataset}, expected={dataset}"
    if cache_proxy_model != proxy_model:
        return None, None, f"proxy_model mismatch: cache={cache_proxy_model}, expected={proxy_model}"
    if cache_proxy_training_seed != int(proxy_training_seed):
        return None, None, f"proxy_training_seed mismatch: cache={cache_proxy_training_seed}, expected={proxy_training_seed}"
    if cache_epochs != int(epochs):
        return None, None, f"epochs mismatch: cache={cache_epochs}, expected={epochs}"
    if cache_normalization != NORMALIZATION_VERSION:
        return None, None, f"normalization mismatch: cache={cache_normalization}, expected={NORMALIZATION_VERSION}"

    if labels.ndim != 1:
        return None, None, f"labels rank invalid: shape={labels.shape}"
    num_samples = labels.shape[0]
    if declared_num_samples != num_samples:
        return None, None, f"num_samples mismatch: declared={declared_num_samples}, actual_labels={num_samples}"

    if raw_foldwise.ndim != 2:
        return None, None, f"raw_foldwise rank invalid: shape={raw_foldwise.shape}"
    if fold_normalized.ndim != 2:
        return None, None, f"fold_normalized rank invalid: shape={fold_normalized.shape}"
    if raw_foldwise.shape != fold_normalized.shape:
        return None, None, f"raw/fold shape mismatch: raw={raw_foldwise.shape}, fold={fold_normalized.shape}"
    if raw_foldwise.shape[1] != num_samples:
        return None, None, f"raw_foldwise sample dimension mismatch: raw={raw_foldwise.shape}, labels={num_samples}"
    if aggregated.shape != (num_samples,):
        return None, None, f"aggregated shape mismatch: aggregated={aggregated.shape}, expected=({num_samples},)"
    if final_normalized.shape != (num_samples,):
        return None, None, f"final_normalized shape mismatch: final={final_normalized.shape}, expected=({num_samples},)"

    if not np.all(np.isfinite(aggregated)):
        return None, None, "aggregated contains NaN/inf"
    if not np.all(np.isfinite(final_normalized)):
        return None, None, "final_normalized contains NaN/inf"

    return (
        DynamicComponentResult(
            raw_foldwise=raw_foldwise,
            fold_normalized=fold_normalized,
            aggregated=aggregated,
            final_normalized=final_normalized,
        ),
        labels,
        "ok",
    )


def _try_load_all_dynamic_components_from_cache(
    *,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
) -> tuple[dict[str, DynamicComponentResult] | None, dict[str, Path], np.ndarray | None, dict[str, str]]:
    component_results: dict[str, DynamicComponentResult] = {}
    component_cache_paths: dict[str, Path] = {}
    component_reasons: dict[str, str] = {}
    labels_ref: np.ndarray | None = None

    for component_name in COMPONENT_NAMES:
        cache_path = resolve_dynamic_component_cache_path(dataset, proxy_model, proxy_training_seed, epochs, component_name)
        component_cache_paths[component_name] = cache_path
        result, labels, reason = _load_dynamic_component_cache_with_reason(
            cache_path=cache_path,
            component_name=component_name,
            dataset=dataset,
            proxy_model=proxy_model,
            proxy_training_seed=proxy_training_seed,
            epochs=epochs,
        )
        if result is None or labels is None:
            component_reasons[component_name] = reason
            return None, component_cache_paths, None, component_reasons

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            component_reasons[component_name] = "labels mismatch with previously loaded component caches"
            return None, component_cache_paths, None, component_reasons

        component_results[component_name] = result
        component_reasons[component_name] = "ok"

    return component_results, component_cache_paths, labels_ref, component_reasons


def get_or_compute_dynamic_component(
    *,
    component_name: str,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    proxy_log_path: Path,
    labels_all: np.ndarray,
    compute_fn: Callable[[], DynamicComponentResult],
) -> tuple[DynamicComponentResult, bool, Path]:
    cache_path = resolve_dynamic_component_cache_path(dataset, proxy_model, proxy_training_seed, epochs, component_name)
    cached = load_dynamic_component_cache_if_valid(
        cache_path=cache_path,
        component_name=component_name,
        dataset=dataset,
        proxy_model=proxy_model,
        proxy_training_seed=proxy_training_seed,
        epochs=epochs,
        proxy_log_path=proxy_log_path,
        labels_all=labels_all,
    )
    if cached is not None:
        return cached, True, cache_path

    computed = compute_fn()
    save_dynamic_component_cache(
        cache_path=cache_path,
        component_name=component_name,
        dataset=dataset,
        proxy_model=proxy_model,
        proxy_training_seed=proxy_training_seed,
        epochs=epochs,
        proxy_log_path=proxy_log_path,
        labels=labels_all,
        result=computed,
    )
    return computed, False, cache_path


def load_or_compute_dynamic_components_for_seed(
    args: argparse.Namespace,
    proxy_training_seed: int,
    resolved_proxy_epochs: int,
) -> tuple[dict[str, DynamicComponentResult], np.ndarray, Path | None, dict[str, Path]]:
    cached_bundle, component_cache_paths, labels_all, cache_reasons = _try_load_all_dynamic_components_from_cache(
        dataset=args.dataset,
        proxy_model=args.proxy_model,
        proxy_training_seed=proxy_training_seed,
        epochs=resolved_proxy_epochs,
    )

    if cached_bundle is not None and labels_all is not None:
        print(f"Dynamic component caches HIT for seed={proxy_training_seed}; proxy logs are not required.")
        for name in COMPONENT_NAMES:
            print(f"Dynamic component cache HIT: {name} @ {component_cache_paths[name]}")
        return cached_bundle, labels_all, None, component_cache_paths

    print(f"Dynamic cache set is not fully usable for seed={proxy_training_seed}. Reasons:")
    for name in COMPONENT_NAMES:
        print(f"  - {name}: {cache_reasons.get(name, 'not attempted')}")

    proxy_log = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        proxy_training_seed,
        proxy_model=args.proxy_model,
        max_epoch=resolved_proxy_epochs,
    )
    if not proxy_log.is_dir():
        reason_lines = "\n".join(
            f"  - {name}: {cache_reasons.get(name, 'unknown')}" for name in COMPONENT_NAMES
        )
        raise FileNotFoundError(
            "未能直接使用保存的动态分量缓存，且也找不到原始代理训练日志。\n"
            f"动态缓存目录: {resolve_dynamic_component_cache_dir(args.dataset, args.proxy_model, proxy_training_seed, resolved_proxy_epochs)}\n"
            f"原始代理日志期望路径: {proxy_log}\n"
            "缓存不匹配的具体原因如下：\n"
            f"{reason_lines}"
        )

    folds, labels_all = load_cv_fold_logs(proxy_log, args.dataset, args.data_root)
    actual_proxy_epochs = int(folds[0].train_logits.shape[0])
    if actual_proxy_epochs != resolved_proxy_epochs:
        print(
            f"INFO: requested proxy epochs={resolved_proxy_epochs}, "
            f"but loaded logs contain {actual_proxy_epochs} epochs; cache will use requested epochs tag."
        )

    component_compute_fns: dict[str, Callable[[], DynamicComponentResult]] = {
        "A": lambda: AbsorptionGainScore().compute(folds=folds, labels_all=labels_all),
        "C": lambda: ConfusionComplementarityScore().compute(folds=folds, labels_all=labels_all),
        "T": lambda: TransferabilityAlignmentScore().compute(folds=folds, labels_all=labels_all),
    }
    component_results: dict[str, DynamicComponentResult] = {}
    for name in COMPONENT_NAMES:
        result, from_cache, cache_path = get_or_compute_dynamic_component(
            component_name=name,
            dataset=args.dataset,
            proxy_model=args.proxy_model,
            proxy_training_seed=proxy_training_seed,
            epochs=resolved_proxy_epochs,
            proxy_log_path=proxy_log,
            labels_all=labels_all,
            compute_fn=component_compute_fns[name],
        )
        component_results[name] = result
        component_cache_paths[name] = cache_path
        print(f"Dynamic component cache {'HIT' if from_cache else 'MISS->RECOMPUTED'}: {name} @ {cache_path}")

    return component_results, labels_all, proxy_log, component_cache_paths


def build_dynamic_target(component_results: dict[str, DynamicComponentResult]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    a_result = component_results["A"]
    c_result = component_results["C"]
    t_result = component_results["T"]

    num_samples = a_result.final_normalized.shape[0]
    for name, arr in {
        "A_final_normalized": a_result.final_normalized,
        "C_final_normalized": c_result.final_normalized,
        "T_final_normalized": t_result.final_normalized,
    }.items():
        if arr.shape != (num_samples,):
            raise ValueError(f"{name} shape mismatch: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/inf values.")

    utility_raw = (
        a_result.final_normalized
        + c_result.final_normalized
        + t_result.final_normalized
    ).astype(np.float32) / 3.0
    u_scores = standard_zscore(utility_raw)
    dynamic_component_values = {
        "A": a_result.final_normalized,
        "C": c_result.final_normalized,
        "T": t_result.final_normalized,
    }
    return u_scores.astype(np.float64), dynamic_component_values


def run_once(args: argparse.Namespace, output_seeds: list[int]) -> None:
    if args.proxy_training_seed is not None and len(output_seeds) > 1:
        raise ValueError(
            "--proxy-training-seed 只允许在单 seed 调试时使用。"
            "多 seed 权重学习应直接使用 --seed 22,42,96，使每个 seed 对应自己的 proxy 动态轨迹。"
        )

    device = torch.device(args.device) if args.device else CONFIG.global_device
    resolved_proxy_epochs = int(args.proxy_epochs) if args.proxy_epochs is not None else resolve_default_proxy_epochs(args.dataset)

    class_names = load_class_names(args.dataset, args.data_root)
    dds_metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=args.dds_k,
        important_eigval_ratio=args.dds_important_eigval_ratio,
    )
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device, k=args.div_k)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=args.data_root,
        debug_prompts=args.debug_prompts,
    )

    dds_loader = build_score_loader(dds_metric.extractor.preprocess, args.data_root, args.dataset, device, args.batch_size, args.num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, args.data_root, args.dataset, device, args.batch_size, args.num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, args.data_root, args.dataset, device, args.batch_size, args.num_workers)

    dataset_for_labels = _build_dataset(args.dataset, args.data_root, transform=None)
    num_samples_static = len(dataset_for_labels)

    output_path = build_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, dict[str, object]]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded

    dataset_entry = data.get(args.dataset, {})

    for seed in output_seeds:
        proxy_training_seed = int(args.proxy_training_seed) if args.proxy_training_seed is not None else int(seed)
        set_seed(proxy_training_seed)
        print(
            f"\n=== Learning weights for output seed={seed}, "
            f"proxy_training_seed={proxy_training_seed}, proxy_epochs={resolved_proxy_epochs} ==="
        )

        component_results, labels_all, proxy_log, component_cache_paths = load_or_compute_dynamic_components_for_seed(
            args,
            proxy_training_seed=proxy_training_seed,
            resolved_proxy_epochs=resolved_proxy_epochs,
        )
        dynamic_scores, dynamic_component_values = build_dynamic_target(component_results)
        num_samples = labels_all.shape[0]

        image_adapter, text_adapter, adapter_paths = load_adapters_for_seed(
            args, args.dataset, dds_metric.extractor.embed_dim, seed, device
        )

        def _compute_scores() -> dict[str, np.ndarray]:
            dds_scores_local = dds_metric.score_dataset(
                tqdm(dds_loader, desc=f"Scoring DDS (seed={seed})", unit="batch"),
                adapter=image_adapter,
            )
            div_scores_local = div_metric.score_dataset(
                tqdm(div_loader, desc=f"Scoring Div (seed={seed})", unit="batch"),
                adapter=image_adapter,
            )
            sa_scores_local = sa_metric.score_dataset(
                tqdm(sa_loader, desc=f"Scoring SA (seed={seed})", unit="batch"),
                adapter_image=image_adapter,
                adapter_text=text_adapter,
            )
            return {
                "sa": sa_scores_local.scores.numpy(),
                "div": div_scores_local.scores.numpy(),
                "dds": dds_scores_local.scores.numpy(),
                "labels": np.asarray(dataset_for_labels.targets),
            }

        static_scores = get_or_compute_static_scores(
            cache_root=PROJECT_ROOT / "static_scores",
            dataset=args.dataset,
            seed=seed,
            clip_model=args.clip_model,
            adapter_image_path=str(adapter_paths["image_path"]),
            adapter_text_path=str(adapter_paths["text_path"]),
            div_k=div_metric.k,
            dds_k=dds_metric.k,
            dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
            dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
            prompt_template=sa_metric.prompt_template,
            num_samples=num_samples_static,
            compute_fn=_compute_scores,
        )
        if num_samples != num_samples_static:
            raise ValueError("Dynamic labels and static dataset sample count mismatch.")
        if not np.array_equal(labels_all, static_scores["labels"]):
            raise ValueError("动态监督标签与静态评分数据集标签不一致。")

        sa_z = standard_zscore_by_class(static_scores["sa"], static_scores["labels"])
        div_z = standard_zscore_by_class(static_scores["div"], static_scores["labels"])
        dds_z = standard_zscore_by_class(static_scores["dds"], static_scores["labels"])

        static_features = np.stack([sa_z, div_z, dds_z], axis=1).astype(np.float64)
        weights, bias = fit_ridge_regression_nonnegative(
            static_features,
            dynamic_scores,
            args.ridge_lambda,
            args.learning_rate,
            args.max_iter,
            args.tol,
        )

        static_component_values = {
            "SA": sa_z,
            "Div": div_z,
            "DDS": dds_z,
        }
        correlation_matrix: dict[str, dict[str, float]] = {}
        print(f"Dynamic-vs-static Pearson correlations (seed={seed}):")
        for d_name, d_values in dynamic_component_values.items():
            correlation_matrix[d_name] = {}
            row_str = []
            for s_name, s_values in static_component_values.items():
                corr = safe_pearson_corr(d_values, s_values)
                correlation_matrix[d_name][s_name] = corr
                row_str.append(f"{s_name}={corr:+.4f}")
            print(f"  {d_name}: " + ", ".join(row_str))

        print(
            f"Learned weights for seed {seed}: "
            f"SA={weights[0]:.6f}, Div={weights[1]:.6f}, DDS={weights[2]:.6f}, bias={bias:.6f}"
        )
        sa_warning = bool(weights[0] > (1.0 / 3.0))
        div_is_lowest = bool(weights[1] <= min(weights[0], weights[2]))
        if sa_warning:
            print(f"WARNING (seed={seed}): learned SA weight is above 1/3.")
        if div_is_lowest:
            print(f"WARNING (seed={seed}): learned Div weight is the lowest among SA/Div/DDS.")

        dynamic_cache_dir = resolve_dynamic_component_cache_dir(
            args.dataset,
            args.proxy_model,
            proxy_training_seed,
            resolved_proxy_epochs,
        )
        dataset_entry[str(seed)] = {
            "sa": float(weights[0]),
            "div": float(weights[1]),
            "dds": float(weights[2]),
            "bias": float(bias),
            "ridge_lambda": float(args.ridge_lambda),
            "proxy_log": str(proxy_log) if proxy_log is not None else "",
            "dynamic_components": list(COMPONENT_NAMES),
            "dynamic_cache_path": str(dynamic_cache_dir),
            "dynamic_component_cache_paths": {k: str(v) for k, v in component_cache_paths.items()},
            "proxy_training_seed": proxy_training_seed,
            "proxy_log_seed_free": False,
            "dynamic_cache_seed_free": False,
            "output_seed": seed,
            "adapter_seed": seed,
            "diagnostics": {
                "dynamic_static_pearson": correlation_matrix,
                "weight_solver": "positive_simplex_ridge_softmax",
                "weights_sum": float(np.sum(weights)),
                "weights_min": float(np.min(weights)),
                "sa_gt_one_third_warning": sa_warning,
                "div_is_lowest_warning": div_is_lowest,
            },
        }

    data[args.dataset] = dataset_entry
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Learned weights saved to", output_path)
    print("Applied output seeds:", output_seeds)


if __name__ == "__main__":
    args = parse_args()
    output_seeds = parse_seed_list(args.seed)
    if not output_seeds:
        raise ValueError("--seed 至少需要一个 seed。")
    run_once(args, output_seeds)
