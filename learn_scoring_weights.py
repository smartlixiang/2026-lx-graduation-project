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
from weights.dynamic_utils import DynamicComponentResult, FoldLogData, load_cv_fold_logs, resolve_epoch_windows

PROJECT_ROOT = Path(__file__).resolve().parent
COMPONENT_NAMES = ("A", "C", "T")
NOISE_GATE_CACHE_VERSION = "noise_gate_v4_learn_after_val_margin"
METHOD_VERSION = "softplus_ratio_noise_gate_v1"


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
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--gate-low", type=float, default=0.2)
    parser.add_argument("--gate-high", type=float, default=0.95)
    parser.add_argument("--ratio-lambda", type=float, default=1e-2)
    parser.add_argument("--regression-learning-rate", type=float, default=1e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-8)
    parser.add_argument(
        "--use-noise-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use the rank-min dynamic gate when building the regression target.",
    )
    parser.add_argument(
        "--force-noise-gate",
        action="store_true",
        help="Recompute noise_gate.npz even if a valid cache exists.",
    )
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


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, samples, classes).")
    if labels.shape != (logits.shape[1],):
        raise ValueError("labels shape mismatch for logits.")
    if np.any(labels < 0) or np.any(labels >= logits.shape[2]):
        raise ValueError("labels out of range for logits.")

    max_logits = np.max(logits, axis=2)
    shifted = logits - max_logits[:, :, None]
    logsumexp = np.log(np.sum(np.exp(shifted), axis=2)) + max_logits
    true_logits = np.take_along_axis(logits, labels.reshape(1, -1, 1), axis=2).squeeze(2)
    losses = logsumexp - true_logits
    return np.nan_to_num(losses, nan=0.0, posinf=1.0e6, neginf=0.0).astype(np.float64)


def _percentile_rank_tie_aware(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains NaN/inf values.")
    n = int(values.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (0.5 * float(i + j - 1)) / float(n - 1)
        i = j
    return ranks.astype(np.float64)


def _validate_gate_thresholds(gate_low: float, gate_high: float) -> tuple[float, float]:
    gate_low = float(gate_low)
    gate_high = float(gate_high)
    if not (0.0 <= gate_low < gate_high <= 1.0):
        raise ValueError("gate_low and gate_high must satisfy 0 <= gate_low < gate_high <= 1.")
    return gate_low, gate_high


def _build_gate_from_final_risk(final_risk: np.ndarray, gate_low: float, gate_high: float) -> np.ndarray:
    gate_low, gate_high = _validate_gate_thresholds(gate_low, gate_high)
    final_risk = np.asarray(final_risk, dtype=np.float64)
    if final_risk.ndim != 1 or not np.all(np.isfinite(final_risk)):
        raise ValueError("final_risk must be a finite one-dimensional array.")
    gate = np.where(
        final_risk <= gate_low,
        1.0,
        np.where(final_risk >= gate_high, 0.0, 1.0 - (final_risk - gate_low) / (gate_high - gate_low)),
    )
    return np.clip(gate, 0.0, 1.0).astype(np.float64)


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.sum(exp, axis=-1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return (exp / denom).astype(np.float64)


def _compute_final_noise_risk(
    folds: list[FoldLogData],
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
) -> np.ndarray:
    labels_all = np.asarray(labels_all, dtype=np.int64)
    num_samples = int(labels_all.shape[0])
    learn_window = int(learn_window)
    learn_min_correct = int(learn_min_correct)
    if learn_window <= 0:
        raise ValueError("learn_window must be positive.")
    if learn_min_correct <= 0 or learn_min_correct > learn_window:
        raise ValueError("learn_min_correct must be in [1, learn_window].")

    learn_sum = np.zeros(num_samples, dtype=np.float64)
    learn_count = np.zeros(num_samples, dtype=np.int64)
    loss_var_sum = np.zeros(num_samples, dtype=np.float64)
    loss_var_count = np.zeros(num_samples, dtype=np.int64)
    val_bias_raw = np.full(num_samples, np.nan, dtype=np.float64)

    for fold in tqdm(folds, desc="compute noise gate", unit="fold", leave=False):
        train_idx = np.asarray(fold.train_indices, dtype=np.int64)
        y_train = labels_all[train_idx]
        train_logits = np.asarray(fold.train_logits, dtype=np.float64)
        train_preds = np.argmax(train_logits, axis=2)
        correct = train_preds == y_train.reshape(1, -1)
        num_epochs = int(correct.shape[0])
        for local_i, sample_idx in enumerate(train_idx):
            seq = correct[:, local_i].astype(np.int64)
            learned_time = None
            if num_epochs >= learn_window:
                window_sum = np.convolve(seq, np.ones(learn_window, dtype=np.int64), mode="valid")
                hit = np.flatnonzero(window_sum >= learn_min_correct)
                if hit.size:
                    learned_time = int(hit[0] + learn_window - 1)
            if learned_time is None:
                learn_time_norm = 1.0
                forget_frequency = 1.0
                learn_risk = 1.0
            else:
                learn_time_norm = float(learned_time - learn_window + 1) / float(max(1, num_epochs - learn_window))
                learn_time_norm = float(np.clip(learn_time_norm, 0.0, 1.0))
                remaining = correct[learned_time + 1 :, local_i]
                if remaining.size == 0:
                    forget_frequency = 1.0
                else:
                    forget_frequency = float(np.mean(~remaining))
                learn_risk = 1.0 - (1.0 - learn_time_norm) * (1.0 - forget_frequency)
                learn_risk = float(np.clip(learn_risk, 0.0, 1.0))
            learn_sum[int(sample_idx)] += float(learn_risk)
            learn_count[int(sample_idx)] += 1

        losses = _cross_entropy_from_logits(train_logits, y_train)
        _, mid_idx, late_idx = resolve_epoch_windows(num_epochs)
        mid_late_idx = np.concatenate([mid_idx, late_idx]).astype(np.int64)
        loss_var_sum[train_idx] += np.var(losses[mid_late_idx], axis=0).astype(np.float64)
        loss_var_count[train_idx] += 1

        val_idx = np.asarray(fold.val_indices, dtype=np.int64)
        y_val = labels_all[val_idx]
        val_logits = np.asarray(fold.val_logits, dtype=np.float64)
        _, val_mid_idx, val_late_idx = resolve_epoch_windows(int(val_logits.shape[0]))
        val_mid_late_idx = np.concatenate([val_mid_idx, val_late_idx]).astype(np.int64)
        probs = _softmax_logits(val_logits[val_mid_late_idx])
        for local_i, sample_idx in enumerate(val_idx):
            label = int(y_val[local_i])
            probs_i = probs[:, local_i, :]
            label_probs = probs_i[:, label]
            positive_margins = np.maximum(probs_i - label_probs[:, None], 0.0)
            positive_margins[:, label] = 0.0
            class_scores = np.mean(positive_margins, axis=0)
            class_scores[label] = 0.0
            val_bias_raw[int(sample_idx)] = float(np.max(class_scores))

    if np.any(learn_count <= 0) or np.any(loss_var_count <= 0) or np.any(~np.isfinite(val_bias_raw)):
        missing = np.where((learn_count <= 0) | (loss_var_count <= 0) | (~np.isfinite(val_bias_raw)))[0]
        raise ValueError(f"Some samples lack noise gate metrics: {missing[:10]}.")

    learn_risk_rank = _percentile_rank_tie_aware(learn_sum / learn_count)
    loss_var_rank = _percentile_rank_tie_aware(loss_var_sum / loss_var_count)
    val_bias_rank = _percentile_rank_tie_aware(val_bias_raw)
    return np.minimum.reduce([learn_risk_rank, loss_var_rank, val_bias_rank]).astype(np.float64)


def _noise_gate_cache_path(dataset: str, proxy_model: str, proxy_training_seed: int, epochs: int) -> Path:
    return resolve_dynamic_component_cache_dir(dataset, proxy_model, proxy_training_seed, epochs) / "noise_gate.npz"


def _load_noise_gate_cache_if_valid(
    cache_path: Path,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
    gate_low: float,
    gate_high: float,
) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(np.asarray(data["cache_version"]).item()) != NOISE_GATE_CACHE_VERSION:
                return None
            if str(np.asarray(data["dataset"]).item()) != str(dataset):
                return None
            if str(np.asarray(data["proxy_model"]).item()) != str(proxy_model):
                return None
            if int(np.asarray(data["proxy_training_seed"]).item()) != int(proxy_training_seed):
                return None
            if int(np.asarray(data["epochs"]).item()) != int(epochs):
                return None
            if int(np.asarray(data["learn_window"]).item()) != int(learn_window):
                return None
            if int(np.asarray(data["learn_min_correct"]).item()) != int(learn_min_correct):
                return None
            labels_cached = np.asarray(data["labels"], dtype=np.int64)
            if not np.array_equal(labels_cached, labels_all.astype(np.int64, copy=False)):
                return None
            final_risk = np.asarray(data["final_risk"], dtype=np.float64)
    except Exception:
        return None
    if final_risk.shape != (int(labels_all.shape[0]),) or not np.all(np.isfinite(final_risk)):
        return None
    return {"final_risk": final_risk, "gate": _build_gate_from_final_risk(final_risk, gate_low, gate_high)}


def _save_noise_gate_cache(
    cache_path: Path,
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
    final_risk: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.array(NOISE_GATE_CACHE_VERSION, dtype=np.str_),
        dataset=np.array(dataset, dtype=np.str_),
        proxy_model=np.array(proxy_model, dtype=np.str_),
        proxy_training_seed=np.array(int(proxy_training_seed), dtype=np.int64),
        epochs=np.array(int(epochs), dtype=np.int64),
        labels=np.asarray(labels_all, dtype=np.int64),
        learn_window=np.array(int(learn_window), dtype=np.int64),
        learn_min_correct=np.array(int(learn_min_correct), dtype=np.int64),
        final_risk=np.asarray(final_risk, dtype=np.float64),
    )


def get_or_compute_noise_gate(
    *,
    folds: list[FoldLogData],
    dataset: str,
    proxy_model: str,
    proxy_training_seed: int,
    epochs: int,
    labels_all: np.ndarray,
    learn_window: int,
    learn_min_correct: int,
    gate_low: float,
    gate_high: float,
    force: bool = False,
) -> dict[str, np.ndarray]:
    _validate_gate_thresholds(gate_low, gate_high)
    cache_path = _noise_gate_cache_path(dataset, proxy_model, proxy_training_seed, epochs)
    if not force:
        cached = _load_noise_gate_cache_if_valid(
            cache_path, dataset, proxy_model, proxy_training_seed, epochs,
            labels_all, learn_window, learn_min_correct, gate_low, gate_high,
        )
        if cached is not None:
            print(f"Noise gate cache HIT: {cache_path}")
            return cached
    final_risk = _compute_final_noise_risk(folds, labels_all, learn_window, learn_min_correct)
    _save_noise_gate_cache(cache_path, dataset, proxy_model, proxy_training_seed, epochs, labels_all, learn_window, learn_min_correct, final_risk)
    print(f"Noise gate cache {'FORCE->RECOMPUTED' if force else 'MISS->COMPUTED'}: {cache_path}")
    return {"final_risk": final_risk, "gate": _build_gate_from_final_risk(final_risk, gate_low, gate_high)}


def fit_softplus_ratio_regression(
    features: np.ndarray,
    targets: np.ndarray,
    ratio_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
    device: torch.device,
) -> dict[str, object]:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets shape mismatch.")
    features_t = torch.as_tensor(features, dtype=torch.float64, device=device)
    targets_t = torch.as_tensor(targets, dtype=torch.float64, device=device)
    theta = torch.zeros(features.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.tensor(float(np.mean(targets)), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta, bias], lr=float(learning_rate))
    prev_loss: float | None = None
    iterations = 0
    final_mse = 0.0
    final_ratio = 0.0
    for iteration in tqdm(range(int(max_iter)), desc="fit softplus-ratio", unit="iter", leave=False):
        optimizer.zero_grad()
        raw_weights = torch.nn.functional.softplus(theta) + 1e-8
        pred = features_t @ raw_weights + bias
        mse = torch.mean((pred - targets_t) ** 2)
        ratio_reg = torch.var(raw_weights, unbiased=False) / (torch.mean(raw_weights) ** 2 + 1e-8)
        loss = mse + float(ratio_lambda) * ratio_reg
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        final_mse = float(mse.detach().cpu())
        final_ratio = float(ratio_reg.detach().cpu())
        iterations = iteration + 1
        if prev_loss is not None and abs(prev_loss - loss_value) < float(tol):
            break
        prev_loss = loss_value

    with torch.no_grad():
        raw_weights_t = torch.nn.functional.softplus(theta) + 1e-8
        pred_t = features_t @ raw_weights_t + bias
        raw_weights_np = raw_weights_t.detach().cpu().numpy().astype(np.float64)
        normalized_weights = raw_weights_np / float(np.sum(raw_weights_np))
        pred_np = pred_t.detach().cpu().numpy().astype(np.float64)
    return {
        "raw_weights": raw_weights_np,
        "normalized_weights": normalized_weights.astype(np.float64),
        "bias": float(bias.detach().cpu()),
        "mse": final_mse,
        "ratio_regularizer": final_ratio,
        "iterations": iterations,
        "pred": pred_np,
    }

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
            continue

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            component_reasons[component_name] = "labels mismatch with previously loaded component caches"
            continue

        component_results[component_name] = result
        component_reasons[component_name] = "ok"

    if len(component_results) != len(COMPONENT_NAMES):
        return None, component_cache_paths, labels_ref, component_reasons

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



def validate_saved_weight_entry(entry: object) -> tuple[bool, dict[str, float] | None, str]:
    if not isinstance(entry, dict):
        return False, None, "entry is not a dict"
    values: dict[str, float] = {}
    for key in ("sa", "div", "dds"):
        if key not in entry:
            return False, None, f"missing required field: {key}"
        try:
            value = float(entry[key])
        except (TypeError, ValueError):
            return False, None, f"field {key} is not convertible to float"
        if not np.isfinite(value):
            return False, None, f"field {key} is not finite"
        if value <= 0.0:
            return False, None, f"field {key} is not > 0"
        values[key] = value
    weight_sum = values["sa"] + values["div"] + values["dds"]
    if abs(weight_sum - 1.0) > 1e-4:
        return False, None, f"weights sum is not approximately 1: sum={weight_sum:.10f}"
    return True, values, "ok"


def report_existing_weight_entry(
    data: dict,
    dataset: str,
    seed: int,
    output_path: Path,
) -> None:
    if not output_path.exists():
        print(f"[weights] no existing scoring_weights file: {output_path}")
        return
    dataset_entry = data.get(dataset)
    if not isinstance(dataset_entry, dict):
        print(f"[weights] no existing dataset entry: dataset={dataset}")
        return
    seed_entry = dataset_entry.get(str(seed))
    if seed_entry is None:
        print(f"[weights] no existing weights for dataset={dataset} seed={seed}")
        return
    valid, weights, reason = validate_saved_weight_entry(seed_entry)
    if valid and weights is not None:
        weight_sum = weights["sa"] + weights["div"] + weights["dds"]
        print(
            "[weights] existing weights are valid and will be overwritten: "
            f"dataset={dataset} seed={seed} "
            f"SA={weights['sa']:.6f}, Div={weights['div']:.6f}, "
            f"DDS={weights['dds']:.6f}, sum={weight_sum:.6f}"
        )
    else:
        print(
            "[weights] existing weights are invalid and will be overwritten: "
            f"dataset={dataset} seed={seed} reason={reason}"
        )

def load_proxy_folds_for_dynamic_seed(
    args: argparse.Namespace,
    proxy_training_seed: int,
    resolved_proxy_epochs: int,
    *,
    reason: str,
) -> tuple[list[FoldLogData], np.ndarray, Path]:
    proxy_log = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        proxy_training_seed,
        proxy_model=args.proxy_model,
        max_epoch=resolved_proxy_epochs,
    )
    if not proxy_log.is_dir():
        raise FileNotFoundError(
            f"{reason}\n"
            f"原始代理日志期望路径: {proxy_log}\n"
            "请先生成对应 proxy logs，或提供完整且匹配当前参数的动态缓存。"
        )

    folds, labels_all = load_cv_fold_logs(proxy_log, args.dataset, args.data_root)
    actual_proxy_epochs = int(folds[0].train_logits.shape[0])
    if actual_proxy_epochs != resolved_proxy_epochs:
        print(
            f"INFO: requested proxy epochs={resolved_proxy_epochs}, "
            f"but loaded logs contain {actual_proxy_epochs} epochs; cache will use requested epochs tag."
        )
    return folds, np.asarray(labels_all, dtype=np.int64), proxy_log


def load_or_compute_dynamic_supervision_for_seed(
    args: argparse.Namespace,
    proxy_training_seed: int,
    resolved_proxy_epochs: int,
) -> tuple[
    dict[str, DynamicComponentResult],
    dict[str, np.ndarray] | None,
    np.ndarray,
    Path | None,
    dict[str, Path],
    dict[str, str],
]:
    cached_bundle, component_cache_paths, labels_all, cache_reasons = _try_load_all_dynamic_components_from_cache(
        dataset=args.dataset,
        proxy_model=args.proxy_model,
        proxy_training_seed=proxy_training_seed,
        epochs=resolved_proxy_epochs,
    )

    reasons: dict[str, str] = {name: cache_reasons.get(name, "not attempted") for name in COMPONENT_NAMES}
    gate_cache_path = _noise_gate_cache_path(args.dataset, args.proxy_model, proxy_training_seed, resolved_proxy_epochs)
    noise_gate_data: dict[str, np.ndarray] | None = None
    if args.use_noise_gate:
        if labels_all is not None and not args.force_noise_gate:
            noise_gate_data = _load_noise_gate_cache_if_valid(
                gate_cache_path,
                args.dataset,
                args.proxy_model,
                proxy_training_seed,
                resolved_proxy_epochs,
                labels_all,
                args.learn_window,
                args.learn_min_correct,
                args.gate_low,
                args.gate_high,
            )
            reasons["gate"] = "ok" if noise_gate_data is not None else f"missing/invalid gate cache: {gate_cache_path}"
        elif labels_all is not None and args.force_noise_gate:
            reasons["gate"] = "forced recompute by --force-noise-gate"
        else:
            reasons["gate"] = "cannot validate gate before labels are loaded because A/C/T caches are incomplete"
    else:
        reasons["gate"] = "disabled by --no-use-noise-gate"

    print(f"Dynamic supervision cache status for seed={proxy_training_seed}")
    for name in (*COMPONENT_NAMES, "gate"):
        print(f"  - {name}: {reasons.get(name, 'not attempted')}")

    all_components_ok = cached_bundle is not None and labels_all is not None
    gate_ok = (not args.use_noise_gate) or (noise_gate_data is not None)
    if all_components_ok and gate_ok:
        print(f"Dynamic supervision caches HIT for seed={proxy_training_seed}; proxy logs are not required.")
        for name in COMPONENT_NAMES:
            print(f"Dynamic component cache HIT: {name} @ {component_cache_paths[name]}")
        if noise_gate_data is not None:
            print(f"Noise gate cache HIT: {gate_cache_path}")
        return cached_bundle, noise_gate_data, labels_all, None, component_cache_paths, reasons

    reason_lines = "\n".join(f"  - {name}: {reasons.get(name, 'unknown')}" for name in (*COMPONENT_NAMES, "gate"))
    folds, labels_all, proxy_log = load_proxy_folds_for_dynamic_seed(
        args,
        proxy_training_seed,
        resolved_proxy_epochs,
        reason=(
            "未能直接使用完整动态伪标签缓存，需要加载原始代理训练日志来补算未命中的分量。\n"
            f"动态缓存目录: {resolve_dynamic_component_cache_dir(args.dataset, args.proxy_model, proxy_training_seed, resolved_proxy_epochs)}\n"
            "缓存不匹配的具体原因如下：\n"
            f"{reason_lines}"
        ),
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
        reasons[name] = "ok" if from_cache else "recomputed and saved"
        print(f"Dynamic component cache {'HIT' if from_cache else 'MISS->RECOMPUTED'}: {name} @ {cache_path}")

    if args.use_noise_gate:
        noise_gate_data = get_or_compute_noise_gate(
            folds=folds,
            dataset=args.dataset,
            proxy_model=args.proxy_model,
            proxy_training_seed=proxy_training_seed,
            epochs=resolved_proxy_epochs,
            labels_all=labels_all,
            learn_window=args.learn_window,
            learn_min_correct=args.learn_min_correct,
            gate_low=args.gate_low,
            gate_high=args.gate_high,
            force=bool(args.force_noise_gate),
        )
        reasons["gate"] = "ok" if reasons.get("gate") == "ok" and not args.force_noise_gate else "recomputed and saved"

    return component_results, noise_gate_data, labels_all, proxy_log, component_cache_paths, reasons

def build_dynamic_target(
    component_results: dict[str, DynamicComponentResult],
    gate: np.ndarray | None = None,
    use_noise_gate: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

    if use_noise_gate and gate is not None:
        gate = np.asarray(gate, dtype=np.float64)
        if gate.shape != (num_samples,) or not np.all(np.isfinite(gate)):
            raise ValueError(f"gate shape/finite mismatch: {gate.shape}")

        def split_gate_positive_part(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            return np.minimum(values, 0.0) + gate * np.maximum(values, 0.0)

        gated_a = split_gate_positive_part(a_result.final_normalized)
        gated_c = split_gate_positive_part(c_result.final_normalized)
        gated_t = split_gate_positive_part(t_result.final_normalized)
        utility_raw = (gated_a + gated_c + gated_t) / 3.0
    else:
        utility_raw = (
            a_result.final_normalized
            + c_result.final_normalized
            + t_result.final_normalized
        ).astype(np.float64) / 3.0
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

        component_results, noise_gate_data, labels_all, proxy_log, component_cache_paths, cache_reasons = load_or_compute_dynamic_supervision_for_seed(
            args,
            proxy_training_seed=proxy_training_seed,
            resolved_proxy_epochs=resolved_proxy_epochs,
        )
        dynamic_scores, dynamic_component_values = build_dynamic_target(
            component_results,
            gate=None if noise_gate_data is None else noise_gate_data["gate"],
            use_noise_gate=bool(args.use_noise_gate),
        )
        if noise_gate_data is not None:
            dynamic_component_values["noise_gate"] = noise_gate_data["gate"]
            dynamic_component_values["noise_final_risk"] = noise_gate_data["final_risk"]
        num_samples = labels_all.shape[0]

        report_existing_weight_entry(data, args.dataset, seed, output_path)

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
        fit_result = fit_softplus_ratio_regression(
            static_features,
            dynamic_scores,
            args.ratio_lambda,
            args.regression_learning_rate,
            args.regression_max_iter,
            args.regression_tol,
            device,
        )
        weights = np.asarray(fit_result["normalized_weights"], dtype=np.float64)
        bias = float(fit_result["bias"])

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
            "method_version": METHOD_VERSION,
            "ratio_lambda": float(args.ratio_lambda),
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
                "weight_solver": "softplus_ratio_no_simplex",
                "method_version": METHOD_VERSION,
                "raw_weights": [float(x) for x in np.asarray(fit_result["raw_weights"], dtype=np.float64)],
                "mse": float(fit_result["mse"]),
                "ratio_regularizer": float(fit_result["ratio_regularizer"]),
                "iterations": int(fit_result["iterations"]),
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
