from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.seed import set_seed
from weights.dynamic_utils import load_cv_fold_logs

# 直接复用正式噪声实验脚本中的新版门控定义，避免诊断脚本与正式脚本分叉。
from learn_scoring_weights import (  # noqa: E402
    NOISE_GATE_CACHE_VERSION,
    _build_gate_from_final_risk,
    _compute_final_noise_risk,
    _load_noise_gate_cache_if_valid,
    _save_noise_gate_cache,
)

COMPONENT_NAMES = ("A", "C", "T")
EPS = 1e-8


@dataclass
class ComponentArrays:
    raw_foldwise: np.ndarray
    final_normalized: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean/noise CIFAR-100 static-weight learning under the new noise gate. "
            "A/C/T are gated after final normalization instead of Gate-2 raw-component gating."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)

    parser.add_argument("--normal-static-root", type=str, default="static_scores")
    parser.add_argument("--normal-dynamic-root", type=str, default="weights/dynamic_cache")
    parser.add_argument("--normal-proxy-root", type=str, default="weights/proxy_logs")

    parser.add_argument("--noise-static-root", type=str, default="noise_exp/static_scores")
    parser.add_argument("--noise-dynamic-root", type=str, default="noise_exp/weights/dynamic_cache")
    parser.add_argument("--noise-proxy-root", type=str, default="noise_exp/weights/proxy_logs")

    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--gate-low", type=float, default=0.1)
    parser.add_argument("--gate-high", type=float, default=0.9)

    # 与 learn_scoring_weights.py 中的 softmax-simplex ridge 拟合保持一致。
    parser.add_argument("--simplex-ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--simplex-learning-rate", type=float, default=1e-2)
    parser.add_argument("--simplex-max-iter", type=int, default=10000)
    parser.add_argument("--simplex-tol", type=float, default=1e-6)

    # 原版 test_regression.py 中的 softplus 非 simplex + ratio regularizer 拟合。
    parser.add_argument("--ratio-lambda", type=float, default=5e-3)
    parser.add_argument("--ratio-learning-rate", type=float, default=2e-3)
    parser.add_argument("--ratio-max-iter", type=int, default=10000)
    parser.add_argument("--ratio-tol", type=float, default=1e-8)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--save-clean-gate-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only clean-mode gate cache may be saved. No other cache/result is written.",
    )
    parser.add_argument(
        "--overwrite-clean-gate-cache",
        action="store_true",
        help="Allow overwriting an existing clean noise_gate.npz cache. Noise-mode cache is never overwritten.",
    )
    return parser.parse_args()


def resolve_default_proxy_epochs(dataset_name: str) -> int:
    if dataset_name in {CIFAR10, CIFAR100}:
        return 200
    if dataset_name == TINY_IMAGENET:
        return 90
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required cache file not found: {path}")
    return path


def load_dynamic_component_cache(
    dynamic_root: str | Path,
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> tuple[dict[str, ComponentArrays], np.ndarray, Path]:
    cache_dir = Path(dynamic_root) / dataset_name / proxy_model / str(int(seed)) / str(int(epochs))
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Dynamic cache directory not found: {cache_dir}")

    components: dict[str, ComponentArrays] = {}
    labels_ref: np.ndarray | None = None

    for name in COMPONENT_NAMES:
        path = require_file(cache_dir / f"{name}.npz")
        with np.load(path, allow_pickle=False) as data:
            for key in ("labels", "raw_foldwise", "final_normalized"):
                if key not in data.files:
                    raise KeyError(f"{path} missing key: {key}")
            labels = np.asarray(data["labels"], dtype=np.int64)
            raw_foldwise = np.asarray(data["raw_foldwise"], dtype=np.float32)
            final_normalized = np.asarray(data["final_normalized"], dtype=np.float64)

        if labels.ndim != 1 or final_normalized.shape != labels.shape:
            raise ValueError(f"Invalid final shape in {path}: labels={labels.shape}, final={final_normalized.shape}")
        if raw_foldwise.ndim != 2 or raw_foldwise.shape[1] != labels.shape[0]:
            raise ValueError(f"Invalid raw_foldwise shape in {path}: raw={raw_foldwise.shape}, labels={labels.shape}")
        if not np.all(np.isfinite(final_normalized)):
            raise ValueError(f"{path} final_normalized contains NaN/inf.")

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(f"Dynamic component labels mismatch at {path}")

        components[name] = ComponentArrays(raw_foldwise=raw_foldwise, final_normalized=final_normalized)

    assert labels_ref is not None
    return components, labels_ref, cache_dir


def load_static_cache_dir(static_root: str | Path, dataset_name: str, seed: int) -> Path:
    root = Path(static_root) / dataset_name / str(int(seed))
    if not root.is_dir():
        raise FileNotFoundError(f"Static cache root not found: {root}")

    candidates: list[Path] = []
    for cache_dir in sorted(root.rglob("*")):
        if not cache_dir.is_dir():
            continue
        required_files = [cache_dir / f"{name}_cache.npz" for name in ("SA", "Div", "DDS")]
        if all(path.is_file() for path in required_files):
            candidates.append(cache_dir)

    if not candidates:
        raise FileNotFoundError(f"No complete SA/Div/DDS static cache found under: {root}")

    candidates.sort(
        key=lambda p: max((p / f"{name}_cache.npz").stat().st_mtime for name in ("SA", "Div", "DDS")),
        reverse=True,
    )
    return candidates[0]


def load_static_scores(static_root: str | Path, dataset_name: str, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, Path]:
    cache_dir = load_static_cache_dir(static_root, dataset_name, seed)

    scores: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None
    for metric_name, key in (("SA", "sa"), ("Div", "div"), ("DDS", "dds")):
        path = require_file(cache_dir / f"{metric_name}_cache.npz")
        with np.load(path, allow_pickle=False) as data:
            for npz_key in ("scores", "labels", "indices"):
                if npz_key not in data.files:
                    raise KeyError(f"{path} missing key: {npz_key}")
            values = np.asarray(data["scores"], dtype=np.float64)
            labels = np.asarray(data["labels"], dtype=np.int64)
            indices = np.asarray(data["indices"])

        if values.shape != labels.shape or values.ndim != 1:
            raise ValueError(f"Invalid static score shape in {path}: scores={values.shape}, labels={labels.shape}")
        if not np.array_equal(indices, np.arange(values.shape[0], dtype=indices.dtype)):
            raise ValueError(f"Invalid indices in {path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} scores contain NaN/inf.")

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(f"Static labels mismatch at {path}")

        scores[key] = values

    assert labels_ref is not None
    return scores, labels_ref, cache_dir


def build_static_features(static_scores: dict[str, np.ndarray], labels: np.ndarray) -> np.ndarray:
    sa_z = standard_zscore_by_class(static_scores["sa"], labels)
    div_z = standard_zscore_by_class(static_scores["div"], labels)
    dds_z = standard_zscore_by_class(static_scores["dds"], labels)
    return np.stack([sa_z, div_z, dds_z], axis=1).astype(np.float64)


def split_gate_positive_part(values: np.ndarray, gate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    if values.shape != gate.shape:
        raise ValueError(f"values/gate shape mismatch: values={values.shape}, gate={gate.shape}")
    return np.minimum(values, 0.0) + gate * np.maximum(values, 0.0)


def build_normalized_gate_dynamic_target(components: dict[str, ComponentArrays], gate: np.ndarray) -> np.ndarray:
    """Build target by gating positive parts after each dynamic component is normalized.

    This deliberately restores the lighter normalized-space gate: A/C/T all use
    final_normalized values, and only their positive parts are multiplied by gate.
    It is used here to test whether Gate-2 raw-component gating is too strong.
    """
    gate = np.asarray(gate, dtype=np.float64)
    gated_parts = []
    for name in COMPONENT_NAMES:
        values = components[name].final_normalized.astype(np.float64)
        if values.shape != gate.shape:
            raise ValueError(f"{name}/gate shape mismatch: {name}={values.shape}, gate={gate.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}.final_normalized contains NaN/inf values.")
        gated_parts.append(split_gate_positive_part(values, gate))

    utility_raw = (gated_parts[0] + gated_parts[1] + gated_parts[2]) / 3.0
    return standard_zscore(utility_raw).astype(np.float64)


def compute_or_load_gate(
    *,
    case_name: str,
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    labels: np.ndarray,
    dynamic_cache_dir: Path,
    proxy_root: str | Path,
    data_root: str | Path,
    learn_window: int,
    learn_min_correct: int,
    gate_low: float,
    gate_high: float,
    save_clean_gate_cache: bool,
    overwrite_clean_gate_cache: bool,
) -> tuple[np.ndarray, np.ndarray, Path]:
    cache_path = dynamic_cache_dir / "noise_gate.npz"
    cached = _load_noise_gate_cache_if_valid(
        cache_path,
        dataset_name,
        proxy_model,
        seed,
        epochs,
        labels,
        learn_window,
        learn_min_correct,
        gate_low,
        gate_high,
    )
    if cached is not None:
        final_risk = np.asarray(cached["final_risk"], dtype=np.float64)
        gate = np.asarray(cached["gate"], dtype=np.float64)
        print(f"[{case_name}] gate cache HIT: {cache_path}")
        return final_risk, gate, cache_path

    proxy_log_path = Path(proxy_root) / dataset_name / proxy_model / str(int(seed)) / str(int(epochs))
    folds, _ = load_cv_fold_logs(proxy_log_path, dataset_name, str(data_root))
    final_risk = _compute_final_noise_risk(folds, labels, learn_window, learn_min_correct)
    gate = _build_gate_from_final_risk(final_risk, gate_low, gate_high)
    print(f"[{case_name}] gate cache MISS -> computed from proxy logs: {proxy_log_path}")

    if case_name == "clean" and save_clean_gate_cache:
        if cache_path.exists() and not overwrite_clean_gate_cache:
            print(f"[clean] existing invalid/outdated gate cache is kept unchanged: {cache_path}")
        else:
            _save_noise_gate_cache(
                cache_path,
                dataset_name,
                proxy_model,
                seed,
                epochs,
                labels,
                learn_window,
                learn_min_correct,
                final_risk,
            )
            print(f"[clean] saved gate cache: {cache_path}")
    elif case_name != "clean":
        print(f"[{case_name}] gate cache is not saved by this diagnostic script.")

    return final_risk, gate, cache_path


def softmax_simplex(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    shifted = theta - float(np.max(theta))
    shifted = np.clip(shifted, -50.0, 50.0)
    exp_theta = np.exp(shifted)
    denom = float(np.sum(exp_theta))
    if (not np.isfinite(denom)) or denom <= 0.0:
        return np.full(theta.shape, 1.0 / theta.size, dtype=np.float64)
    return (exp_theta / denom).astype(np.float64)


def fit_simplex_ridge_regression(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f"features must have shape (N,3), got {features.shape}")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError(f"target shape mismatch: features={features.shape}, targets={targets.shape}")
    if l2_lambda < 0:
        raise ValueError("l2_lambda must be non-negative.")

    x = features.astype(np.float64, copy=False)
    y = targets.astype(np.float64, copy=False)
    n, d = x.shape
    theta = np.zeros(d, dtype=np.float64)
    weights = softmax_simplex(theta)
    bias = float(np.mean(y))

    final_iter = 0
    for step in tqdm(range(max_iter), desc="Fitting simplex ridge", unit="iter", leave=False):
        pred = x @ weights + bias
        errors = pred - y
        grad_w = (x.T @ errors) / n + l2_lambda * weights
        grad_b = float(np.mean(errors))
        grad_theta = weights * (grad_w - float(np.dot(weights, grad_w)))

        next_theta = theta - learning_rate * grad_theta
        next_bias = bias - learning_rate * grad_b
        next_weights = softmax_simplex(next_theta)
        final_iter = step + 1

        if np.linalg.norm(next_weights - weights) < tol and abs(next_bias - bias) < tol:
            theta = next_theta
            weights = next_weights
            bias = next_bias
            break

        theta = next_theta
        weights = next_weights
        bias = next_bias

    pred = x @ weights + bias
    mse = float(np.mean((pred - y) ** 2))
    return {
        "raw_weights": weights.astype(np.float64),
        "normalized_weights": weights.astype(np.float64),
        "bias": float(bias),
        "mse": mse,
        "regularizer": float(np.sum(weights * weights)),
        "iterations": final_iter,
        "pred": pred.astype(np.float64),
    }


def fit_softplus_ratio_regularized_regression(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ratio_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
    device: str,
) -> dict[str, object]:
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError(f"features must have shape (N,3), got {features.shape}")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError(f"target shape mismatch: features={features.shape}, targets={targets.shape}")
    if ratio_lambda < 0:
        raise ValueError("ratio_lambda must be non-negative.")

    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = torch.as_tensor(targets, dtype=torch.float64, device=device)
    theta = torch.zeros(3, dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.tensor(float(np.mean(targets)), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta, bias], lr=learning_rate)

    last_loss = None
    final_iter = 0
    for step in tqdm(range(max_iter), desc="Fitting softplus ratio", unit="iter", leave=False):
        optimizer.zero_grad()
        raw_weights = torch.nn.functional.softplus(theta) + 1e-8
        pred = x @ raw_weights + bias
        mse = torch.mean((pred - y) ** 2)
        mean_w = torch.mean(raw_weights)
        ratio_reg = torch.var(raw_weights, unbiased=False) / (mean_w * mean_w + EPS)
        loss = mse + float(ratio_lambda) * ratio_reg
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu())
        final_iter = step + 1
        if last_loss is not None and abs(last_loss - current_loss) < tol:
            break
        last_loss = current_loss

    with torch.no_grad():
        raw_weights_t = torch.nn.functional.softplus(theta) + 1e-8
        raw_weights = raw_weights_t.detach().cpu().numpy().astype(np.float64)
        weight_sum = float(np.sum(raw_weights))
        normalized_weights = raw_weights / weight_sum if weight_sum > 0 else np.full(3, 1.0 / 3.0)
        pred = (x @ raw_weights_t + bias).detach().cpu().numpy().astype(np.float64)
        mse_value = float(np.mean((pred - targets) ** 2))
        ratio_value = float(np.var(raw_weights) / (np.mean(raw_weights) ** 2 + EPS))
        bias_value = float(bias.detach().cpu())

    return {
        "raw_weights": raw_weights,
        "normalized_weights": normalized_weights,
        "bias": bias_value,
        "mse": mse_value,
        "regularizer": ratio_value,
        "iterations": final_iter,
        "pred": pred,
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan")
    xx = x[finite]
    yy = y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def print_weight_result(case_name: str, solver_name: str, result: dict[str, object], target: np.ndarray) -> None:
    raw = np.asarray(result["raw_weights"], dtype=np.float64)
    norm = np.asarray(result["normalized_weights"], dtype=np.float64)
    pred = np.asarray(result["pred"], dtype=np.float64)
    print(f"\n[{case_name} | {solver_name}]")
    print(
        "  raw_weights:        "
        f"SA={raw[0]:.8f}, Div={raw[1]:.8f}, DDS={raw[2]:.8f}, sum={float(np.sum(raw)):.8f}"
    )
    print(
        "  normalized_weights: "
        f"SA={norm[0]:.8f}, Div={norm[1]:.8f}, DDS={norm[2]:.8f}, sum={float(np.sum(norm)):.8f}"
    )
    print(
        "  diagnostics:        "
        f"bias={float(result['bias']):.8f}, mse={float(result['mse']):.8f}, "
        f"reg={float(result['regularizer']):.8f}, corr={safe_corr(pred, target):+.6f}, "
        f"iters={int(result['iterations'])}"
    )


def run_case(
    *,
    case_name: str,
    dataset_name: str,
    seed: int,
    proxy_model: str,
    epochs: int,
    static_root: str | Path,
    dynamic_root: str | Path,
    proxy_root: str | Path,
    args: argparse.Namespace,
) -> None:
    print("=" * 100)
    print(f"[{case_name}] dataset={dataset_name}, seed={seed}, proxy_model={proxy_model}, epochs={epochs}")

    components, dynamic_labels, dynamic_cache_dir = load_dynamic_component_cache(dynamic_root, dataset_name, proxy_model, seed, epochs)
    static_scores, static_labels, static_cache_dir = load_static_scores(static_root, dataset_name, seed)
    if not np.array_equal(dynamic_labels, static_labels):
        raise ValueError(
            f"{case_name}: dynamic labels and static-score labels mismatch.\n"
            f"dynamic_cache={dynamic_cache_dir}\nstatic_cache={static_cache_dir}"
        )

    final_risk, gate, gate_cache_path = compute_or_load_gate(
        case_name=case_name,
        dataset_name=dataset_name,
        proxy_model=proxy_model,
        seed=seed,
        epochs=epochs,
        labels=dynamic_labels,
        dynamic_cache_dir=dynamic_cache_dir,
        proxy_root=proxy_root,
        data_root=args.data_root,
        learn_window=args.learn_window,
        learn_min_correct=args.learn_min_correct,
        gate_low=args.gate_low,
        gate_high=args.gate_high,
        save_clean_gate_cache=args.save_clean_gate_cache,
        overwrite_clean_gate_cache=args.overwrite_clean_gate_cache,
    )

    target = build_normalized_gate_dynamic_target(components, gate)
    features = build_static_features(static_scores, static_labels)

    print(f"dynamic_cache: {dynamic_cache_dir}")
    print(f"static_cache:  {static_cache_dir}")
    print(f"gate_cache:    {gate_cache_path}")
    print(
        "gate summary:  "
        f"mean_gate={float(np.mean(gate)):.6f}, min_gate={float(np.min(gate)):.6f}, "
        f"max_gate={float(np.max(gate)):.6f}, mean_final_risk={float(np.mean(final_risk)):.6f}"
    )
    print(
        "target summary: "
        f"mean={float(np.mean(target)):.6f}, std={float(np.std(target)):.6f}, "
        f"min={float(np.min(target)):.6f}, max={float(np.max(target)):.6f}"
    )

    simplex_result = fit_simplex_ridge_regression(
        features,
        target,
        l2_lambda=args.simplex_ridge_lambda,
        learning_rate=args.simplex_learning_rate,
        max_iter=args.simplex_max_iter,
        tol=args.simplex_tol,
    )
    print_weight_result(case_name, "simplex-ridge(sum=1)", simplex_result, target)

    ratio_result = fit_softplus_ratio_regularized_regression(
        features,
        target,
        ratio_lambda=args.ratio_lambda,
        learning_rate=args.ratio_learning_rate,
        max_iter=args.ratio_max_iter,
        tol=args.ratio_tol,
        device=args.device,
    )
    print_weight_result(case_name, "softplus-ratio(no simplex)", ratio_result, target)
    print()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    epochs = int(args.proxy_epochs) if args.proxy_epochs is not None else resolve_default_proxy_epochs(args.dataset)

    print("=== test_regression_2: clean/noise weight-learning comparison with normalized-space gate ===")
    print(f"dataset={args.dataset}, seed={args.seed}, proxy_model={args.proxy_model}, proxy_epochs={epochs}")
    print(f"noise_gate_cache_version={NOISE_GATE_CACHE_VERSION}")
    print(f"gate: learn_window={args.learn_window}, learn_min_correct={args.learn_min_correct}, low={args.gate_low}, high={args.gate_high}")
    print("This script never writes static/dynamic/weight results. It may only save clean gate cache when enabled. A/C/T are gated after final normalization.")

    run_case(
        case_name="clean",
        dataset_name=args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        epochs=epochs,
        static_root=args.normal_static_root,
        dynamic_root=args.normal_dynamic_root,
        proxy_root=args.normal_proxy_root,
        args=args,
    )

    run_case(
        case_name="noise",
        dataset_name=args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        epochs=epochs,
        static_root=args.noise_static_root,
        dynamic_root=args.noise_dynamic_root,
        proxy_root=args.noise_proxy_root,
        args=args,
    )


if __name__ == "__main__":
    main()
