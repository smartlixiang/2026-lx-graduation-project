from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.seed import set_seed

COMPONENT_NAMES = ("A", "C", "T")
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read cached static scores and dynamic components, then test the modified "
            "linear regression solver without simplex constraint during fitting."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--normal-static-root", type=str, default="static_scores")
    parser.add_argument("--normal-dynamic-root", type=str, default="weights/dynamic_cache")
    parser.add_argument("--noise-static-root", type=str, default="noise_exp/static_scores")
    parser.add_argument("--noise-dynamic-root", type=str, default="noise_exp/weights/dynamic_cache")
    parser.add_argument("--ratio-lambda", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cpu")
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
) -> tuple[dict[str, np.ndarray], np.ndarray, Path]:
    cache_dir = Path(dynamic_root) / dataset_name / proxy_model / str(int(seed)) / str(int(epochs))
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Dynamic cache directory not found: {cache_dir}")

    components: dict[str, np.ndarray] = {}
    labels_ref: np.ndarray | None = None

    for name in COMPONENT_NAMES:
        path = require_file(cache_dir / f"{name}.npz")
        with np.load(path, allow_pickle=False) as data:
            for key in ("labels", "final_normalized"):
                if key not in data.files:
                    raise KeyError(f"{path} missing key: {key}")
            labels = np.asarray(data["labels"], dtype=np.int64)
            values = np.asarray(data["final_normalized"], dtype=np.float64)

        if labels.ndim != 1 or values.shape != labels.shape:
            raise ValueError(f"Invalid shape in {path}: labels={labels.shape}, final={values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} final_normalized contains NaN/inf.")

        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise ValueError(f"Dynamic component labels mismatch at {path}")

        components[name] = values

    assert labels_ref is not None
    return components, labels_ref, cache_dir


def load_noise_gate_cache(dynamic_cache_dir: Path, labels_all: np.ndarray) -> dict[str, np.ndarray]:
    path = require_file(dynamic_cache_dir / "noise_gate.npz")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "labels",
            "gate",
            "penalty",
            "dynamic_failure",
            "ever_learned",
            "forgetting_rate",
            "loss_difficulty",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")

        labels = np.asarray(data["labels"], dtype=np.int64)
        if not np.array_equal(labels, labels_all.astype(np.int64, copy=False)):
            raise ValueError(f"noise_gate labels mismatch: {path}")

        gate_data = {
            "gate": np.asarray(data["gate"], dtype=np.float64),
            "penalty": np.asarray(data["penalty"], dtype=np.float64),
            "dynamic_failure": np.asarray(data["dynamic_failure"], dtype=np.float64),
            "ever_learned": np.asarray(data["ever_learned"], dtype=np.float64),
            "forgetting_rate": np.asarray(data["forgetting_rate"], dtype=np.float64),
            "loss_difficulty": np.asarray(data["loss_difficulty"], dtype=np.float64),
        }

    n = labels_all.shape[0]
    for name, values in gate_data.items():
        if values.shape != (n,):
            raise ValueError(f"noise_gate {name} shape mismatch: {values.shape}, expected=({n},)")
        if name != "forgetting_rate" and not np.all(np.isfinite(values)):
            raise ValueError(f"noise_gate {name} contains NaN/inf.")
        if name == "forgetting_rate" and np.any(np.isinf(values[np.isfinite(values)])):
            raise ValueError("forgetting_rate contains inf.")

    if not np.all((gate_data["gate"] >= -1e-8) & (gate_data["gate"] <= 1.0 + 1e-8)):
        raise ValueError("noise_gate gate values are outside [0, 1].")

    gate_data["gate"] = np.clip(gate_data["gate"], 0.0, 1.0)
    return gate_data


def split_gate_positive_part(values: np.ndarray, gate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    return np.minimum(values, 0.0) + gate * np.maximum(values, 0.0)


def build_dynamic_target(components: dict[str, np.ndarray], *, gate: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(components["A"], dtype=np.float64)
    c = np.asarray(components["C"], dtype=np.float64)
    t = np.asarray(components["T"], dtype=np.float64)

    if gate is None:
        utility_raw = (a + c + t) / 3.0
    else:
        if gate.shape != a.shape:
            raise ValueError(f"gate shape mismatch: gate={gate.shape}, components={a.shape}")
        utility_raw = (
            split_gate_positive_part(a, gate)
            + split_gate_positive_part(c, gate)
            + split_gate_positive_part(t, gate)
        ) / 3.0

    return standard_zscore(utility_raw).astype(np.float64)


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


def fit_nonnegative_ratio_regularized_regression(
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
    for step in range(max_iter):
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
        "ratio_regularizer": ratio_value,
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


def run_case(
    *,
    title: str,
    dataset_name: str,
    seed: int,
    proxy_model: str,
    epochs: int,
    static_root: str | Path,
    dynamic_root: str | Path,
    use_noise_gate: bool,
    args: argparse.Namespace,
) -> None:
    print("=" * 90)
    print(f"[{title}] dataset={dataset_name}, seed={seed}, proxy_model={proxy_model}, epochs={epochs}")

    components, dynamic_labels, dynamic_cache_dir = load_dynamic_component_cache(
        dynamic_root, dataset_name, proxy_model, seed, epochs
    )
    static_scores, static_labels, static_cache_dir = load_static_scores(static_root, dataset_name, seed)

    if not np.array_equal(dynamic_labels, static_labels):
        raise ValueError(
            f"{title}: dynamic labels and static-score labels mismatch.\n"
            f"dynamic_cache={dynamic_cache_dir}\n"
            f"static_cache={static_cache_dir}"
        )

    if use_noise_gate:
        gate_data = load_noise_gate_cache(dynamic_cache_dir, dynamic_labels)
        target = build_dynamic_target(components, gate=gate_data["gate"])
        print(f"dynamic_cache: {dynamic_cache_dir}")
        print(f"static_cache:  {static_cache_dir}")
        print(
            "noise_gate:    "
            f"mean_gate={float(np.mean(gate_data['gate'])):.6f}, "
            f"mean_penalty={float(np.mean(gate_data['penalty'])):.6f}, "
            f"mean_failure={float(np.mean(gate_data['dynamic_failure'])):.6f}"
        )
    else:
        target = build_dynamic_target(components, gate=None)
        print(f"dynamic_cache: {dynamic_cache_dir}")
        print(f"static_cache:  {static_cache_dir}")

    features = build_static_features(static_scores, static_labels)
    result = fit_nonnegative_ratio_regularized_regression(
        features,
        target,
        ratio_lambda=args.ratio_lambda,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        tol=args.tol,
        device=args.device,
    )

    raw = np.asarray(result["raw_weights"], dtype=np.float64)
    norm = np.asarray(result["normalized_weights"], dtype=np.float64)
    pred = np.asarray(result["pred"], dtype=np.float64)

    print(
        "raw_weights:        "
        f"SA={raw[0]:.8f}, Div={raw[1]:.8f}, DDS={raw[2]:.8f}, "
        f"sum={float(np.sum(raw)):.8f}"
    )
    print(
        "normalized_weights: "
        f"SA={norm[0]:.8f}, Div={norm[1]:.8f}, DDS={norm[2]:.8f}, "
        f"sum={float(np.sum(norm)):.8f}"
    )
    print(
        "diagnostics:        "
        f"bias={float(result['bias']):.8f}, "
        f"mse={float(result['mse']):.8f}, "
        f"ratio_reg={float(result['ratio_regularizer']):.8f}, "
        f"corr={safe_corr(pred, target):+.6f}, "
        f"iters={int(result['iterations'])}"
    )
    print()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={args.device}, but torch.cuda.is_available() is False.")

    epochs = int(args.proxy_epochs) if args.proxy_epochs is not None else resolve_default_proxy_epochs(args.dataset)

    print("=== Modified linear regression diagnostic ===")
    print("Fitting rule: nonnegative weights, no simplex constraint during fitting.")
    print("Regularizer: Var(w) / Mean(w)^2. Final weights are normalized to sum to 1.")
    print(
        f"config: dataset={args.dataset}, seed={args.seed}, proxy_model={args.proxy_model}, "
        f"epochs={epochs}, ratio_lambda={args.ratio_lambda}, lr={args.learning_rate}, "
        f"max_iter={args.max_iter}, device={args.device}"
    )
    print()

    run_case(
        title="clean-data cached regression",
        dataset_name=args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        epochs=epochs,
        static_root=args.normal_static_root,
        dynamic_root=args.normal_dynamic_root,
        use_noise_gate=False,
        args=args,
    )

    run_case(
        title="noise-data cached regression",
        dataset_name=args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        epochs=epochs,
        static_root=args.noise_static_root,
        dynamic_root=args.noise_dynamic_root,
        use_noise_gate=True,
        args=args,
    )


if __name__ == "__main__":
    main()
