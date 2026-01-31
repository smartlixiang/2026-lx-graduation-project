"""Learn scoring weights from proxy training dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100
from model.adapter import AdapterMLP
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.global_config import CONFIG
from utils.seed import parse_seed_list, set_seed
from weights import CoverageGainScore, EarlyLearnabilityScore, MarginScore, StabilityScore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn scoring weights from proxy logs.")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root path.",
    )
    parser.add_argument(
        "--proxy-log",
        type=str,
        default="weights/proxy_logs/cifar10_resnet18_2026_01_12_14_31.npz",
        help="Path to proxy training log (.npz).",
    )
    parser.add_argument("--adapter-path", type=str, default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt", help="Optional adapter path.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=int, default=10)
    parser.add_argument("--dds-k", type=float, default=10)
    parser.add_argument("--early-epochs", type=int, default=None)
    parser.add_argument("--margin-delta", type=float, default=1.0)
    parser.add_argument("--coverage-tau-g", type=float, default=0.15)
    parser.add_argument("--coverage-s-g", type=float, default=0.07)
    parser.add_argument("--coverage-k", type=int, default=10)
    parser.add_argument("--coverage-gamma", type=float, default=1.0)
    parser.add_argument("--coverage-q-low", type=float, default=0.1)
    parser.add_argument("--coverage-q-high", type=float, default=0.99)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=str,
        default="weights/scoring_weights.json",
        help="Path to write learned weights JSON.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    return parser.parse_args()


def _build_dataset(
    dataset_name: str, data_root: str, transform
) -> datasets.VisionDataset:
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_score_loader(
    preprocess, data_root: str, dataset_name: str, device: torch.device, batch_size: int, num_workers: int
) -> DataLoader:
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
    return dataset.classes  # type: ignore[attr-defined]


def load_adapter(adapter_path: str | None, input_dim: int, device: torch.device) -> AdapterMLP | None:
    if not adapter_path:
        return None
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter 权重不存在: {path}")
    adapter = AdapterMLP(input_dim=input_dim)
    state_dict = torch.load(path, map_location=device)
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()
    return adapter


def fit_ridge_regression_nonnegative(
    features: np.ndarray,
    targets: np.ndarray,
    l2_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of samples.")
    if l2_lambda < 0:
        raise ValueError("ridge-lambda must be non-negative.")
    if learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if max_iter <= 0:
        raise ValueError("max-iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    num_samples, num_features = features.shape
    weights = np.full(num_features, 1.0 / num_features, dtype=np.float64)
    bias = float(targets.mean())

    features = features.astype(np.float64)
    targets = targets.astype(np.float64)

    for _ in range(max_iter):
        preds = features @ weights + bias
        errors = preds - targets
        grad_w = (features.T @ errors) / num_samples + l2_lambda * weights
        grad_b = errors.mean()

        next_weights = project_to_simplex(weights - learning_rate * grad_w)
        next_bias = bias - learning_rate * grad_b

        if np.linalg.norm(next_weights - weights) < tol and abs(next_bias - bias) < tol:
            weights = next_weights
            bias = next_bias
            break

        weights = next_weights
        bias = next_bias

    return weights, float(bias)


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    if vector.ndim != 1:
        raise ValueError("vector must be a 1D array.")
    if vector.size == 0:
        raise ValueError("vector must be non-empty.")
    sorted_vec = np.sort(vector)[::-1]
    cumulative_sum = np.cumsum(sorted_vec)
    rho_candidates = sorted_vec - (cumulative_sum - 1) / np.arange(1, vector.size + 1)
    rho_indices = np.where(rho_candidates > 0)[0]
    if rho_indices.size == 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)
    rho = rho_indices[-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    projected_sum = projected.sum()
    if projected_sum <= 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)
    return (projected / projected_sum).astype(np.float64)


def normalize_scores_with_quantiles(
    scores: np.ndarray,
    labels: np.ndarray,
    lower_q: float = 0.001,
    upper_q: float = 0.999,
) -> np.ndarray:
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels length must match scores length.")
    normalized = np.zeros_like(scores, dtype=np.float64)
    for cls in np.unique(labels):
        mask = labels == cls
        if not np.any(mask):
            continue
        class_scores = scores[mask]
        lower = float(np.quantile(class_scores, lower_q))
        upper = float(np.quantile(class_scores, upper_q))
        if upper <= lower:
            normalized[mask] = 0.5
            continue
        clipped = np.clip(class_scores, lower, upper)
        normalized[mask] = (clipped - lower) / (upper - lower)
    return normalized.astype(np.float64)


def build_output_path(base_path: str) -> Path:
    return Path(base_path)


def resolve_proxy_log_path(proxy_log_arg: str, dataset: str, seed: int) -> Path:
    candidate = Path(proxy_log_arg)
    if candidate.suffix == ".npz" and candidate.exists():
        return candidate

    if candidate.is_dir():
        base_dir = candidate
    else:
        base_dir = candidate.parent

    if not base_dir.exists():
        base_dir = Path("weights") / "proxy_logs" / str(seed)

    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"未找到代理训练日志目录: {base_dir}")

    matches = sorted(base_dir.glob(f"{dataset}_resnet18_*.npz"))
    if not matches:
        matches = sorted(base_dir.glob("*.npz"))
    if not matches:
        seed_dir = base_dir / str(seed)
        if seed_dir.exists() and seed_dir.is_dir():
            matches = sorted(seed_dir.glob(f"{dataset}_resnet18_*.npz"))
            if not matches:
                matches = sorted(seed_dir.glob("*.npz"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"未找到代理训练日志文件: {base_dir}")
    return matches[0]


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    proxy_log = resolve_proxy_log_path(args.proxy_log, args.dataset, seed)

    early_result = EarlyLearnabilityScore(
        proxy_log, early_epochs=args.early_epochs
    ).compute()
    margin_result = MarginScore(proxy_log, delta=args.margin_delta).compute()
    stability_result = StabilityScore(proxy_log).compute()
    coverage_result = CoverageGainScore(
        proxy_log,
        tau_g=args.coverage_tau_g,
        s_g=args.coverage_s_g,
        k=args.coverage_k,
        gamma=args.coverage_gamma,
        q_low=args.coverage_q_low,
        q_high=args.coverage_q_high,
    ).compute()

    if (
        not np.array_equal(early_result.indices, margin_result.indices)
        or not np.array_equal(early_result.indices, stability_result.indices)
        or not np.array_equal(early_result.indices, coverage_result.indices)
    ):
        raise ValueError("动态指标的 indices 不一致，无法对齐样本。")

    dynamic_scores = (
        stability_result.scores + early_result.scores + coverage_result.scores
    ) / 3.0
    if early_result.labels is None:
        raise ValueError("代理训练日志缺少 labels，无法按类别归一化动态分数。")
    dynamic_scores = normalize_scores_with_quantiles(dynamic_scores, early_result.labels)

    class_names = load_class_names(args.dataset, args.data_root)
    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device, k=args.dds_k
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=args.div_k,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )

    adapter = load_adapter(args.adapter_path, dds_metric.extractor.embed_dim, device)

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        args.data_root,
        args.dataset,
        device,
        args.batch_size,
        args.num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        args.data_root,
        args.dataset,
        device,
        args.batch_size,
        args.num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        args.data_root,
        args.dataset,
        device,
        args.batch_size,
        args.num_workers,
    )

    dds_scores = dds_metric.score_dataset(dds_loader, adapter=adapter)
    div_scores = div_metric.score_dataset(div_loader, adapter=adapter)
    sa_scores = sa_metric.score_dataset(sa_loader, adapter=adapter)

    if early_result.labels is not None:
        if not np.array_equal(early_result.labels, sa_scores.labels.numpy()):
            raise ValueError("代理训练日志的标签与评分数据集标签不一致。")

    static_features = np.stack(
        [
            sa_scores.scores.numpy(),
            div_scores.scores.numpy(),
            dds_scores.scores.numpy(),
        ],
        axis=1,
    ).astype(np.float64)
    dynamic_scores = dynamic_scores.astype(np.float64)

    weights, bias = fit_ridge_regression_nonnegative(
        static_features,
        dynamic_scores,
        args.ridge_lambda,
        args.learning_rate,
        args.max_iter,
        args.tol,
    )

    output_path = build_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, dict[str, object]]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded

    dataset_entry = data.get(args.dataset, {})
    seed_key = str(seed)
    dataset_entry[seed_key] = {
        "sa": float(weights[0]),
        "div": float(weights[1]),
        "dds": float(weights[2]),
    }
    dataset_entry[f"{seed_key}_meta"] = {
        "raw_sa": float(weights[0]),
        "raw_div": float(weights[1]),
        "raw_dds": float(weights[2]),
        "bias": float(bias),
        "ridge_lambda": float(args.ridge_lambda),
        "proxy_log": str(proxy_log),
    }
    data[args.dataset] = dataset_entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Learned weights saved to", output_path)
    print("Learned weights:", dataset_entry[seed_key])


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)
