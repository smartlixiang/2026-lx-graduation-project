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
from utils.proxy_log_utils import load_proxy_log, resolve_proxy_log_path
from utils.seed import parse_seed_list, set_seed
from utils.score_utils import quantile_minmax
from utils.static_score_cache import get_or_compute_static_scores
from weights import (
    AbsorptionEfficiencyScore,
    CoverageGainScore,
    InformativenessScore,
    RiskScore,
    PersistentDifficultyScore,
    TransferGainScore,
)

PROJECT_ROOT = Path(__file__).resolve().parent


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
        default="weights/proxy_logs",
        help="Proxy log root path (or a specific log file/dir).",
    )
    parser.add_argument(
        "--proxy-epochs",
        type=int,
        default=None,
        help="Max epochs for proxy log directory name. Defaults to latest epoch folder.",
    )
    parser.add_argument("--adapter-path", type=str, default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt", help="Optional adapter path.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=int, default=10)
    parser.add_argument("--dds-k", type=float, default=10)
    parser.add_argument("--coverage-tau-g", type=float, default=0.15)
    parser.add_argument("--coverage-s-g", type=float, default=0.07)
    parser.add_argument(
        "--coverage-k-pct",
        type=float,
        default=0.005,
        help="Class-wise k ratio for CoverageGainScore (fraction of class size).",
    )
    parser.add_argument("--coverage-q-low", type=float, default=0.002)
    parser.add_argument("--coverage-q-high", type=float, default=0.998)
    parser.add_argument(
        "--sanity-keep-ratio",
        type=float,
        default=0.6,
        help="Keep ratio for top-k overlap sanity check between u_raw_base and u_raw.",
    )
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


def build_output_path(base_path: str) -> Path:
    return Path(base_path)


def _align_scores(scores: np.ndarray, indices: np.ndarray, num_samples: int) -> np.ndarray:
    if scores.shape[0] != indices.shape[0]:
        raise ValueError("scores and indices length mismatch.")
    full = np.full((num_samples,), np.nan, dtype=np.float32)
    if np.min(indices) < 0 or np.max(indices) >= num_samples:
        raise ValueError("indices out of range when aligning scores.")
    full[indices.astype(np.int64)] = scores.astype(np.float32)
    if np.any(~np.isfinite(full)):
        raise ValueError("aligned scores contain NaN/inf values.")
    return full


def _topk_indices(values: np.ndarray, keep_ratio: float) -> np.ndarray:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")
    k = max(1, int(np.ceil(values.size * keep_ratio)))
    return np.argpartition(values, -k)[-k:]


def _topk_overlap(base: np.ndarray, other: np.ndarray, keep_ratio: float) -> float:
    base_idx = set(_topk_indices(base, keep_ratio).tolist())
    other_idx = set(_topk_indices(other, keep_ratio).tolist())
    if not base_idx:
        return 0.0
    return len(base_idx & other_idx) / float(len(base_idx))


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    proxy_log = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        seed,
        max_epoch=args.proxy_epochs,
    )
    proxy_data = load_proxy_log(proxy_log, args.dataset, args.data_root)

    absorption_result = AbsorptionEfficiencyScore(proxy_log).compute(proxy_logs=proxy_data)
    informativeness_result = InformativenessScore(proxy_log).compute(proxy_logs=proxy_data)
    coverage_result = CoverageGainScore(
        proxy_log,
        tau_g=args.coverage_tau_g,
        s_g=args.coverage_s_g,
        k_pct=args.coverage_k_pct,
        q_low=args.coverage_q_low,
        q_high=args.coverage_q_high,
    ).compute(proxy_logs=proxy_data)
    risk_result = RiskScore(proxy_log).compute(proxy_logs=proxy_data)

    indices = absorption_result.indices
    if (
        not np.array_equal(indices, informativeness_result.indices)
        or not np.array_equal(indices, coverage_result.indices)
        or not np.array_equal(indices, risk_result.indices)
    ):
        raise ValueError("动态指标的 indices 不一致，无法对齐样本。")
    if absorption_result.labels is None:
        raise ValueError("代理训练日志缺少 labels，无法对齐动态分数。")

    num_samples = absorption_result.labels.shape[0]
    if np.array_equal(indices, np.arange(num_samples)):
        a_scores = absorption_result.scores.astype(np.float32)
        b_scores = informativeness_result.scores.astype(np.float32)
        c_scores = coverage_result.scores.astype(np.float32)
        r_scores = risk_result.scores.astype(np.float32)
    else:
        a_scores = _align_scores(absorption_result.scores, absorption_result.indices, num_samples)
        b_scores = _align_scores(informativeness_result.scores, informativeness_result.indices, num_samples)
        c_scores = _align_scores(coverage_result.scores, coverage_result.indices, num_samples)
        r_scores = _align_scores(risk_result.scores, risk_result.indices, num_samples)

    cv_log_dir = Path(proxy_log)
    if not cv_log_dir.exists():
        raise FileNotFoundError(f"cv_log_dir not found: {cv_log_dir}")
    if cv_log_dir.is_file():
        raise ValueError("cv_log_dir must be a directory containing fold_*.npz files.")

    dataset = _build_dataset(args.dataset, args.data_root, transform=None)
    t_result = TransferGainScore().compute(cv_log_dir, dataset)
    t_scores_raw = t_result["score"].astype(np.float32)
    t_scores = quantile_minmax(t_scores_raw, q_low=0.002, q_high=0.998)
    v_result = PersistentDifficultyScore().compute(cv_log_dir, dataset)
    v_scores = v_result["score"].astype(np.float32)

    for name, arr in {
        "A": a_scores,
        "B": b_scores,
        "C": c_scores,
        "R": r_scores,
        "T": t_scores,
        "V": v_scores,
    }.items():
        if arr.shape != (num_samples,):
            raise ValueError(f"{name} score shape mismatch: {arr.shape}, expected ({num_samples},).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} scores contain NaN/inf values.")

    u_raw_base = a_scores + b_scores + c_scores - r_scores
    u_raw = u_raw_base + t_scores + v_scores
    if not np.all(np.isfinite(u_raw)):
        raise ValueError("u_raw contains NaN/inf values.")
    u_scores = quantile_minmax(u_raw.astype(np.float32), q_low=0.002, q_high=0.998)
    if not np.all(np.isfinite(u_scores)):
        raise ValueError("u_norm contains NaN/inf values.")

    base_mean = float(np.mean(u_raw_base))
    base_var = float(np.var(u_raw_base))
    tv_mean = float(np.mean(u_raw))
    tv_var = float(np.var(u_raw))
    overlap = _topk_overlap(u_raw_base, u_raw, args.sanity_keep_ratio)
    print(
        "Sanity check u_raw: "
        f"base_mean={base_mean:.6f}, base_var={base_var:.6f}, "
        f"tv_mean={tv_mean:.6f}, tv_var={tv_var:.6f}, "
        f"topk_overlap={overlap:.6f} (keep_ratio={args.sanity_keep_ratio})"
    )

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

    dataset_for_labels = _build_dataset(args.dataset, args.data_root, transform=None)
    num_samples = len(dataset_for_labels)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(dds_loader, adapter=adapter)
        div_scores_local = div_metric.score_dataset(div_loader, adapter=adapter)
        sa_scores_local = sa_metric.score_dataset(sa_loader, adapter=adapter)
        return {
            "sa": sa_scores_local.scores.numpy(),
            "div": div_scores_local.scores.numpy(),
            "dds": dds_scores_local.scores.numpy(),
            "labels": np.asarray(dataset_for_labels.targets),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=args.dataset,
        clip_model=args.clip_model,
        adapter_path=args.adapter_path,
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    if absorption_result.labels is not None:
        if not np.array_equal(absorption_result.labels, static_scores["labels"]):
            raise ValueError("代理训练日志的标签与评分数据集标签不一致。")

    static_features = np.stack(
        [
            static_scores["sa"],
            static_scores["div"],
            static_scores["dds"],
        ],
        axis=1,
    ).astype(np.float64)
    dynamic_scores = u_scores.astype(np.float64)

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
