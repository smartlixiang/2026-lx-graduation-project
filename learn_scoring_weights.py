"""Learn scoring weights from proxy training dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from model.adapter import AdapterMLP, load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.proxy_log_utils import resolve_proxy_log_path
from utils.seed import parse_seed_list, set_seed
from utils.score_utils import quantile_minmax
from utils.static_score_cache import get_or_compute_static_scores
from weights import (
    DynamicClassComplementarityScore,
    EarlyLearnabilityScore,
    OOFPatternGapScore,
    OOFSupportScore,
)
from weights.dynamic_v2_utils import default_dynamic_v2_cache_path, load_cv_fold_logs

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
    parser.add_argument(
        "--adapter-image-path",
        type=str,
        default=None,
        help="图像 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument(
        "--adapter-text-path",
        type=str,
        default=None,
        help="文本 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=float, default=0.05)
    parser.add_argument(
        "--dds-k",
        type=int,
        default=5,
        help="Deprecated compatibility arg. DDS legacy naming is kept, but direction count is now ratio-driven.",
    )
    parser.add_argument(
        "--dds-important-eigval-ratio",
        type=float,
        default=0.5,
        help=(
            "Cumulative explained-variance ratio threshold for the third structural component "
            "(legacy DDS name retained for compatibility). Default: 0.5."
        ),
    )
    # Deprecated dynamic-v1 args are intentionally kept for CLI compatibility.
    parser.add_argument("--coverage-tau-g", type=float, default=0.15)
    parser.add_argument("--coverage-s-g", type=float, default=0.07)
    parser.add_argument(
        "--coverage-k-pct",
        type=float,
        default=0.05,
        help="Deprecated: kept for compatibility, unused in dynamic v2.",
    )
    parser.add_argument("--coverage-q-low", type=float, default=0.002)
    parser.add_argument("--coverage-q-high", type=float, default=0.998)
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
        "--debug-prompts",
        action="store_true",
        help="打印 tiny-imagenet 前几个最终英文 prompt（调试用）。",
    )
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
    if dataset_name == TINY_IMAGENET:
        train_root = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
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
    return resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=data_root,
        class_names=dataset.classes,  # type: ignore[attr-defined]
    )


def load_adapters_for_seed(
    args: argparse.Namespace,
    dataset_name: str,
    input_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[AdapterMLP, AdapterMLP, dict[str, Path]]:
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

    for _ in tqdm(range(max_iter), desc="Fitting ridge weights", unit="iter", leave=False):
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


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    del multi_seed
    set_seed(seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    proxy_log = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        seed,
        max_epoch=args.proxy_epochs,
    )

    if not proxy_log.is_dir():
        raise ValueError("Dynamic v2 requires proxy log directory containing fold_*.npz files.")

    folds, labels_all = load_cv_fold_logs(proxy_log, args.dataset, args.data_root)
    num_samples = labels_all.shape[0]

    a_result = EarlyLearnabilityScore().compute(folds=folds, labels_all=labels_all)
    c_result = DynamicClassComplementarityScore().compute(folds=folds, labels_all=labels_all)
    g_result = OOFSupportScore().compute(folds=folds, labels_all=labels_all)
    d_result = OOFPatternGapScore().compute(folds=folds, labels_all=labels_all)

    for name, arr in {
        "A_norm2": a_result.norm2,
        "C_norm2": c_result.norm2,
        "G_norm2": g_result.norm2,
        "D_norm2": d_result.norm2,
    }.items():
        if arr.shape != (num_samples,):
            raise ValueError(f"{name} shape mismatch: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/inf values.")

    u_raw = a_result.norm2 + c_result.norm2 + g_result.norm2 + d_result.norm2
    if not np.all(np.isfinite(u_raw)):
        raise ValueError("u_raw contains NaN/inf values.")
    u_scores = quantile_minmax(u_raw.astype(np.float32), q_low=0.002, q_high=0.998, fallback_value=0.5)
    if not np.all(np.isfinite(u_scores)):
        raise ValueError("u_norm contains NaN/inf values.")

    dynamic_cache_path = default_dynamic_v2_cache_path(args.dataset, seed)
    dynamic_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dynamic_cache_path,
        version=np.array("dynamic_v2", dtype=object),
        component_names=np.array(["A", "C", "G", "D"], dtype=object),
        dataset=np.array(args.dataset, dtype=object),
        seed=np.array(seed, dtype=np.int64),
        proxy_log_path=np.array(str(proxy_log), dtype=object),
        indices=np.arange(num_samples, dtype=np.int64),
        labels=labels_all.astype(np.int64),
        A_norm1_foldwise=a_result.foldwise_norm1.astype(np.float32),
        C_norm1_foldwise=c_result.foldwise_norm1.astype(np.float32),
        G_norm1_foldwise=g_result.foldwise_norm1.astype(np.float32),
        D_norm1_foldwise=d_result.foldwise_norm1.astype(np.float32),
        A_agg=a_result.agg.astype(np.float32),
        A_norm2=a_result.norm2.astype(np.float32),
        C_agg=c_result.agg.astype(np.float32),
        C_norm2=c_result.norm2.astype(np.float32),
        G_agg=g_result.agg.astype(np.float32),
        G_norm2=g_result.norm2.astype(np.float32),
        D_agg=d_result.agg.astype(np.float32),
        D_norm2=d_result.norm2.astype(np.float32),
        u_raw=u_raw.astype(np.float32),
        u_norm=u_scores.astype(np.float32),
    )

    class_names = load_class_names(args.dataset, args.data_root)
    dds_metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=args.dds_k,
        important_eigval_ratio=args.dds_important_eigval_ratio,
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=args.div_k,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=args.data_root,
        debug_prompts=args.debug_prompts,
    )

    image_adapter, text_adapter, adapter_paths = load_adapters_for_seed(
        args, args.dataset, dds_metric.extractor.embed_dim, seed, device
    )

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
    num_samples_static = len(dataset_for_labels)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        )
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        )
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
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
    dataset_entry.pop(f"{seed_key}_meta", None)
    dataset_entry[seed_key] = {
        "sa": float(weights[0]),
        "div": float(weights[1]),
        "dds": float(weights[2]),
        "bias": float(bias),
        "ridge_lambda": float(args.ridge_lambda),
        "proxy_log": str(proxy_log),
        "dynamic_version": "v2",
        "dynamic_components": ["A", "C", "G", "D"],
        "dynamic_cache_path": str(dynamic_cache_path),
    }
    data[args.dataset] = dataset_entry

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Learned weights saved to", output_path)
    print("Learned weights:", dataset_entry[seed_key])
    print("Dynamic v2 cache saved to", dynamic_cache_path)


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)
