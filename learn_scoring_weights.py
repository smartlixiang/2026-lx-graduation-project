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
from weights import EarlyLossScore, ForgettingScore, MarginScore


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
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional adapter path.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=int, default=10)
    parser.add_argument("--div-cdf", action="store_true", help="Enable Normal-CDF adjustment for Div.")
    parser.add_argument("--dds-k", type=float, default=10)
    parser.add_argument("--early-epochs", type=int, default=None)
    parser.add_argument("--margin-delta", type=float, default=1.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-1)
    parser.add_argument(
        "--output",
        type=str,
        default="weights/scoring_weights.json",
        help="Path to write learned weights JSON.",
    )
    parser.add_argument("--device", type=str, default=None)
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


def fit_ridge_regression(
    features: np.ndarray, targets: np.ndarray, l2_lambda: float
) -> tuple[np.ndarray, float]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of samples.")
    if l2_lambda < 0:
        raise ValueError("ridge-lambda must be non-negative.")

    num_samples, num_features = features.shape
    ones = np.ones((num_samples, 1), dtype=features.dtype)
    features_bias = np.concatenate([features, ones], axis=1)

    xtx = features_bias.T @ features_bias
    reg = np.zeros_like(xtx)
    reg[:num_features, :num_features] = l2_lambda * np.eye(num_features, dtype=features.dtype)
    solution = np.linalg.solve(xtx + reg, features_bias.T @ targets)

    weights = solution[:num_features]
    bias = float(solution[-1])
    return weights, bias


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    clipped = np.maximum(weights, 0.0)
    total = float(clipped.sum())
    if total <= 0:
        return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
    return (clipped / total).astype(np.float64)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else CONFIG.global_device
    proxy_log = Path(args.proxy_log)
    if not proxy_log.exists():
        raise FileNotFoundError(f"未找到代理训练日志: {proxy_log}")

    early_result = EarlyLossScore(proxy_log, early_epochs=args.early_epochs).compute()
    margin_result = MarginScore(proxy_log, delta=args.margin_delta).compute()
    forgetting_result = ForgettingScore(proxy_log).compute()

    if not np.array_equal(early_result.indices, margin_result.indices) or not np.array_equal(
        early_result.indices, forgetting_result.indices
    ):
        raise ValueError("动态指标的 indices 不一致，无法对齐样本。")

    dynamic_scores = (
        early_result.scores + margin_result.scores + forgetting_result.scores
    ) / 3.0

    class_names = load_class_names(args.dataset, args.data_root)
    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device, k=args.dds_k
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        k=args.div_k,
        div_cdf=args.div_cdf,
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

    weights, bias = fit_ridge_regression(static_features, dynamic_scores, args.ridge_lambda)
    normalized = normalize_weights(weights)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, dict[str, float]]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded

    dataset_entry = data.get(args.dataset, {})
    dataset_entry["learned"] = {
        "sa": float(normalized[0]),
        "div": float(normalized[1]),
        "dds": float(normalized[2]),
    }
    dataset_entry["learned_meta"] = {
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
    print("Normalized weights:", dataset_entry["learned"])


if __name__ == "__main__":
    main()
