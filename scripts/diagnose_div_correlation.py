"""Diagnose correlations between static metrics and dynamic targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100  # noqa: E402
from model.adapter import AdapterMLP  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from weights import EarlyLossScore, ForgettingScore, MarginScore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Div correlations and distributions.")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--proxy-log",
        type=str,
        default="weights/proxy_logs/cifar10_resnet18_2026_01_12_14_31.npz",
    )
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--div-k", type=int, default=10)
    parser.add_argument("--dds-k", type=float, default=10)
    parser.add_argument("--early-epochs", type=int, default=None)
    parser.add_argument("--margin-delta", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="diagnostics")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    return parser.parse_args()


def _build_dataset(dataset_name: str, data_root: str, transform) -> datasets.VisionDataset:
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_score_loader(
    preprocess,
    data_root: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = _build_dataset(dataset_name, data_root, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_class_names(dataset_name: str, data_root: str) -> list[str]:
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


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))
    return pearson_corr(x_rank.astype(np.float64), y_rank.astype(np.float64))


def plot_distributions(output_dir: Path, sa: np.ndarray, div: np.ndarray, dds: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    bins = 50
    plt.hist(sa, bins=bins, alpha=0.5, label="SA", density=True)
    plt.hist(div, bins=bins, alpha=0.5, label="Div", density=True)
    plt.hist(dds, bins=bins, alpha=0.5, label="DDS", density=True)
    plt.legend()
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Distribution of SA / Div / DDS")
    plt.tight_layout()
    plt.savefig(output_dir / "static_score_distributions.png", dpi=200)
    plt.close()


def resolve_output_dir(base_dir: str, seed: int, multi_seed: bool) -> Path:
    output_dir = Path(base_dir)
    if not multi_seed:
        return output_dir
    return output_dir / f"seed_{seed}"


def run_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
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

    sa = sa_scores.scores.numpy().astype(np.float64)
    div = div_scores.scores.numpy().astype(np.float64)
    dds = dds_scores.scores.numpy().astype(np.float64)
    dynamic = dynamic_scores.astype(np.float64)

    print("Correlation with dynamic target (u_i):")
    for name, values in ("SA", sa), ("Div", div), ("DDS", dds):
        print(
            f"- {name}: pearson={pearson_corr(values, dynamic):.4f}, "
            f"spearman={spearman_corr(values, dynamic):.4f}"
        )

    output_dir = resolve_output_dir(args.output_dir, seed, multi_seed)
    plot_distributions(output_dir, sa, div, dds)
    print("Saved distribution plot to", output_dir / "static_score_distributions.png")


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_for_seed(args, seed, multi_seed)
