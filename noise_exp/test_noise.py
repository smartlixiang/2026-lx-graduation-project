from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from model.adapter import load_trained_adapters
from scoring import SemanticAlignment
from utils.global_config import CONFIG
from utils.proxy_log_utils import resolve_proxy_log_path
from utils.seed import set_seed
from weights.dynamic_utils import FoldLogData, load_cv_fold_logs, quantile_minmax_dynamic, resolve_epoch_windows

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose SA and proxy-training dynamics under label noise."
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--noise-root", type=str, default="noise")
    parser.add_argument("--proxy-log", type=str, default="weights/proxy_logs")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--adapter-image-path", type=str, default=None)
    parser.add_argument("--adapter-text-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args()


def build_train_dataset(dataset_name: str, data_root: str, transform=None) -> datasets.VisionDataset:
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


def extract_labels(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([label for _, label in dataset.samples], dtype=np.int64)

    labels = np.empty(len(dataset), dtype=np.int64)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels[idx] = int(label.item() if hasattr(label, "item") else label)
    return labels


def set_dataset_labels(dataset: torch.utils.data.Dataset, labels: np.ndarray) -> None:
    labels = np.asarray(labels, dtype=np.int64)
    labels_list = [int(x) for x in labels.tolist()]
    updated = False

    if hasattr(dataset, "targets"):
        dataset.targets = labels_list
        updated = True
    if hasattr(dataset, "labels"):
        dataset.labels = labels_list
        updated = True
    if hasattr(dataset, "samples"):
        dataset.samples = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.samples)]
        updated = True
    if hasattr(dataset, "imgs"):
        dataset.imgs = [(path, int(labels[idx])) for idx, (path, _) in enumerate(dataset.imgs)]
        updated = True

    if not updated:
        raise TypeError("当前数据集对象不支持原地修改标签。")


def read_noise_list(noise_path: Path) -> np.ndarray:
    if not noise_path.exists():
        raise FileNotFoundError(f"未找到标签注噪文件: {noise_path}")

    mapping = np.loadtxt(noise_path, dtype=np.int64)
    if mapping.ndim == 1:
        mapping = mapping.reshape(1, 2)
    if mapping.ndim != 2 or mapping.shape[1] != 2:
        raise ValueError(f"标签注噪文件必须是两列 txt: {noise_path}, 当前 shape={mapping.shape}")
    return mapping


def build_noisy_labels(
    clean_labels: np.ndarray,
    dataset_name: str,
    seed: int,
    noise_root: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    clean_labels = np.asarray(clean_labels, dtype=np.int64)
    noise_path = Path(noise_root) / dataset_name / f"noise_list_{seed}.txt"
    mapping = read_noise_list(noise_path)

    noisy_indices = mapping[:, 0].astype(np.int64)
    noisy_new_labels = mapping[:, 1].astype(np.int64)

    if len(np.unique(noisy_indices)) != noisy_indices.shape[0]:
        raise ValueError(f"标签注噪文件存在重复 sample_id: {noise_path}")
    if np.any(noisy_indices < 0) or np.any(noisy_indices >= clean_labels.shape[0]):
        raise ValueError(f"标签注噪文件存在越界 sample_id: {noise_path}")
    if np.any(noisy_new_labels == clean_labels[noisy_indices]):
        raise ValueError(f"标签注噪文件中存在 noisy_label 与 clean label 相同的样本: {noise_path}")

    noisy_labels = clean_labels.copy()
    noisy_labels[noisy_indices] = noisy_new_labels

    is_noisy = np.zeros(clean_labels.shape[0], dtype=bool)
    is_noisy[noisy_indices] = True
    return noisy_labels, is_noisy


def compute_sa_scores(
    *,
    args: argparse.Namespace,
    noisy_labels: np.ndarray,
    class_names: list[str],
    device: torch.device,
) -> np.ndarray:
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=args.data_root,
        debug_prompts=args.debug_prompts,
    )

    score_dataset = build_train_dataset(
        args.dataset,
        args.data_root,
        transform=sa_metric.extractor.preprocess,
    )
    set_dataset_labels(score_dataset, noisy_labels)

    loader = DataLoader(
        score_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=sa_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
        adapter_image_path=args.adapter_image_path,
        adapter_text_path=args.adapter_text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    result = sa_metric.score_dataset(
        tqdm(loader, desc="Computing SA with noisy labels", unit="batch"),
        adapter_image=image_adapter,
        adapter_text=text_adapter,
    )
    return result.scores.numpy().astype(np.float64)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(np.clip(shifted, -50.0, 50.0)), axis=2) + EPS)
    logsumexp = logsumexp + np.max(logits, axis=2)
    true_logits = np.take_along_axis(logits, labels.reshape(1, -1, 1), axis=2).squeeze(2)
    return (logsumexp - true_logits).astype(np.float64)


def classwise_quantile_score(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros(values.shape[0], dtype=np.float64)

    for cls in np.unique(labels):
        idx = np.where(labels == int(cls))[0]
        if idx.size == 0:
            continue
        out[idx] = quantile_minmax_dynamic(values[idx].astype(np.float32)).astype(np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def compute_dynamic_metrics(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute train-view dynamics against noisy labels.

    forgetting_rate:
        首次预测正确之后，错误 epoch 数 / 首次正确之后的 epoch 数。
        从未学会或最后一轮才首次学会的样本没有“学会后阶段”，记为 NaN。

    loss_difficulty:
        后期 loss 均值和后期 loss 波动的类内分位最大值。
        这个定义同时覆盖“后期仍然难以拟合”和“后期状态不稳定”两种情形，且不引入手工权重。
    """
    num_samples = labels_for_metric.shape[0]
    forgetting_sum = np.zeros(num_samples, dtype=np.float64)
    forgetting_count = np.zeros(num_samples, dtype=np.int64)
    loss_difficulty_sum = np.zeros(num_samples, dtype=np.float64)
    learned_sum = np.zeros(num_samples, dtype=np.float64)
    count = np.zeros(num_samples, dtype=np.int64)

    for fold in tqdm(folds, desc="Computing train-view dynamics", unit="fold"):
        train_idx = fold.train_indices.astype(np.int64)
        y_train = labels_for_metric[train_idx]

        logits = np.asarray(fold.train_logits, dtype=np.float64)
        preds = np.argmax(logits, axis=2).astype(np.int64)
        correct = preds == y_train.reshape(1, -1)
        num_epochs = correct.shape[0]

        has_correct = np.any(correct, axis=0)
        first_correct = np.argmax(correct, axis=0).astype(np.int64)
        learned_sum[train_idx] += has_correct.astype(np.float64)

        seen_correct = np.maximum.accumulate(correct, axis=0)
        forgetting_count_raw = np.sum(seen_correct & (~correct), axis=0).astype(np.float64)
        after_learn_epochs = (num_epochs - first_correct - 1).astype(np.float64)
        valid_forgetting = has_correct & (after_learn_epochs > 0)
        forgetting_rate = np.full(correct.shape[1], np.nan, dtype=np.float64)
        forgetting_rate[valid_forgetting] = (
            forgetting_count_raw[valid_forgetting] / after_learn_epochs[valid_forgetting]
        )

        finite_forgetting = np.isfinite(forgetting_rate)
        forgetting_sum[train_idx[finite_forgetting]] += forgetting_rate[finite_forgetting]
        forgetting_count[train_idx[finite_forgetting]] += 1

        losses = cross_entropy_from_logits(logits, y_train)
        _, _, late_idx = resolve_epoch_windows(num_epochs)
        late_loss_mean = np.mean(losses[late_idx], axis=0)
        late_loss_std = np.std(losses[late_idx], axis=0)
        late_mean_score = classwise_quantile_score(late_loss_mean, y_train)
        late_std_score = classwise_quantile_score(late_loss_std, y_train)
        loss_difficulty = np.maximum(late_mean_score, late_std_score)

        loss_difficulty_sum[train_idx] += loss_difficulty
        count[train_idx] += 1

    if np.any(count == 0):
        missing = np.where(count == 0)[0]
        raise ValueError(f"部分样本没有出现在任何 train fold 中: {missing[:10]}")

    forgetting_rate_all = np.full(num_samples, np.nan, dtype=np.float64)
    valid = forgetting_count > 0
    forgetting_rate_all[valid] = forgetting_sum[valid] / forgetting_count[valid]

    return {
        "forgetting_rate": forgetting_rate_all,
        "loss_difficulty": loss_difficulty_sum / count,
        "ever_learned": learned_sum / count,
    }


def build_noise_score(
    sa_scores: np.ndarray,
    noisy_labels: np.ndarray,
    forgetting_rate: np.ndarray,
    loss_difficulty: np.ndarray,
) -> dict[str, np.ndarray]:
    # SA 越低越可疑，因此对 -SA 做类内分位归一化。
    sa_noise = classwise_quantile_score(-sa_scores, noisy_labels)

    # 从未学会的样本由 loss_difficulty 捕捉；遗忘率 NaN 不直接当作 1。
    forgetting_evidence = np.nan_to_num(forgetting_rate, nan=0.0, posinf=1.0, neginf=0.0)
    forgetting_evidence = np.clip(forgetting_evidence, 0.0, 1.0)
    loss_evidence = np.clip(loss_difficulty, 0.0, 1.0)

    dynamic_noise = np.maximum(forgetting_evidence, loss_evidence)
    noise_score = np.sqrt(np.clip(sa_noise * dynamic_noise, 0.0, 1.0))

    return {
        "sa_noise": sa_noise,
        "dynamic_noise": dynamic_noise,
        "noise_score": noise_score,
    }


def summarize(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(selected)), float(np.std(selected, ddof=0)), int(selected.size)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")

    xx = x[finite]
    yy = y[finite]
    if float(np.std(xx)) < 1e-12 or float(np.std(yy)) < 1e-12:
        return float("nan")

    return float(np.corrcoef(xx, yy)[0, 1])


def print_group_stats(metric_name: str, values: np.ndarray, is_noisy: np.ndarray) -> None:
    clean_mean, clean_std, clean_n = summarize(values, ~is_noisy)
    noisy_mean, noisy_std, noisy_n = summarize(values, is_noisy)

    print(f"\n[{metric_name}]")
    print(f"  clean: mean={clean_mean:.6f}, std={clean_std:.6f}, n={clean_n}")
    print(f"  noisy: mean={noisy_mean:.6f}, std={noisy_std:.6f}, n={noisy_n}")


def print_correlation_table(title: str, metrics: dict[str, np.ndarray], mask: np.ndarray) -> None:
    names = list(metrics.keys())
    arrays = [np.asarray(metrics[name], dtype=np.float64)[mask] for name in names]

    print(f"\n[{title} Pearson correlation]")
    print("              " + "  ".join(f"{name:>15s}" for name in names))
    for i, name_i in enumerate(names):
        row = []
        for j in range(len(names)):
            corr = 1.0 if i == j else safe_pearson(arrays[i], arrays[j])
            row.append(f"{corr:15.6f}")
        print(f"{name_i:>12s}  " + "  ".join(row))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else CONFIG.global_device

    clean_dataset = build_train_dataset(args.dataset, args.data_root, transform=None)
    clean_labels = extract_labels(clean_dataset)
    class_names = list(getattr(clean_dataset, "classes", []))
    if not class_names:
        raise ValueError("无法从数据集中读取 class_names。")

    noisy_labels, is_noisy = build_noisy_labels(
        clean_labels,
        dataset_name=args.dataset,
        seed=args.seed,
        noise_root=args.noise_root,
    )

    proxy_log_path = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        max_epoch=args.proxy_epochs,
    )
    folds, labels_from_logs = load_cv_fold_logs(proxy_log_path, args.dataset, args.data_root)
    if labels_from_logs.shape[0] != clean_labels.shape[0]:
        raise ValueError("proxy log labels and dataset labels have different lengths.")

    print("=== Noise dynamic diagnostic ===")
    print(f"dataset={args.dataset}, seed={args.seed}, proxy_model={args.proxy_model}")
    print(f"proxy_log={proxy_log_path}")
    print(f"num_samples={clean_labels.shape[0]}, num_noisy={int(np.sum(is_noisy))}")
    print("Metrics are computed against noisy labels constructed from noise_list_{seed}.txt.")
    print("ForgettingRate = wrong epochs after first correct / epochs after first correct; never-learned samples are NaN.")
    print("LossDifficulty = max(classwise late-loss-mean score, classwise late-loss-std score).")
    print("NoiseScore = sqrt(SA-noise evidence * max(ForgettingRate, LossDifficulty)).")

    sa_scores = compute_sa_scores(
        args=args,
        noisy_labels=noisy_labels,
        class_names=class_names,
        device=device,
    )
    dynamic_metrics = compute_dynamic_metrics(folds, noisy_labels)
    score_metrics = build_noise_score(
        sa_scores=sa_scores,
        noisy_labels=noisy_labels,
        forgetting_rate=dynamic_metrics["forgetting_rate"],
        loss_difficulty=dynamic_metrics["loss_difficulty"],
    )

    print_group_stats("SA", sa_scores, is_noisy)
    print_group_stats("ForgettingRate", dynamic_metrics["forgetting_rate"], is_noisy)
    print_group_stats("LossDifficulty", dynamic_metrics["loss_difficulty"], is_noisy)
    print_group_stats("EverLearned", dynamic_metrics["ever_learned"], is_noisy)
    print_group_stats("SA-noise evidence", score_metrics["sa_noise"], is_noisy)
    print_group_stats("Dynamic-noise evidence", score_metrics["dynamic_noise"], is_noisy)
    print_group_stats("NoiseScore", score_metrics["noise_score"], is_noisy)

    correlation_metrics = {
        "SA": sa_scores,
        "ForgetRate": dynamic_metrics["forgetting_rate"],
        "LossDiff": dynamic_metrics["loss_difficulty"],
        "NoiseScore": score_metrics["noise_score"],
    }
    print_correlation_table("Clean-sample", correlation_metrics, ~is_noisy)
    print_correlation_table("Noisy-sample", correlation_metrics, is_noisy)


if __name__ == "__main__":
    main()
