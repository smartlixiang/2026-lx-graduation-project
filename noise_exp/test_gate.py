from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from utils.proxy_log_utils import resolve_proxy_log_path
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.seed import set_seed
from weights import (
    AbsorptionGainScore,
    ConfusionComplementarityScore,
    TransferabilityAlignmentScore,
)
from weights.dynamic_utils import (
    DynamicComponentResult,
    FoldLogData,
    load_cv_fold_logs,
    quantile_minmax_dynamic,
    resolve_epoch_windows,
)

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the revised noise-aware gate for dynamic pseudo-label construction. "
            "Version 1 gates the positive part of A/C/T instead of only A/C."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--noise-root", type=str, default="noise")
    parser.add_argument("--proxy-log", type=str, default="weights/proxy_logs")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--bins", type=int, default=60)
    return parser.parse_args()


def resolve_default_proxy_epochs(dataset_name: str) -> int:
    if dataset_name in {CIFAR10, CIFAR100}:
        return 200
    if dataset_name == TINY_IMAGENET:
        return 90
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_train_dataset(dataset_name: str, data_root: str):
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=None)
    if dataset_name == TINY_IMAGENET:
        train_root = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=None)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_labels(dataset) -> np.ndarray:
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


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    max_logits = np.max(logits, axis=2, keepdims=True)
    shifted = logits - max_logits
    logsumexp = np.log(np.sum(np.exp(np.clip(shifted, -50.0, 50.0)), axis=2) + EPS)
    logsumexp = logsumexp + max_logits.squeeze(2)

    true_logits = np.take_along_axis(logits, labels.reshape(1, -1, 1), axis=2).squeeze(2)
    return (logsumexp - true_logits).astype(np.float64)


def classwise_quantile_score(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros(values.shape[0], dtype=np.float64)

    for cls in np.unique(labels):
        idx = np.where(labels == int(cls))[0]
        if idx.size == 0:
            continue
        output[idx] = quantile_minmax_dynamic(values[idx].astype(np.float32)).astype(np.float64)

    return np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)


def compute_gate_source_metrics(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the dynamic-failure signals used by the gate."""
    num_samples = labels_for_metric.shape[0]
    learned_sum = np.zeros(num_samples, dtype=np.float64)
    forgetting_sum = np.zeros(num_samples, dtype=np.float64)
    forgetting_count = np.zeros(num_samples, dtype=np.int64)
    loss_difficulty_sum = np.zeros(num_samples, dtype=np.float64)
    fold_count = np.zeros(num_samples, dtype=np.int64)

    for fold in tqdm(folds, desc="Computing gate source metrics", unit="fold"):
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
        fold_count[train_idx] += 1

    if np.any(fold_count == 0):
        missing = np.where(fold_count == 0)[0]
        raise ValueError(f"部分样本没有出现在任何 train fold 中: {missing[:10]}")

    forgetting_rate_all = np.full(num_samples, np.nan, dtype=np.float64)
    valid = forgetting_count > 0
    forgetting_rate_all[valid] = forgetting_sum[valid] / forgetting_count[valid]

    return {
        "ever_learned": learned_sum / fold_count,
        "forgetting_rate": forgetting_rate_all,
        "loss_difficulty": loss_difficulty_sum / fold_count,
    }


def build_noise_gate(metrics: dict[str, np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forgetting_rate = np.nan_to_num(
        metrics["forgetting_rate"],
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    dynamic_failure = np.maximum.reduce([
        1.0 - np.clip(metrics["ever_learned"], 0.0, 1.0),
        np.clip(forgetting_rate, 0.0, 1.0),
        np.clip(metrics["loss_difficulty"], 0.0, 1.0),
    ])

    penalty = np.maximum(0.0, standard_zscore_by_class(dynamic_failure, labels))
    gate = np.maximum(0.0, 1.0 - penalty)
    gate = np.clip(gate, 0.0, 1.0)

    return dynamic_failure.astype(np.float64), penalty.astype(np.float64), gate.astype(np.float64)


def compute_dynamic_components(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
) -> dict[str, DynamicComponentResult]:
    print("\nComputing A/C/T dynamic components with noisy labels...")
    return {
        "A": AbsorptionGainScore().compute(folds=folds, labels_all=labels_for_metric),
        "C": ConfusionComplementarityScore().compute(folds=folds, labels_all=labels_for_metric),
        "T": TransferabilityAlignmentScore().compute(folds=folds, labels_all=labels_for_metric),
    }


def split_gate_positive_part(values: np.ndarray, gate: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)

    positive = np.maximum(values, 0.0)
    negative = np.minimum(values, 0.0)
    return negative + gate * positive


def build_pseudo_labels(
    component_results: dict[str, DynamicComponentResult],
    gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original, old-gated, and revised-gated pseudo-labels.

    old-gated:
        Only the positive parts of A/C are gated. T is preserved.

    revised-gated:
        The positive parts of A/C/T are all gated. Negative contributions are preserved.
        This prevents obvious noisy samples from receiving high pseudo-labels through T.
    """
    a = component_results["A"].final_normalized.astype(np.float64)
    c = component_results["C"].final_normalized.astype(np.float64)
    t = component_results["T"].final_normalized.astype(np.float64)

    for name, values in {"A": a, "C": c, "T": t}.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN/inf values.")

    original_raw = (a + c + t) / 3.0

    gated_a = split_gate_positive_part(a, gate)
    gated_c = split_gate_positive_part(c, gate)
    old_gated_raw = (gated_a + gated_c + t) / 3.0

    gated_t = split_gate_positive_part(t, gate)
    revised_gated_raw = (gated_a + gated_c + gated_t) / 3.0

    original_target = standard_zscore(original_raw).astype(np.float64)
    old_gated_target = standard_zscore(old_gated_raw).astype(np.float64)
    revised_gated_target = standard_zscore(revised_gated_raw).astype(np.float64)
    return original_target, old_gated_target, revised_gated_target


def summarize(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, int]:
    selected = np.asarray(values, dtype=np.float64)[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(selected)),
        float(np.max(selected)),
        float(np.min(selected)),
        int(selected.size),
    )


def print_group_stats(metric_name: str, values: np.ndarray, is_noisy: np.ndarray) -> None:
    clean_mean, clean_max, clean_min, clean_n = summarize(values, ~is_noisy)
    noisy_mean, noisy_max, noisy_min, noisy_n = summarize(values, is_noisy)

    print(f"\n[{metric_name}]")
    print(f"  clean: mean={clean_mean:.6f}, max={clean_max:.6f}, min={clean_min:.6f}, n={clean_n}")
    print(f"  noisy: mean={noisy_mean:.6f}, max={noisy_max:.6f}, min={noisy_min:.6f}, n={noisy_n}")


def plot_pseudo_label_histogram(
    pseudo_labels: np.ndarray,
    is_noisy: np.ndarray,
    output_path: Path,
    bins: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_values = np.asarray(pseudo_labels, dtype=np.float64)[~is_noisy]
    noisy_values = np.asarray(pseudo_labels, dtype=np.float64)[is_noisy]
    clean_values = clean_values[np.isfinite(clean_values)]
    noisy_values = noisy_values[np.isfinite(noisy_values)]

    all_values = np.concatenate([clean_values, noisy_values], axis=0)
    if all_values.size == 0:
        raise ValueError("No finite pseudo-label values to plot.")

    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if abs(value_max - value_min) < 1e-12:
        value_min -= 1.0
        value_max += 1.0

    hist_bins = np.linspace(value_min, value_max, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(clean_values, bins=hist_bins, density=True, alpha=0.55, color="blue", label="Clean samples")
    plt.hist(noisy_values, bins=hist_bins, density=True, alpha=0.55, color="red", label="Noisy samples")
    plt.xlabel("Revised gated dynamic pseudo-label")
    plt.ylabel("Density")
    plt.title("Pseudo-label distribution after revised noise-aware gating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    resolved_proxy_epochs = (
        int(args.proxy_epochs) if args.proxy_epochs is not None else resolve_default_proxy_epochs(args.dataset)
    )

    dataset = build_train_dataset(args.dataset, args.data_root)
    clean_labels = extract_labels(dataset)
    noisy_labels, is_noisy = build_noisy_labels(
        clean_labels=clean_labels,
        dataset_name=args.dataset,
        seed=args.seed,
        noise_root=args.noise_root,
    )

    proxy_log_path = resolve_proxy_log_path(
        args.proxy_log,
        args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        max_epoch=resolved_proxy_epochs,
    )
    folds, labels_from_logs = load_cv_fold_logs(proxy_log_path, args.dataset, args.data_root)
    if labels_from_logs.shape[0] != clean_labels.shape[0]:
        raise ValueError("proxy log labels and dataset labels have different lengths.")

    print("=== Revised noise gate diagnostic ===")
    print(f"dataset={args.dataset}, seed={args.seed}, proxy_model={args.proxy_model}, proxy_epochs={resolved_proxy_epochs}")
    print(f"proxy_log={proxy_log_path}")
    print(f"num_samples={clean_labels.shape[0]}, num_noisy={int(np.sum(is_noisy))}")
    print("Gate is computed against noisy labels constructed from noise_list_{seed}.txt.")
    print("Revision: positive parts of A/C/T are all gated; negative parts are preserved.")

    gate_metrics = compute_gate_source_metrics(folds, noisy_labels)
    dynamic_failure, penalty, gate = build_noise_gate(gate_metrics, noisy_labels)

    component_results = compute_dynamic_components(folds, noisy_labels)
    original_target, old_gated_target, revised_gated_target = build_pseudo_labels(component_results, gate)

    print_group_stats("EverLearned", gate_metrics["ever_learned"], is_noisy)
    print_group_stats("ForgettingRate", gate_metrics["forgetting_rate"], is_noisy)
    print_group_stats("LossDifficulty", gate_metrics["loss_difficulty"], is_noisy)
    print_group_stats("DynamicFailure D", dynamic_failure, is_noisy)
    print_group_stats("Penalty P", penalty, is_noisy)
    print_group_stats("Gate g", gate, is_noisy)

    print_group_stats("Original pseudo-label", original_target, is_noisy)
    print_group_stats("Old gated pseudo-label (A/C only)", old_gated_target, is_noisy)
    print_group_stats("Revised gated pseudo-label (A/C/T)", revised_gated_target, is_noisy)

    output_path = Path(__file__).resolve().parent / f"gate1_pseudolabel_hist_{args.dataset}_seed{args.seed}.png"
    plot_pseudo_label_histogram(revised_gated_target, is_noisy, output_path, args.bins)
    print(f"\nSaved histogram to: {output_path}")


if __name__ == "__main__":
    main()
