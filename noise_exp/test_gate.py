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

THIS_FILE = Path(__file__).resolve()
NOISE_EXP_ROOT = THIS_FILE.parent
PROJECT_ROOT = NOISE_EXP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from utils.proxy_log_utils import resolve_proxy_log_path
from utils.score_utils import standard_zscore
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
    resolve_epoch_windows,
)

EPS = 1e-8
NOISE_DYNAMIC_CACHE_ROOT = NOISE_EXP_ROOT / "weights" / "dynamic_cache"
NOISE_PROXY_LOG_ROOT = NOISE_EXP_ROOT / "weights" / "proxy_logs"
CLEAN_DYNAMIC_CACHE_ROOT = PROJECT_ROOT / "weights" / "dynamic_cache"
CLEAN_PROXY_LOG_ROOT = PROJECT_ROOT / "weights" / "proxy_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the current normalized-space positive-part noise gate with clean/noise modes. "
            "The script computes the same rank-min risk gate used by the formal dynamic target flow."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument("--mode", type=str, default="noise", choices=("clean", "noise"), help="noise: use noise_exp caches/logs and noisy labels; clean: use normal caches/logs and clean labels.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--noise-root", type=str, default="noise")
    parser.add_argument(
        "--dynamic-cache-root",
        type=str,
        default=None,
        help=(
            "Dynamic component cache root. Default depends on --mode: "
            "noise -> noise_exp/weights/dynamic_cache; clean -> weights/dynamic_cache."
        ),
    )
    parser.add_argument(
        "--proxy-log",
        type=str,
        default=None,
        help=(
            "Proxy log root or specific fold-log directory. Default depends on --mode: "
            "noise -> noise_exp/weights/proxy_logs; clean -> weights/proxy_logs."
        ),
    )
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--learn-window", type=int, default=10)
    parser.add_argument("--learn-min-correct", type=int, default=8)
    parser.add_argument("--gate-low", type=float, default=0.2)
    parser.add_argument("--gate-high", type=float, default=0.95)
    parser.add_argument("--sample-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only read A/C/T dynamic component caches; the gate still requires proxy fold logs.",
    )
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


def default_dynamic_cache_root(mode: str) -> Path:
    if mode == "noise":
        return NOISE_DYNAMIC_CACHE_ROOT
    if mode == "clean":
        return CLEAN_DYNAMIC_CACHE_ROOT
    raise ValueError(f"Unsupported mode: {mode}")


def default_proxy_log_root(mode: str) -> Path:
    if mode == "noise":
        return NOISE_PROXY_LOG_ROOT
    if mode == "clean":
        return CLEAN_PROXY_LOG_ROOT
    raise ValueError(f"Unsupported mode: {mode}")


def dynamic_cache_dir(
    cache_root: str | Path,
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> Path:
    return Path(cache_root) / dataset_name / proxy_model / str(int(seed)) / str(int(epochs))


def resolve_existing_proxy_log_path(args: argparse.Namespace, epochs: int) -> Path:
    root = Path(args.proxy_log) if args.proxy_log is not None else default_proxy_log_root(args.mode)
    path = resolve_proxy_log_path(
        str(root),
        args.dataset,
        seed=args.seed,
        proxy_model=args.proxy_model,
        max_epoch=epochs,
    )
    if not path.exists():
        raise FileNotFoundError(f"No proxy log path found for mode={args.mode}: {path}")
    return path


def load_folds(args: argparse.Namespace, epochs: int, clean_n: int) -> list[FoldLogData]:
    proxy_log_path = resolve_existing_proxy_log_path(args, epochs)
    folds, labels_from_logs = load_cv_fold_logs(proxy_log_path, args.dataset, args.data_root)
    if labels_from_logs.shape[0] != clean_n:
        raise ValueError("proxy log labels and dataset labels have different lengths.")
    print(f"[proxy] loaded {args.mode} logs: {proxy_log_path}")
    return folds


def load_dynamic_component_cache(
    cache_dir: Path,
    component_name: str,
    expected_labels: np.ndarray,
) -> DynamicComponentResult | None:
    path = cache_dir / f"{component_name}.npz"
    if not path.is_file():
        return None

    required = {"labels", "raw_foldwise", "fold_normalized", "aggregated", "final_normalized"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(set(data.files)):
                return None
            labels = np.asarray(data["labels"], dtype=np.int64)
            if not np.array_equal(labels, expected_labels.astype(np.int64, copy=False)):
                return None
            raw_foldwise = np.asarray(data["raw_foldwise"], dtype=np.float32)
            fold_normalized = np.asarray(data["fold_normalized"], dtype=np.float32)
            aggregated = np.asarray(data["aggregated"], dtype=np.float32)
            final_normalized = np.asarray(data["final_normalized"], dtype=np.float32)
    except Exception:
        return None

    num_samples = expected_labels.shape[0]
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


def load_component_caches(
    cache_dir: Path,
    expected_labels: np.ndarray,
) -> dict[str, DynamicComponentResult] | None:
    results: dict[str, DynamicComponentResult] = {}
    for component_name in ("A", "C", "T"):
        result = load_dynamic_component_cache(cache_dir, component_name, expected_labels)
        if result is None:
            return None
        results[component_name] = result
    return results


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    exp_shifted = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = np.sum(exp_shifted, axis=2, keepdims=True)
    denom = np.where(denom > EPS, denom, 1.0)
    return (exp_shifted / denom).astype(np.float64)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    max_logits = np.max(logits, axis=2, keepdims=True)
    shifted = logits - max_logits
    logsumexp = np.log(np.sum(np.exp(np.clip(shifted, -50.0, 50.0)), axis=2) + EPS)
    logsumexp = logsumexp + max_logits.squeeze(2)

    true_logits = np.take_along_axis(logits, labels.reshape(1, -1, 1), axis=2).squeeze(2)
    return (logsumexp - true_logits).astype(np.float64)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    safe = np.nan_to_num(values, nan=np.nanmax(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 1.0)
    n = safe.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    order = np.argsort(safe, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = safe[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    return (ranks / float(n - 1)).astype(np.float64)


def compute_learning_window(correct: np.ndarray, window: int, min_correct: int) -> tuple[bool, int]:
    correct_i = np.asarray(correct, dtype=np.bool_)
    num_epochs = int(correct_i.shape[0])
    if num_epochs == 0:
        return False, num_epochs
    if window <= 1:
        learned_epochs = np.flatnonzero(correct_i)
        if learned_epochs.size == 0:
            return False, num_epochs
        return True, int(learned_epochs[0])
    if num_epochs < window:
        return False, num_epochs

    counts = np.convolve(correct_i.astype(np.int32), np.ones(window, dtype=np.int32), mode="valid")
    hit = np.flatnonzero(counts >= int(min_correct))
    if hit.size == 0:
        return False, num_epochs
    # The condition becomes observable at the end of the first satisfied window.
    return True, int(hit[0] + window - 1)


def compute_gate_risk_metrics(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
    *,
    learn_window: int,
    learn_min_correct: int,
    sample_chunk_size: int,
) -> dict[str, np.ndarray]:
    labels_for_metric = np.asarray(labels_for_metric, dtype=np.int64)
    num_samples = int(labels_for_metric.shape[0])

    learned_sum = np.zeros(num_samples, dtype=np.float64)
    learn_time_sum = np.zeros(num_samples, dtype=np.float64)
    forget_freq_sum = np.zeros(num_samples, dtype=np.float64)
    learn_risk_sum = np.zeros(num_samples, dtype=np.float64)
    train_count = np.zeros(num_samples, dtype=np.int64)

    loss_var_sum = np.zeros(num_samples, dtype=np.float64)
    loss_var_count = np.zeros(num_samples, dtype=np.int64)

    val_bias_raw = np.zeros(num_samples, dtype=np.float64)
    val_seen = np.zeros(num_samples, dtype=np.int64)

    for fold in tqdm(folds, desc="compute noise gate", unit="fold"):
        train_idx = fold.train_indices.astype(np.int64)
        val_idx = fold.val_indices.astype(np.int64)

        y_train = labels_for_metric[train_idx]
        train_logits = np.asarray(fold.train_logits, dtype=np.float64)
        train_preds = np.argmax(train_logits, axis=2).astype(np.int64)
        correct = train_preds == y_train.reshape(1, -1)
        num_epochs = int(correct.shape[0])
        _, mid_idx, late_idx = resolve_epoch_windows(num_epochs)
        mid_late_idx = np.concatenate([mid_idx, late_idx]).astype(np.int64)

        for local_j, global_i in enumerate(train_idx):
            learned, learn_epoch = compute_learning_window(
                correct[:, local_j],
                window=learn_window,
                min_correct=learn_min_correct,
            )
            if learned:
                learn_time = float(learn_epoch - learn_window + 1) / float(max(1, num_epochs - learn_window))
                learn_time = float(np.clip(learn_time, 0.0, 1.0))
                remaining = correct[learn_epoch + 1 :, local_j]
                if remaining.size == 0:
                    forget_freq = 1.0
                else:
                    forget_freq = float(np.mean(~remaining))
                learn_risk = 1.0 - (1.0 - learn_time) * (1.0 - forget_freq)
                learn_risk = float(np.clip(learn_risk, 0.0, 1.0))
            else:
                learn_time = 1.0
                forget_freq = 1.0
                learn_risk = 1.0

            learned_sum[global_i] += float(learned)
            learn_time_sum[global_i] += learn_time
            forget_freq_sum[global_i] += forget_freq
            learn_risk_sum[global_i] += learn_risk
            train_count[global_i] += 1

        # Mid-late train loss variance, computed in sample chunks to avoid large temporary arrays.
        for start in range(0, train_idx.shape[0], sample_chunk_size):
            end = min(start + sample_chunk_size, train_idx.shape[0])
            chunk_indices = train_idx[start:end]
            losses = cross_entropy_from_logits(
                train_logits[mid_late_idx, start:end, :],
                y_train[start:end],
            )
            loss_var_sum[chunk_indices] += np.var(losses, axis=0)
            loss_var_count[chunk_indices] += 1

        # Validation-view maximum average positive margin to any non-label class.
        val_logits = np.asarray(fold.val_logits, dtype=np.float64)
        y_val = labels_for_metric[val_idx]
        val_epochs = int(val_logits.shape[0])
        _, val_mid_idx, val_late_idx = resolve_epoch_windows(val_epochs)
        val_mid_late_idx = np.concatenate([val_mid_idx, val_late_idx]).astype(np.int64)
        for start in range(0, val_idx.shape[0], sample_chunk_size):
            end = min(start + sample_chunk_size, val_idx.shape[0])
            chunk_global = val_idx[start:end]
            chunk_labels = y_val[start:end]
            chunk_logits = val_logits[val_mid_late_idx, start:end, :]
            chunk_probs = softmax_logits(chunk_logits)

            for local_j, global_i in enumerate(chunk_global):
                label = int(chunk_labels[local_j])
                probs_j = chunk_probs[:, local_j, :]
                label_probs = probs_j[:, label]
                positive_margins = np.maximum(probs_j - label_probs[:, None], 0.0)
                positive_margins[:, label] = 0.0
                class_scores = np.mean(positive_margins, axis=0)
                class_scores[label] = 0.0
                val_bias_raw[global_i] = float(np.max(class_scores))
                val_seen[global_i] += 1

    if np.any(train_count == 0):
        missing = np.where(train_count == 0)[0]
        raise ValueError(f"部分样本没有出现在任何 train fold 中: {missing[:10]}")
    if np.any(loss_var_count == 0):
        missing = np.where(loss_var_count == 0)[0]
        raise ValueError(f"部分样本没有 loss variance 统计: {missing[:10]}")
    if not np.all(val_seen == 1):
        bad = np.where(val_seen != 1)[0]
        raise ValueError(f"Validation view should cover each sample exactly once; bad={bad[:10]}")

    learned_rate = learned_sum / train_count
    learn_time = learn_time_sum / train_count
    forget_freq = forget_freq_sum / train_count
    learn_risk_raw = learn_risk_sum / train_count
    loss_var_raw = loss_var_sum / loss_var_count

    learn_risk_rank = rank_percentile(learn_risk_raw)
    loss_var_rank = rank_percentile(loss_var_raw)
    val_bias_rank = rank_percentile(val_bias_raw)
    final_risk = np.minimum.reduce([learn_risk_rank, loss_var_rank, val_bias_rank])

    return {
        "learned_rate": learned_rate.astype(np.float64),
        "learn_time": learn_time.astype(np.float64),
        "forget_freq": forget_freq.astype(np.float64),
        "learn_risk_raw": learn_risk_raw.astype(np.float64),
        "loss_var_raw": loss_var_raw.astype(np.float64),
        "val_bias_raw": val_bias_raw.astype(np.float64),
        "learn_risk_rank": learn_risk_rank.astype(np.float64),
        "loss_var_rank": loss_var_rank.astype(np.float64),
        "val_bias_rank": val_bias_rank.astype(np.float64),
        "final_risk": final_risk.astype(np.float64),
    }


def build_truncated_gate(risk: np.ndarray, low: float, high: float) -> np.ndarray:
    risk = np.asarray(risk, dtype=np.float64)
    if not (0.0 <= low < high <= 1.0):
        raise ValueError(f"gate thresholds must satisfy 0 <= low < high <= 1, got low={low}, high={high}")
    gate = np.ones_like(risk, dtype=np.float64)
    gate[risk >= high] = 0.0
    mid = (risk > low) & (risk < high)
    gate[mid] = 1.0 - (risk[mid] - low) / (high - low)
    return np.clip(gate, 0.0, 1.0)


def compute_dynamic_components(
    folds: list[FoldLogData],
    labels_for_metric: np.ndarray,
) -> dict[str, DynamicComponentResult]:
    print("\nComputing A/C/T dynamic components with current mode labels...")
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
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    a = component_results["A"].final_normalized.astype(np.float64)
    c = component_results["C"].final_normalized.astype(np.float64)
    t = component_results["T"].final_normalized.astype(np.float64)

    for name, values in {"A": a, "C": c, "T": t}.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN/inf values.")

    original_raw = (a + c + t) / 3.0

    a_gate_after_norm = split_gate_positive_part(a, gate)
    c_gate_after_norm = split_gate_positive_part(c, gate)
    t_gate_after_norm = split_gate_positive_part(t, gate)
    gated_raw = (a_gate_after_norm + c_gate_after_norm + t_gate_after_norm) / 3.0

    original_target = standard_zscore(original_raw).astype(np.float64)
    gated_target = standard_zscore(gated_raw).astype(np.float64)

    component_values = {
        "A normalized positive-part gated": a_gate_after_norm,
        "C normalized positive-part gated": c_gate_after_norm,
        "T normalized positive-part gated": t_gate_after_norm,
    }
    return original_target, gated_target, component_values




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


def print_group_stats(
    metric_name: str,
    values: np.ndarray,
    reference_mask: np.ndarray,
    *,
    positive_label: str = "noisy",
    negative_label: str = "clean",
) -> None:
    negative_mean, negative_max, negative_min, negative_n = summarize(values, ~reference_mask)
    positive_mean, positive_max, positive_min, positive_n = summarize(values, reference_mask)

    print(f"\n[{metric_name}]")
    print(f"  {negative_label}: mean={negative_mean:.6f}, max={negative_max:.6f}, min={negative_min:.6f}, n={negative_n}")
    print(f"  {positive_label}: mean={positive_mean:.6f}, max={positive_max:.6f}, min={positive_min:.6f}, n={positive_n}")


def plot_pseudo_label_histogram(
    pseudo_labels: np.ndarray,
    reference_mask: np.ndarray,
    output_path: Path,
    bins: int,
    *,
    positive_label: str,
    negative_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    negative_values = np.asarray(pseudo_labels, dtype=np.float64)[~reference_mask]
    positive_values = np.asarray(pseudo_labels, dtype=np.float64)[reference_mask]
    negative_values = negative_values[np.isfinite(negative_values)]
    positive_values = positive_values[np.isfinite(positive_values)]

    all_values = np.concatenate([negative_values, positive_values], axis=0)
    if all_values.size == 0:
        raise ValueError("No finite pseudo-label values to plot.")

    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if abs(value_max - value_min) < 1e-12:
        value_min -= 1.0
        value_max += 1.0

    hist_bins = np.linspace(value_min, value_max, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(negative_values, bins=hist_bins, density=True, alpha=0.55, color="blue", label=negative_label)
    plt.hist(positive_values, bins=hist_bins, density=True, alpha=0.55, color="red", label=positive_label)
    plt.xlabel("Gated dynamic pseudo-label")
    plt.ylabel("Density")
    plt.title("Pseudo-label distribution after noise risk gating")
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

    args.mode = args.mode.strip().lower()

    dataset = build_train_dataset(args.dataset, args.data_root)
    clean_labels = extract_labels(dataset)
    noisy_labels, reference_mask = build_noisy_labels(
        clean_labels=clean_labels,
        dataset_name=args.dataset,
        seed=args.seed,
        noise_root=args.noise_root,
    )
    labels_for_metric = noisy_labels if args.mode == "noise" else clean_labels

    positive_label = "noisy" if args.mode == "noise" else "marked-by-noise-list"
    negative_label = "clean" if args.mode == "noise" else "unmarked"

    dynamic_root = Path(args.dynamic_cache_root) if args.dynamic_cache_root is not None else default_dynamic_cache_root(args.mode)
    cache_dir = dynamic_cache_dir(
        dynamic_root,
        args.dataset,
        args.proxy_model,
        args.seed,
        resolved_proxy_epochs,
    )

    print("=== Noise gate risk diagnostic ===")
    print(f"mode={args.mode}, dataset={args.dataset}, seed={args.seed}, proxy_model={args.proxy_model}, proxy_epochs={resolved_proxy_epochs}")
    print(f"dynamic_cache={cache_dir}")
    print(f"proxy_log_root={Path(args.proxy_log) if args.proxy_log is not None else default_proxy_log_root(args.mode)}")
    print(f"num_samples={clean_labels.shape[0]}, reference_marked={int(np.sum(reference_mask))}")
    print("Risk terms: train learnability, mid-late train loss variance, validation stable non-label bias.")
    print(f"Gate thresholds: low={args.gate_low:.3f}, high={args.gate_high:.3f}")
    print("Rule: final risk = min(rank(learn risk), rank(loss-var risk), rank(val-bias risk)).")
    if args.mode == "clean":
        print("Clean mode: metrics use original labels and normal caches/logs; marked group is only the seed's noise-list reference subset.")

    folds = load_folds(args, resolved_proxy_epochs, clean_labels.shape[0])
    risk_metrics = compute_gate_risk_metrics(
        folds,
        labels_for_metric,
        learn_window=args.learn_window,
        learn_min_correct=args.learn_min_correct,
        sample_chunk_size=max(1, int(args.sample_chunk_size)),
    )
    gate = build_truncated_gate(risk_metrics["final_risk"], low=float(args.gate_low), high=float(args.gate_high))

    component_results = load_component_caches(cache_dir, labels_for_metric)
    if component_results is None:
        if args.cache_only:
            raise FileNotFoundError(f"Missing A/C/T dynamic component caches under: {cache_dir}")
        print("[cache] A/C/T MISS; computing components from proxy logs without saving cache.")
        component_results = compute_dynamic_components(folds, labels_for_metric)
    else:
        print("[cache] A/C/T HIT")

    original_target, gated_target, component_values = build_pseudo_labels(
        component_results,
        gate,
    )

    mean_gate = float(np.mean(gate))
    gate_suppression = 1.0 - mean_gate
    print(f"\n[Gate suppression summary]")
    print(f"  mean_gate={mean_gate:.6f}, r_gate=1-mean_gate={gate_suppression:.6f}")

    print_group_stats("LearnedRate(window)", risk_metrics["learned_rate"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("LearnTime(window-normalized)", risk_metrics["learn_time"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("ForgetFrequency", risk_metrics["forget_freq"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("LearnRisk raw", risk_metrics["learn_risk_raw"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("MidLateLossVar raw", risk_metrics["loss_var_raw"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("ValStableOtherBias raw", risk_metrics["val_bias_raw"], reference_mask, positive_label=positive_label, negative_label=negative_label)

    print_group_stats("LearnRisk rank", risk_metrics["learn_risk_rank"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("MidLateLossVar rank", risk_metrics["loss_var_rank"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("ValStableOtherBias rank", risk_metrics["val_bias_rank"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("FinalRisk min-rank", risk_metrics["final_risk"], reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("Gate g", gate, reference_mask, positive_label=positive_label, negative_label=negative_label)

    for name, values in component_values.items():
        print_group_stats(name, values, reference_mask, positive_label=positive_label, negative_label=negative_label)

    print_group_stats("Original pseudo-label", original_target, reference_mask, positive_label=positive_label, negative_label=negative_label)
    print_group_stats("Gated pseudo-label", gated_target, reference_mask, positive_label=positive_label, negative_label=negative_label)

    output_path = Path(__file__).resolve().parent / f"test_gate_pseudolabel_hist_{args.mode}_{args.dataset}_seed{args.seed}.png"
    plot_pseudo_label_histogram(gated_target, reference_mask, output_path, args.bins, positive_label=positive_label, negative_label=negative_label)
    print(f"\nSaved histogram to: {output_path}")


if __name__ == "__main__":
    main()
