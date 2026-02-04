from __future__ import annotations

# Example:
# python z.py --cv_log_dir weights/proxy_logs/cifar10/resnet18/0/100 --dataset cifar10 --seed 0 --keep_ratio 0.60 --w_list 1.0,0.25

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BaseDataLoader  # noqa: E402
from utils.proxy_log_utils import load_proxy_log  # noqa: E402
from utils.score_utils import quantile_minmax  # noqa: E402
from weights.AbsorptionEfficiencyScore import AbsorptionEfficiencyScore  # noqa: E402
from weights.CoverageGainScore import CoverageGainScore  # noqa: E402
from weights.InformativenessScore import InformativenessScore  # noqa: E402
from weights.PersistentDifficultyScore import PersistentDifficultyScore  # noqa: E402
from weights.RiskScore import RiskScore  # noqa: E402
from weights.TransferGainScore import TransferGainScore  # noqa: E402


# Key interfaces:
# - main(): computes A/B/C/R/u from proxy log, saves histograms, prints correlation tables.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A/B/C/R/u distributions and correlations.")
    parser.add_argument(
        "--proxy_log",
        type=str,
        default="weights/proxy_logs/cifar10/resnet18/22/100",
        help="Path to proxy training log (.npz) or k-fold log directory.",
    )
    parser.add_argument(
        "--cv_log_dir",
        type=str,
        default=None,
        help="Path to k-fold proxy log directory (for TransferGainScore).",
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root path.")
    parser.add_argument("--seed", type=int, default=0, help="Dataset seed for loader.")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds (for metadata).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./debug_z",
        help="Output directory for histogram plots.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    parser.add_argument(
        "--wT_list",
        type=str,
        default="1.0,0.25",
        help="Comma-separated list of weights for TransferGainScore.",
    )
    parser.add_argument(
        "--w_list",
        type=str,
        default="1.0,0.25",
        help="Comma-separated list of weights for TransferGainScore and PersistentDifficultyScore.",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.60,
        help="Top-k keep ratio for overlap analysis.",
    )
    parser.add_argument("--no_pause", action="store_true", default=True, help="Disable input() pause.")
    return parser.parse_args()


def _to_full(scores: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    full = np.full((n,), np.nan, dtype=np.float32)
    if indices.shape[0] != scores.shape[0]:
        raise ValueError("indices and scores length mismatch in a score result")
    if np.min(indices) < 0 or np.max(indices) >= n:
        raise ValueError("score result indices out of range")
    full[indices.astype(np.int64)] = scores.astype(np.float32)
    return full


def _compute_components(
    proxy_log: Path,
    dataset_name: str,
    data_root: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    proxy_data = load_proxy_log(proxy_log, dataset_name, data_root)
    absorption_res = AbsorptionEfficiencyScore(proxy_log).compute(proxy_logs=proxy_data)
    informativeness_res = InformativenessScore(proxy_log).compute(proxy_logs=proxy_data)
    coverage_res = CoverageGainScore(proxy_log).compute(proxy_logs=proxy_data)
    risk_res = RiskScore(proxy_log).compute(proxy_logs=proxy_data)

    indices = absorption_res.indices
    if (
        not np.array_equal(indices, informativeness_res.indices)
        or not np.array_equal(indices, coverage_res.indices)
        or not np.array_equal(indices, risk_res.indices)
    ):
        raise ValueError("动态指标的 indices 不一致，无法对齐样本。")
    if absorption_res.labels is None:
        raise ValueError("代理训练日志缺少 labels，无法对齐动态分数。")

    n = absorption_res.labels.shape[0]
    a_full = _to_full(absorption_res.scores, absorption_res.indices, n)
    a_raw_full = _to_full(absorption_res.raw_score, absorption_res.indices, n)
    b_full = _to_full(informativeness_res.scores, informativeness_res.indices, n)
    b_raw_full = _to_full(informativeness_res.raw_score, informativeness_res.indices, n)
    c_full = _to_full(coverage_res.scores, coverage_res.indices, n)
    c_raw = getattr(coverage_res, "raw_score", coverage_res.scores)
    c_raw_full = _to_full(c_raw, coverage_res.indices, n)
    r_full = _to_full(risk_res.scores, risk_res.indices, n)
    r_raw_full = _to_full(risk_res.raw_score, risk_res.indices, n)

    if (
        np.any(np.isnan(a_full))
        or np.any(np.isnan(a_raw_full))
        or np.any(np.isnan(b_full))
        or np.any(np.isnan(b_raw_full))
        or np.any(np.isnan(c_full))
        or np.any(np.isnan(c_raw_full))
        or np.any(np.isnan(r_full))
        or np.any(np.isnan(r_raw_full))
    ):
        raise RuntimeError("Failed to align some dynamic scores to full index space (NaNs found).")

    u_raw = a_full + b_full + c_full - r_full
    u_norm = normalize_u_raw(u_raw.astype(np.float32))
    components = {
        "A": a_full.astype(np.float32),
        "B": b_full.astype(np.float32),
        "C": c_full.astype(np.float32),
        "R": r_full.astype(np.float32),
        "u": u_norm.astype(np.float32),
    }
    raw_components = {
        "A_raw": a_raw_full.astype(np.float32),
        "B_raw": b_raw_full.astype(np.float32),
        "C_raw": c_raw_full.astype(np.float32),
        "R_raw": r_raw_full.astype(np.float32),
        "u_raw": u_raw.astype(np.float32),
    }
    return components, raw_components, absorption_res.labels.astype(np.int64)


def normalize_u_raw(u_raw: np.ndarray) -> np.ndarray:
    return quantile_minmax(u_raw.astype(np.float32), q_low=0.002, q_high=0.998)


def _load_train_dataset(dataset_name: str, data_root: str, seed: int):
    loader = BaseDataLoader(
        dataset_name,
        data_path=Path(data_root),
        batch_size=1,
        num_workers=0,
        val_split=0.0,
        seed=seed,
        augment=False,
        normalize=False,
    )
    train_loader, _, _ = loader.load()
    return train_loader.dataset


def _plot_compare_hist(
    out_path: Path,
    base_values: np.ndarray,
    with_t_values: np.ndarray,
    bins: int,
    title: str,
    label_base: str,
    label_with_t: str,
    color_base: str = "#4C72B0",
    color_with: str = "#55A868",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(base_values, bins=bins, color=color_base, alpha=0.6, label=label_base)
    ax.hist(with_t_values, bins=bins, color=color_with, alpha=0.6, label=label_with_t)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_risk_hist(
    out_path: Path,
    values: np.ndarray,
    bins: int,
    title_prefix: str,
) -> None:
    zeros = int(np.sum(values == 0.0))
    nonzero_values = values[values > 0.0]
    nonzero_count = int(nonzero_values.size)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(nonzero_values, bins=bins, color="#C44E52", alpha=0.85)
    ax.set_title(f"{title_prefix} (nonzero only) | zeros={zeros} | nonzero={nonzero_count}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _print_quantiles(values: np.ndarray, name: str) -> None:
    quantiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    percentiles = np.percentile(values, quantiles)
    print(f"\n{name} quantiles:")
    for q, v in zip(quantiles, percentiles, strict=True):
        print(f"  p{q:02d}: {v: .6f}")


def _print_zero_one_counts(values: np.ndarray, name: str) -> None:
    zero_count = int(np.sum(values == 0.0))
    one_count = int(np.sum(values == 1.0))
    print(f"{name} exact 0 count: {zero_count}")
    print(f"{name} exact 1 count: {one_count}")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_vals = values[order]
    n = values.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i: j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corrcoef(matrix: np.ndarray) -> np.ndarray:
    ranked = np.vstack([_rankdata(matrix[i]) for i in range(matrix.shape[0])])
    return np.corrcoef(ranked)


def _print_corr_table(names: list[str], corr: np.ndarray, title: str) -> None:
    header = "\t".join([""] + names)
    print(f"\n{title}")
    print(header)
    for name, row in zip(names, corr, strict=True):
        values = "\t".join(f"{val: .4f}" for val in row)
        print(f"{name}\t{values}")


def _format_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _parse_weight_list(raw: str) -> list[float]:
    weights = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not weights:
        raise ValueError("weight list is empty.")
    return weights


def _topk_indices(values: np.ndarray, keep_ratio: float) -> np.ndarray:
    if not 0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be within (0,1].")
    k = int(np.ceil(values.size * keep_ratio))
    order = np.argsort(values)[::-1]
    return order[:k]


def _overlap_ratio(set_a: set[int], set_b: set[int]) -> float:
    if not set_a:
        return 0.0
    return len(set_a & set_b) / float(len(set_a))


def _jaccard_ratio(set_a: set[int], set_b: set[int]) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / float(len(union))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy_log = Path(args.proxy_log)
    if not proxy_log.exists():
        raise FileNotFoundError(f"Proxy log not found: {proxy_log}")

    arrays, raw_arrays, labels = _compute_components(proxy_log, args.dataset, args.data_root)
    num_samples = labels.shape[0]
    for key, values in arrays.items():
        if values.shape[0] != num_samples:
            raise ValueError(f"{key} length does not match labels length.")
    for key, values in raw_arrays.items():
        if values.shape[0] != num_samples:
            raise ValueError(f"{key} length does not match labels length.")

    cv_log_dir = Path(args.cv_log_dir) if args.cv_log_dir is not None else proxy_log
    if not cv_log_dir.exists():
        raise FileNotFoundError(f"cv_log_dir not found: {cv_log_dir}")

    dataset = _load_train_dataset(args.dataset, args.data_root, args.seed)
    t_result = TransferGainScore().compute(cv_log_dir, dataset)
    t_scores = t_result["score"].astype(np.float32)
    if t_scores.shape[0] != num_samples:
        raise ValueError("TransferGainScore length does not match labels length.")
    v_result = PersistentDifficultyScore().compute(cv_log_dir, dataset)
    v_scores = v_result["score"].astype(np.float32)
    if v_scores.shape[0] != num_samples:
        raise ValueError("PersistentDifficultyScore length does not match labels length.")

    u_raw_base = raw_arrays["u_raw"].astype(np.float32)
    u_norm_base = normalize_u_raw(u_raw_base)

    if args.w_list == "1.0,0.25" and args.wT_list != "1.0,0.25":
        weight_list = _parse_weight_list(args.wT_list)
    else:
        weight_list = _parse_weight_list(args.w_list)

    keep_tag = _format_tag(args.keep_ratio)

    for name, values in arrays.items():
        if name == "R":
            _plot_risk_hist(
                out_dir / f"hist_{name}_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png",
                values,
                args.bins,
                f"{name} Histogram",
            )
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins=args.bins, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{name} Histogram")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"hist_{name}_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png",
            dpi=150,
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(raw_arrays["B_raw"], bins=args.bins, color="#55A868", alpha=0.85)
    ax.set_title("B_raw Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"hist_B_raw_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png",
        dpi=150,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(t_scores, bins=args.bins, color="#8172B2", alpha=0.85)
    ax.set_title("T (TransferGainScore) Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"hist_T_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png",
        dpi=150,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(v_scores, bins=args.bins, color="#CCB974", alpha=0.85)
    ax.set_title("V (PersistentDifficultyScore) Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"hist_V_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png",
        dpi=150,
    )
    plt.close(fig)

    for w_t in weight_list:
        tag_wt = _format_tag(w_t)
        u_raw_with_t = u_raw_base + w_t * t_scores
        u_norm_with_t = normalize_u_raw(u_raw_with_t)
        _plot_compare_hist(
            out_dir / f"hist_u_raw_base_vs_T_seed{args.seed}_wT{tag_wt}_wV0p00_keep{keep_tag}.png",
            u_raw_base,
            u_raw_with_t,
            args.bins,
            f"u_raw_base vs u_raw_withT (w_T={w_t})",
            "u_raw_base",
            f"u_raw_withT (w_T={w_t})",
        )
        _plot_compare_hist(
            out_dir / f"hist_u_norm_base_vs_T_seed{args.seed}_wT{tag_wt}_wV0p00_keep{keep_tag}.png",
            u_norm_base,
            u_norm_with_t,
            args.bins,
            f"u_norm_base vs u_norm_withT (w_T={w_t})",
            "u_norm_base",
            f"u_norm_withT (w_T={w_t})",
        )

    for w_v in weight_list:
        tag_wv = _format_tag(w_v)
        u_raw_with_v = u_raw_base + w_v * v_scores
        u_norm_with_v = normalize_u_raw(u_raw_with_v)
        _plot_compare_hist(
            out_dir / f"hist_u_raw_base_vs_V_seed{args.seed}_wT0p00_wV{tag_wv}_keep{keep_tag}.png",
            u_raw_base,
            u_raw_with_v,
            args.bins,
            f"u_raw_base vs u_raw_withV (w_V={w_v})",
            "u_raw_base",
            f"u_raw_withV (w_V={w_v})",
        )
        _plot_compare_hist(
            out_dir / f"hist_u_norm_base_vs_V_seed{args.seed}_wT0p00_wV{tag_wv}_keep{keep_tag}.png",
            u_norm_base,
            u_norm_with_v,
            args.bins,
            f"u_norm_base vs u_norm_withV (w_V={w_v})",
            "u_norm_base",
            f"u_norm_withV (w_V={w_v})",
        )

    for w_t in weight_list:
        for w_v in weight_list:
            tag_wt = _format_tag(w_t)
            tag_wv = _format_tag(w_v)
            u_raw_with_tv = u_raw_base + w_t * t_scores + w_v * v_scores
            u_norm_with_tv = normalize_u_raw(u_raw_with_tv)
            _plot_compare_hist(
                out_dir
                / f"hist_u_raw_base_vs_TV_seed{args.seed}_wT{tag_wt}_wV{tag_wv}_keep{keep_tag}.png",
                u_raw_base,
                u_raw_with_tv,
                args.bins,
                f"u_raw_base vs u_raw_withT+V (w_T={w_t}, w_V={w_v})",
                "u_raw_base",
                f"u_raw_withT+V (w_T={w_t}, w_V={w_v})",
            )
            _plot_compare_hist(
                out_dir
                / f"hist_u_norm_base_vs_TV_seed{args.seed}_wT{tag_wt}_wV{tag_wv}_keep{keep_tag}.png",
                u_norm_base,
                u_norm_with_tv,
                args.bins,
                f"u_norm_base vs u_norm_withT+V (w_T={w_t}, w_V={w_v})",
                "u_norm_base",
                f"u_norm_withT+V (w_T={w_t}, w_V={w_v})",
            )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for idx, name in enumerate(["A", "B", "C", "R", "u"]):
        ax = axes[idx]
        if name == "R":
            values = arrays[name]
            nonzero_values = values[values > 0.0]
            zeros = int(np.sum(values == 0.0))
            ax.hist(nonzero_values, bins=args.bins, color="#4C72B0", alpha=0.85)
            ax.set_title(f"{name} (zeros={zeros})")
        else:
            ax.hist(arrays[name], bins=args.bins, color="#4C72B0", alpha=0.85)
            ax.set_title(name)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"hist_overview_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for idx, name in enumerate(["A_raw", "B_raw", "C_raw", "R_raw", "u_raw"]):
        ax = axes[idx]
        if name == "R_raw":
            values = raw_arrays[name]
            nonzero_values = values[values > 0.0]
            zeros = int(np.sum(values == 0.0))
            ax.hist(nonzero_values, bins=args.bins, color="#C44E52", alpha=0.85)
            ax.set_title(f"{name} (zeros={zeros})")
        else:
            ax.hist(raw_arrays[name], bins=args.bins, color="#C44E52", alpha=0.85)
            ax.set_title(name)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"hist_overview_raw_seed{args.seed}_wT0p00_wV0p00_keep{keep_tag}.png", dpi=150)
    plt.close(fig)

    corr_weight = 1.0 if 1.0 in weight_list else weight_list[0]
    u_raw_with_tv = u_raw_base + corr_weight * t_scores + corr_weight * v_scores
    u_norm_with_tv = normalize_u_raw(u_raw_with_tv)

    corr_names = [
        "A",
        "B",
        "C",
        "R",
        "T",
        "V",
        "u_raw_base",
        "u_raw_TV",
        "u_norm_base",
        "u_norm_TV",
    ]
    corr_matrix = np.vstack(
        [
            arrays["A"],
            arrays["B"],
            arrays["C"],
            arrays["R"],
            t_scores,
            v_scores,
            u_raw_base,
            u_raw_with_tv,
            u_norm_base,
            u_norm_with_tv,
        ]
    )
    pearson = np.corrcoef(corr_matrix)
    spearman = _spearman_corrcoef(corr_matrix)

    corr_csv_path = out_dir / (
        f"corr_seed{args.seed}_wT{_format_tag(corr_weight)}_wV{_format_tag(corr_weight)}_keep{keep_tag}.csv"
    )
    with corr_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([""] + corr_names)
        for name, row in zip(corr_names, spearman, strict=True):
            writer.writerow([name] + [f"{val:.6f}" for val in row])

    print(f"Loaded proxy log: {proxy_log}")
    print(f"Saved histogram plots to: {out_dir}")
    _print_quantiles(arrays["A"], "A_score")
    _print_quantiles(raw_arrays["A_raw"], "A_raw")
    _print_quantiles(raw_arrays["B_raw"], "B_raw")
    _print_quantiles(arrays["R"], "R_score")
    _print_quantiles(raw_arrays["R_raw"], "R_raw")
    _print_zero_one_counts(arrays["A"], "A_score")
    _print_zero_one_counts(arrays["R"], "R_score")
    neg_ratio = float(np.mean(raw_arrays["u_raw"] < 0))
    zero_ratio_r = float(np.mean(arrays["R"] == 0))
    nonzero_ratio_r = float(np.mean(arrays["R"] > 0))
    print(f"u_raw neg_ratio: {neg_ratio:.6f}")
    print(f"R zero_ratio: {zero_ratio_r:.6f}")
    print(f"R nonzero_ratio: {nonzero_ratio_r:.6f}")
    _print_corr_table(corr_names, pearson, "Pearson Correlation")
    _print_corr_table(corr_names, spearman, "Spearman Correlation")

    base_set = set(_topk_indices(u_raw_base, args.keep_ratio).tolist())
    w11 = 1.0 if 1.0 in weight_list else weight_list[0]
    w025 = 0.25 if 0.25 in weight_list else weight_list[0]
    u_raw_11 = u_raw_base + w11 * t_scores + w11 * v_scores
    u_raw_025 = u_raw_base + w025 * t_scores + w025 * v_scores
    set_11 = set(_topk_indices(u_raw_11, args.keep_ratio).tolist())
    set_025 = set(_topk_indices(u_raw_025, args.keep_ratio).tolist())

    comparisons = [
        ("base", "w11", base_set, set_11),
        ("base", "w025", base_set, set_025),
        ("w11", "w025", set_11, set_025),
    ]
    overlap_records = []
    print("\nTop-k overlap analysis:")
    for name_a, name_b, set_a, set_b in comparisons:
        overlap = _overlap_ratio(set_a, set_b)
        jaccard = _jaccard_ratio(set_a, set_b)
        print(f"  {name_a} vs {name_b}: overlap={overlap:.4f}, jaccard={jaccard:.4f}")
        overlap_records.append(
            {
                "a": name_a,
                "b": name_b,
                "overlap": overlap,
                "jaccard": jaccard,
                "keep_ratio": args.keep_ratio,
                "w11": w11,
                "w025": w025,
            }
        )

    overlap_json = out_dir / (
        f"overlap_seed{args.seed}_keep{keep_tag}_w11{_format_tag(w11)}_w025{_format_tag(w025)}.json"
    )
    overlap_csv = out_dir / (
        f"overlap_seed{args.seed}_keep{keep_tag}_w11{_format_tag(w11)}_w025{_format_tag(w025)}.csv"
    )
    overlap_json.write_text(json.dumps(overlap_records, indent=2), encoding="utf-8")
    with overlap_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["a", "b", "overlap", "jaccard", "keep_ratio", "w11", "w025"])
        for record in overlap_records:
            writer.writerow(
                [
                    record["a"],
                    record["b"],
                    f"{record['overlap']:.6f}",
                    f"{record['jaccard']:.6f}",
                    f"{record['keep_ratio']:.2f}",
                    f"{record['w11']:.2f}",
                    f"{record['w025']:.2f}",
                ]
            )

    if not args.no_pause:
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
