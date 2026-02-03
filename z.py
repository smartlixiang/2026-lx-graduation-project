from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.proxy_log_utils import load_proxy_log  # noqa: E402
from utils.score_utils import quantile_minmax  # noqa: E402
from weights.AbsorptionEfficiencyScore import AbsorptionEfficiencyScore  # noqa: E402
from weights.CoverageGainScore import CoverageGainScore  # noqa: E402
from weights.InformativenessScore import InformativenessScore  # noqa: E402
from weights.RiskScore import RiskScore  # noqa: E402


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
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root path.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./debug_z",
        help="Output directory for histogram plots.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
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
    u_norm = quantile_minmax(u_raw.astype(np.float32), q_low=0.01, q_high=0.99)
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

    for name, values in arrays.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins=args.bins, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{name} Histogram")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(raw_arrays["B_raw"], bins=args.bins, color="#55A868", alpha=0.85)
    ax.set_title("B_raw Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_B_raw.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for idx, name in enumerate(["A", "B", "C", "R", "u"]):
        ax = axes[idx]
        ax.hist(arrays[name], bins=args.bins, color="#4C72B0", alpha=0.85)
        ax.set_title(name)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_overview.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for idx, name in enumerate(["A_raw", "B_raw", "C_raw", "R_raw", "u_raw"]):
        ax = axes[idx]
        ax.hist(raw_arrays[name], bins=args.bins, color="#C44E52", alpha=0.85)
        ax.set_title(name)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_overview_raw.png", dpi=150)
    plt.close(fig)

    names = ["A", "B", "C", "R", "u"]
    matrix = np.vstack([arrays[name] for name in names])
    pearson = np.corrcoef(matrix)
    spearman = _spearman_corrcoef(matrix)

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
    _print_corr_table(names, pearson, "Pearson Correlation")
    _print_corr_table(names, spearman, "Spearman Correlation")

    if not args.no_pause:
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
