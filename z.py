"""Generate utility label distributions from proxy training logs.

This script computes two variants of utility labels:
1) ForgettingScore + MarginScore + EarlyLossScore
2) StabilityScore + MarginScore + EarlyLossScore
Then plots histogram distributions and saves images.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weights import (
    BoundaryInfoScore,
    EarlyLossScore,
    ForgettingScore,
    MarginScore,
    StabilityScore,
)

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
if _SCIPY_AVAILABLE:
    from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute utility label scores and plot histograms."
    )
    parser.add_argument(
        "--proxy-log",
        type=str,
        default="weights/proxy_logs/22/cifar10_resnet18_2026_01_20_11_42.npz",
        help="Path to proxy training log (.npz).",
    )
    parser.add_argument("--early-epochs", type=int, default=None)
    parser.add_argument(
        "--margin-start-ratio",
        type=float,
        default=0.2,
        help="Start ratio for MarginScore temporal averaging.",
    )
    parser.add_argument(
        "--margin-tau",
        type=float,
        default=1.0,
        help="Temperature for MarginScore softmax.",
    )
    parser.add_argument(
        "--margin-delta",
        dest="margin_start_ratio",
        type=float,
        default=0.2,
        help="(deprecated) Same as --margin-start-ratio for MarginScore.",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_weights_debug",
        help="Directory to save histogram images.",
    )
    parser.add_argument(
        "--topk-fracs",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated fractions for top-k overlap diagnostics.",
    )
    parser.add_argument(
        "--scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to output scatter plots for score pairs.",
    )
    parser.add_argument(
        "--decompose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to output utility stability component diagnostics.",
    )
    parser.add_argument(
        "--early-curves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to plot early window loss curves.",
    )
    parser.add_argument(
        "--n-curves",
        type=int,
        default=20,
        help="Number of top/bottom curves to plot for early-curves.",
    )
    return parser.parse_args()


def _ensure_same_indices(results: dict[str, np.ndarray]) -> None:
    iterator = iter(results.items())
    _, first = next(iterator)
    for name, arr in iterator:
        if not np.array_equal(first, arr):
            raise ValueError(
                "Dynamic score indices mismatch between results. "
                f"Expected same indices, got mismatch at {name}."
            )


def _plot_hist(values: np.ndarray, title: str, xlabel: str, output_path: Path, bins: int) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def _parse_topk_fracs(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return [frac for frac in values if frac > 0]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty_like(sorted_vals, dtype=float)
    n = len(values)
    idx = 0
    while idx < n:
        next_idx = idx
        while next_idx + 1 < n and sorted_vals[next_idx + 1] == sorted_vals[idx]:
            next_idx += 1
        rank = 0.5 * (idx + next_idx) + 1.0
        ranks[idx: next_idx + 1] = rank
        idx = next_idx + 1
    result = np.empty_like(ranks)
    result[order] = ranks
    return result


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) <= 1:
        return float("nan")
    if _SCIPY_AVAILABLE:
        return float(spearmanr(a, b).correlation)
    ranks_a = _rankdata(a)
    ranks_b = _rankdata(b)
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = _spearman_corr(x, y)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=8, alpha=0.2, color="#4C78A8", edgecolor="none")
    plt.title(f"{title}\nPearson: {pearson:.3f} | Spearman: {spearman:.3f}")
    plt.xlabel("Score A")
    plt.ylabel("Score B")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def _topk_overlap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    fracs: list[float],
    label: str,
) -> list[float]:
    n = scores_a.shape[0]
    overlaps: list[float] = []
    for frac in fracs:
        k = max(1, int(frac * n))
        top_a = np.argsort(-scores_a)[:k]
        top_b = np.argsort(-scores_b)[:k]
        overlap = len(set(top_a).intersection(top_b)) / k
        overlaps.append(overlap)
        print(f"Top-{frac:.0%} overlap {label}: {overlap:.3f}")
    return overlaps


def _plot_overlap_curve(
    fracs: list[float],
    series: dict[str, list[float]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    for name, values in series.items():
        plt.plot(fracs, values, marker="o", label=name)
    plt.xlabel("Top-k fraction")
    plt.ylabel("Overlap ratio")
    plt.title("Top-k overlap diagnostics")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
    plt.close()


def _describe_component(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def _format_component_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name:<12} mean={stats['mean']:.4f} "
        f"p10={stats['p10']:.4f} p50={stats['p50']:.4f} p90={stats['p90']:.4f}"
    )


def _print_decompose_stats(
    output_dir: Path,
    utility_scores: np.ndarray,
    stability_scores: np.ndarray,
    margin_scores: np.ndarray,
    early_scores: np.ndarray,
    top_frac: float = 0.1,
) -> None:
    n = utility_scores.shape[0]
    k = max(1, int(top_frac * n))
    order = np.argsort(-utility_scores)
    top_idx = order[:k]
    bottom_idx = order[-k:]

    components = {
        "stability": stability_scores,
        "margin": margin_scores,
        "early": early_scores,
    }
    lines = []
    lines.append(f"Top {top_frac:.0%} (k={k}) utility_stability group stats:")
    for name, values in components.items():
        stats = _describe_component(values[top_idx])
        line = _format_component_stats(name, stats)
        lines.append(line)
        print(line)
    lines.append("")
    lines.append(f"Bottom {top_frac:.0%} (k={k}) utility_stability group stats:")
    for name, values in components.items():
        stats = _describe_component(values[bottom_idx])
        line = _format_component_stats(name, stats)
        lines.append(line)
        print(line)
    lines.append("")
    lines.append("Contribution delta (mean_top - mean_bottom):")
    for name, values in components.items():
        mean_top = float(np.mean(values[top_idx]))
        mean_bottom = float(np.mean(values[bottom_idx]))
        delta = mean_top - mean_bottom
        line = f"{name:<12} delta_mean={delta:.4f}"
        lines.append(line)
        print(line)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decompose_stats.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path}")


def _plot_early_curves(
    output_dir: Path,
    loss: np.ndarray,
    early_scores: np.ndarray,
    early_epochs: int,
    n_curves: int,
) -> None:
    early_epochs = max(1, min(early_epochs, loss.shape[0]))
    curves = np.log1p(loss[:early_epochs])
    order = np.argsort(-early_scores)
    top_idx = order[:n_curves]
    bottom_idx = order[-n_curves:]

    def _plot_group(indices: np.ndarray, title: str, output_path: Path) -> None:
        plt.figure(figsize=(7, 4))
        mean_curve = curves[:, indices].mean(axis=1)
        plt.plot(mean_curve, color="#F58518", linewidth=2, label="mean")
        for idx in indices:
            plt.plot(curves[:, idx], color="#4C78A8", alpha=0.2, linewidth=1)
        plt.title(title)
        plt.xlabel("Early epoch")
        plt.ylabel("log1p(loss)")
        plt.legend()
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)
        print(f"Saved: {output_path}")
        plt.close()

    _plot_group(top_idx, "Early loss curves (Top)", output_dir / "early_curves_top.png")
    _plot_group(
        bottom_idx, "Early loss curves (Bottom)", output_dir / "early_curves_bottom.png"
    )


def main() -> None:
    args = parse_args()
    proxy_log = Path(args.proxy_log)
    if not proxy_log.exists():
        raise FileNotFoundError(f"Proxy log not found: {proxy_log}")

    with np.load(proxy_log) as data:
        has_logits = "logits" in data and "labels" in data

    forgetting_result = ForgettingScore(proxy_log).compute()
    margin_result = MarginScore(
        proxy_log,
        tau_m=args.margin_tau,
        start_ratio=args.margin_start_ratio,
    ).compute()
    early_result = EarlyLossScore(proxy_log, early_epochs=args.early_epochs).compute()
    stability_result = StabilityScore(proxy_log).compute()
    boundary_result = None
    if has_logits:
        boundary_result = BoundaryInfoScore(proxy_log).compute()
    else:
        print(
            "Skipping BoundaryInfoScore: proxy log missing required logits/labels arrays."
        )

    _ensure_same_indices(
        {
            "forgetting": forgetting_result.indices,
            "margin": margin_result.indices,
            "early": early_result.indices,
            "stability": stability_result.indices,
        }
    )
    if boundary_result is not None:
        _ensure_same_indices(
            {
                "boundary": boundary_result.indices,
                "early": early_result.indices,
                "stability": stability_result.indices,
            }
        )

    utility_forgetting = (
        forgetting_result.scores + margin_result.scores + early_result.scores
    ) / 3.0
    utility_stability = (
        stability_result.scores + margin_result.scores + early_result.scores
    ) / 3.0
    utility_boundary = None
    if boundary_result is not None:
        utility_boundary = (
            boundary_result.scores + early_result.scores + stability_result.scores
        ) / 3.0

    output_dir = Path(args.output_dir)
    _plot_hist(
        utility_forgetting,
        "Utility label distribution (Forgetting + Margin + EarlyLoss)",
        "Utility label score",
        output_dir / "utility_hist_forgetting.png",
        args.bins,
    )
    _plot_hist(
        utility_stability,
        "Utility label distribution (Stability + Margin + EarlyLoss)",
        "Utility label score",
        output_dir / "utility_hist_stability.png",
        args.bins,
    )
    if utility_boundary is not None:
        _plot_hist(
            utility_boundary,
            "BoundaryInfoScore + EarlyLossScore + StabilityScore (mean)",
            "Utility label score",
            output_dir / "utility_hist_boundary_early_stability.png",
            args.bins,
        )
    _plot_hist(
        early_result.scores,
        "EarlyLossScore distribution",
        "EarlyLossScore",
        output_dir / "early_loss_hist.png",
        args.bins,
    )
    _plot_hist(
        margin_result.scores,
        "MarginScore distribution (probability margin based)",
        "MarginScore (probability margin based)",
        output_dir / "margin_hist.png",
        args.bins,
    )
    _plot_hist(
        forgetting_result.scores,
        "ForgettingScore distribution",
        "ForgettingScore",
        output_dir / "forgetting_hist.png",
        args.bins,
    )
    _plot_hist(
        stability_result.scores,
        "StabilityScore distribution",
        "StabilityScore",
        output_dir / "stability_hist.png",
        args.bins,
    )
    if boundary_result is not None:
        _plot_hist(
            boundary_result.scores,
            "BoundaryInfoScore distribution",
            "BoundaryInfoScore",
            output_dir / "boundary_info_hist.png",
            args.bins,
        )

    # 计算 EarlyLossScore 与 StabilityScore / ForgettingScore 的 Pearson 相关系数并打印
    corr_early_stability = np.corrcoef(early_result.scores, stability_result.scores)[0, 1]
    corr_early_forgetting = np.corrcoef(early_result.scores, forgetting_result.scores)[0, 1]

    print(f"Correlation EarlyLoss vs Stability: {corr_early_stability:.6f}")
    print(f"Correlation EarlyLoss vs Forgetting: {corr_early_forgetting:.6f}")
    if boundary_result is not None:
        corr_boundary_early = np.corrcoef(boundary_result.scores, early_result.scores)[
            0, 1
        ]
        corr_boundary_stability = np.corrcoef(
            boundary_result.scores, stability_result.scores
        )[0, 1]
        print(f"Correlation BoundaryInfo vs EarlyLoss: {corr_boundary_early:.6f}")
        print(f"Correlation BoundaryInfo vs Stability: {corr_boundary_stability:.6f}")

    saved_paths = [
        output_dir / "utility_hist_forgetting.png",
        output_dir / "utility_hist_stability.png",
        output_dir / "early_loss_hist.png",
        output_dir / "margin_hist.png",
        output_dir / "forgetting_hist.png",
        output_dir / "stability_hist.png",
    ]
    if utility_boundary is not None:
        saved_paths.append(output_dir / "utility_hist_boundary_early_stability.png")
    if boundary_result is not None:
        saved_paths.append(output_dir / "boundary_info_hist.png")

    print("Saved histogram images to:", *saved_paths)


if __name__ == "__main__":
    main()
