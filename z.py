"""Generate utility label distributions from proxy training logs.

This script computes two variants of utility labels:
1) ForgettingScore + MarginScore + EarlyLossScore
2) StabilityScore + MarginScore + EarlyLossScore
Then plots histogram distributions and saves images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weights import EarlyLossScore, ForgettingScore, MarginScore, StabilityScore


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
    parser.add_argument("--margin-delta", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_weights_debug",
        help="Directory to save histogram images.",
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


def main() -> None:
    args = parse_args()
    proxy_log = Path(args.proxy_log)
    if not proxy_log.exists():
        raise FileNotFoundError(f"Proxy log not found: {proxy_log}")

    forgetting_result = ForgettingScore(proxy_log).compute()
    margin_result = MarginScore(proxy_log, delta=args.margin_delta).compute()
    early_result = EarlyLossScore(proxy_log, early_epochs=args.early_epochs).compute()
    stability_result = StabilityScore(proxy_log).compute()

    _ensure_same_indices(
        {
            "forgetting": forgetting_result.indices,
            "margin": margin_result.indices,
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
    _plot_hist(
        early_result.scores,
        "EarlyLossScore distribution",
        "EarlyLossScore",
        output_dir / "early_loss_hist.png",
        args.bins,
    )
    _plot_hist(
        margin_result.scores,
        "MarginScore distribution",
        "MarginScore",
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

    # 计算 EarlyLossScore 与 StabilityScore / ForgettingScore 的 Pearson 相关系数并打印
    corr_early_stability = np.corrcoef(early_result.scores, stability_result.scores)[0, 1]
    corr_early_forgetting = np.corrcoef(early_result.scores, forgetting_result.scores)[0, 1]

    print(f"Correlation EarlyLoss vs Stability: {corr_early_stability:.6f}")
    print(f"Correlation EarlyLoss vs Forgetting: {corr_early_forgetting:.6f}")

    print(
        "Saved histogram images to:",
        output_dir / "utility_hist_forgetting.png",
        output_dir / "utility_hist_stability.png",
        output_dir / "early_loss_hist.png",
        output_dir / "margin_hist.png",
        output_dir / "forgetting_hist.png",
        output_dir / "stability_hist.png",
    )


if __name__ == "__main__":
    main()
