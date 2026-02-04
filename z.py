from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset import BaseDataLoader  # noqa: E402
from utils.proxy_log_utils import load_proxy_log  # noqa: E402
from weights.AbsorptionEfficiencyScore import AbsorptionEfficiencyScore  # noqa: E402
from weights.CoverageGainScore import CoverageGainScore  # noqa: E402
from weights.InformativenessScore import InformativenessScore  # noqa: E402
from weights.PersistentDifficultyScore import PersistentDifficultyScore  # noqa: E402
from weights.RiskScore import RiskScore  # noqa: E402
from weights.TransferGainScore import TransferGainScore  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "debug_z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep score hyperparameters and plot histograms.")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="weights/proxy_logs/cifar10/resnet18/22/100",
        help="Path to proxy training log (.npz) for A/B/C/R.",
    )
    parser.add_argument(
        "--cv_log_dir",
        type=str,
        default="weights/proxy_logs/cifar10/resnet18/22/100",
        help="Path to k-fold proxy log directory for T/V.",
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name.")
    parser.add_argument("--data_root", type=str, default="./data", help="Dataset root path.")
    parser.add_argument("--seed", type=int, default=0, help="Dataset seed (for output tag).")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional cap on number of sweep cases per component.",
    )
    return parser.parse_args()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.3g}"
        if "e" in text or "E" in text:
            text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")
    return str(value)


def _params_to_tag(params: Iterable[tuple[str, Any]]) -> str:
    parts = [f"{key}={_format_value(val)}" for key, val in params]
    return "_".join(parts)


def _plot_hist_pair(
    out_path: Path,
    raw: np.ndarray,
    norm: np.ndarray,
    bins: int,
    title_raw: str,
    title_norm: str,
    risk_nonzero: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if risk_nonzero:
        zeros_raw = int(np.sum(raw == 0.0))
        zeros_norm = int(np.sum(norm == 0.0))
        raw_vals = raw[raw > 0.0]
        norm_vals = norm[norm > 0.0]
        axes[0].hist(raw_vals, bins=bins, color="#4C72B0", alpha=0.85)
        axes[0].set_title(f"{title_raw} | zeros={zeros_raw}")
        axes[1].hist(norm_vals, bins=bins, color="#55A868", alpha=0.85)
        axes[1].set_title(f"{title_norm} | zeros={zeros_norm}")
    else:
        axes[0].hist(raw, bins=bins, color="#4C72B0", alpha=0.85)
        axes[0].set_title(title_raw)
        axes[1].hist(norm, bins=bins, color="#55A868", alpha=0.85)
        axes[1].set_title(title_norm)
    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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


def _limit_cases(cases: list[dict[str, Any]], max_cases: int | None) -> list[dict[str, Any]]:
    if max_cases is None or max_cases <= 0:
        return cases
    return cases[:max_cases]


def _sweep_absorption() -> list[dict[str, Any]]:
    cases = []
    for temp_progress, sigma_level in itertools.product(
        [0.5, 1.0, 2.0, 4.0, 8.0],
        [0.5, 1.0, 2.0, 4.0, 8.0],
    ):
        cases.append(
            {
                "temp_progress": temp_progress,
                "sigma_level": sigma_level,
            }
        )
    return cases


def _sweep_informativeness() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for mu_percentile, stats_by_class, tau_p_mode in itertools.product(
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [True, False],
        ["fixed", "percentile"],
    ):
        if tau_p_mode == "percentile":
            for tau_p_percentile in [50.0, 70.0, 90.0, 97.0]:
                cases.append(
                    {
                        "mu_percentile": mu_percentile,
                        "stats_by_class": stats_by_class,
                        "tau_p_mode": tau_p_mode,
                        "tau_p_percentile": tau_p_percentile,
                    }
                )
        else:
            for tau_p in [0.005, 0.02, 0.05, 0.1]:
                cases.append(
                    {
                        "mu_percentile": mu_percentile,
                        "stats_by_class": stats_by_class,
                        "tau_p_mode": tau_p_mode,
                        "tau_p": tau_p,
                    }
                )
    return cases


def _sweep_coverage() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for tau_g_mode, tau_g_by_class, k_pct, s_g in itertools.product(
        ["fixed", "percentile"],
        [True, False],
        [0.001, 0.0025, 0.005, 0.01, 0.02],
        [0.03, 0.07, 0.15],
    ):
        if tau_g_mode == "percentile":
            for tau_g_percentile in [10.0, 20.0, 30.0, 40.0, 50.0]:
                cases.append(
                    {
                        "tau_g_mode": tau_g_mode,
                        "tau_g_percentile": tau_g_percentile,
                        "tau_g_by_class": tau_g_by_class,
                        "k_pct": k_pct,
                        "s_g": s_g,
                    }
                )
        else:
            for tau_g in [0.05, 0.1, 0.15, 0.25, 0.4]:
                cases.append(
                    {
                        "tau_g_mode": tau_g_mode,
                        "tau_g": tau_g,
                        "tau_g_by_class": tau_g_by_class,
                        "k_pct": k_pct,
                        "s_g": s_g,
                    }
                )
    return cases


def _sweep_transfer() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for tau_p_mode in ["fixed", "percentile"]:
        if tau_p_mode == "percentile":
            for tau_p_percentile in [50.0, 70.0, 90.0, 97.0]:
                cases.append(
                    {
                        "tau_p_mode": tau_p_mode,
                        "tau_p_percentile": tau_p_percentile,
                    }
                )
        else:
            for tau_p in [0.02, 0.05, 0.1, 0.2, 0.4]:
                cases.append(
                    {
                        "tau_p_mode": tau_p_mode,
                        "tau_p": tau_p,
                    }
                )
    return cases


def _sweep_persistent() -> list[dict[str, Any]]:
    cases = []
    for late_ratio, tau_m in itertools.product(
        [0.3, 0.5, 0.7],
        [0.05, 0.1, 0.15, 0.25],
    ):
        cases.append({"late_ratio": late_ratio, "tau_m": tau_m})
    return cases


def _sweep_risk() -> list[dict[str, Any]]:
    cases = []
    for tail_q0, tail_q1, lambda_improve in itertools.product(
        [0.90, 0.95, 0.97],
        [0.99, 0.995, 0.999],
        [0.3, 0.7, 1.2],
    ):
        cases.append(
            {
                "tail_q0": tail_q0,
                "tail_q1": tail_q1,
                "lambda_improve": lambda_improve,
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"npz_path not found: {npz_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["A", "B", "C", "T", "V", "R"]:
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    proxy_data = load_proxy_log(npz_path, args.dataset, args.data_root)

    cv_log_dir = Path(args.cv_log_dir) if args.cv_log_dir else None
    dataset = None
    if cv_log_dir is not None:
        if not cv_log_dir.exists():
            raise FileNotFoundError(f"cv_log_dir not found: {cv_log_dir}")
        dataset = _load_train_dataset(args.dataset, args.data_root, args.seed)

    for params in _limit_cases(_sweep_absorption(), args.max_cases):
        scorer = AbsorptionEfficiencyScore(npz_path, **params)
        result = scorer.compute(proxy_logs=proxy_data)
        tag = _params_to_tag(
            [
                ("temp_progress", params["temp_progress"]),
                ("sigma_level", params["sigma_level"]),
            ]
        )
        out_path = OUTPUT_ROOT / "A" / f"{tag}.png"
        _plot_hist_pair(
            out_path,
            result.raw_score,
            result.scores,
            args.bins,
            "AbsorptionEfficiencyScore raw",
            "AbsorptionEfficiencyScore norm",
        )

    for params in _limit_cases(_sweep_informativeness(), args.max_cases):
        scorer = InformativenessScore(npz_path, **params)
        result = scorer.compute(proxy_logs=proxy_data)
        tag_parts = [
            ("tau_p_mode", params["tau_p_mode"]),
            ("mu_pct", params["mu_percentile"]),
            ("stats_by_class", params["stats_by_class"]),
        ]
        if params["tau_p_mode"] == "percentile":
            tag_parts.append(("tau_p_pct", params["tau_p_percentile"]))
        else:
            tag_parts.append(("tau_p", params["tau_p"]))
        tag = _params_to_tag(tag_parts)
        out_path = OUTPUT_ROOT / "B" / f"{tag}.png"
        _plot_hist_pair(
            out_path,
            result.raw_score,
            result.scores,
            args.bins,
            "InformativenessScore raw",
            "InformativenessScore norm",
        )

    for params in _limit_cases(_sweep_coverage(), args.max_cases):
        scorer = CoverageGainScore(npz_path, **params)
        result = scorer.compute(proxy_logs=proxy_data)
        tag_parts = [
            ("tau_g_mode", params["tau_g_mode"]),
            ("tau_g_by_class", params["tau_g_by_class"]),
            ("k_pct", params["k_pct"]),
            ("s_g", params["s_g"]),
        ]
        if params["tau_g_mode"] == "percentile":
            tag_parts.append(("tau_g_pct", params["tau_g_percentile"]))
        else:
            tag_parts.append(("tau_g", params["tau_g"]))
        tag = _params_to_tag(tag_parts)
        out_path = OUTPUT_ROOT / "C" / f"{tag}.png"
        _plot_hist_pair(
            out_path,
            result.knn_distance,
            result.scores,
            args.bins,
            "CoverageGainScore raw",
            "CoverageGainScore norm",
        )

    for params in _limit_cases(_sweep_risk(), args.max_cases):
        scorer = RiskScore(npz_path, **params)
        result = scorer.compute(proxy_logs=proxy_data)
        tag = _params_to_tag(
            [
                ("tail_q0", params["tail_q0"]),
                ("tail_q1", params["tail_q1"]),
                ("lambda_improve", params["lambda_improve"]),
            ]
        )
        out_path = OUTPUT_ROOT / "R" / f"{tag}.png"
        _plot_hist_pair(
            out_path,
            result.raw_score,
            result.scores,
            args.bins,
            "RiskScore raw (nonzero)",
            "RiskScore norm (nonzero)",
            risk_nonzero=True,
        )

    if cv_log_dir is not None and dataset is not None:
        for params in _limit_cases(_sweep_transfer(), args.max_cases):
            scorer = TransferGainScore(**params)
            result = scorer.compute(cv_log_dir, dataset)
            scores = result["score"].astype(np.float32)
            tag_parts = [("tau_p_mode", params["tau_p_mode"])]
            if params["tau_p_mode"] == "percentile":
                tag_parts.append(("tau_p_pct", params["tau_p_percentile"]))
            else:
                tag_parts.append(("tau_p", params["tau_p"]))
            tag = _params_to_tag(tag_parts)
            out_path = OUTPUT_ROOT / "T" / f"{tag}.png"
            _plot_hist_pair(
                out_path,
                scores,
                scores,
                args.bins,
                "TransferGainScore raw",
                "TransferGainScore norm",
            )

        for params in _limit_cases(_sweep_persistent(), args.max_cases):
            scorer = PersistentDifficultyScore(**params)
            result = scorer.compute(cv_log_dir, dataset)
            scores = result["score"].astype(np.float32)
            tag = _params_to_tag(
                [
                    ("late_ratio", params["late_ratio"]),
                    ("tau_m", params["tau_m"]),
                ]
            )
            out_path = OUTPUT_ROOT / "V" / f"{tag}.png"
            _plot_hist_pair(
                out_path,
                scores,
                scores,
                args.bins,
                "PersistentDifficultyScore raw",
                "PersistentDifficultyScore norm",
            )


if __name__ == "__main__":
    main()
