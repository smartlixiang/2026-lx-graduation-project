import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from weights.StabilityScore_v5 import StabilityScoreV5

# =========================
# User config
# =========================
NPZ_PATH = r"weights/proxy_logs/22/cifar10_resnet18_2026_01_20_11_42.npz"

# If you changed StabilityScore defaults in code, keep these as None.
# Otherwise, set explicit overrides here to match the run you want to debug.
BETA_OVERRIDE = None
THETA_OVERRIDE = None
LAMBDA_OVERRIDE = None

TOPK = 50000  # for reporting top-k by score

# Scatter downsampling
SCATTER_MAX_POINTS = 10000
SCATTER_SEED = 42

# Quadrant thresholds (if None, use quantiles on learnable subset)
L_TH = None           # e.g. 0.5
S_TH = None           # e.g. 0.90
L_TH_QUANTILE = 0.50  # default median
S_TH_QUANTILE = 0.75  # default upper quartile

OUT_DIR = Path("./debug_outputs_v5")


# =========================
# Utilities
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_npz_name(npz_path: str) -> dict:
    """Best-effort parse dataset/model from filename like cifar10_resnet18_YYYY_MM_DD_HH_MM.npz"""
    name = Path(npz_path).stem
    m = re.match(r"(?P<ds>[^_]+)_(?P<model>[^_]+)_(?P<rest>.+)", name)
    if not m:
        return {"dataset": "unknown", "model": "unknown", "tag": name}
    return {"dataset": m.group("ds"), "model": m.group("model"), "tag": name}


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }


def _print_stats(title: str, stats: dict) -> None:
    if stats.get("n", 0) == 0:
        print(f"{title}: n=0")
        return
    print(
        f"{title}: n={stats['n']} | mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, p10={stats['p10']:.4f}, p25={stats['p25']:.4f}, "
        f"p50={stats['p50']:.4f}, p75={stats['p75']:.4f}, p90={stats['p90']:.4f}, max={stats['max']:.4f}"
    )


def _print_quantiles(title: str, x: np.ndarray, quantiles: list[int] | None = None) -> None:
    if quantiles is None:
        quantiles = [1, 5, 50, 95, 99]
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        print(f"{title}: n=0")
        return
    values = np.percentile(x, quantiles)
    joined = ", ".join(f"p{q}={v:.4f}" for q, v in zip(quantiles, values))
    print(f"{title}: {joined}")


def _save_hist(
    x: np.ndarray,
    *,
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 40,
    vlines: list[tuple[float, str]] | None = None,
) -> None:
    plt.figure()
    plt.hist(x, bins=bins)
    if vlines:
        for v, lab in vlines:
            plt.axvline(v, linestyle="--", alpha=0.8, label=lab)
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    max_points: int = SCATTER_MAX_POINTS,
    seed: int = SCATTER_SEED,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    n = x.size
    if n == 0:
        return

    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
        c = c[idx]

    plt.figure()
    sc = plt.scatter(x, y, c=c, s=6, alpha=0.7)
    plt.colorbar(sc, label="StabilityScore")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _quadrant_stats(L: np.ndarray, S: np.ndarray, score: np.ndarray, L_th: float, S_th: float) -> list[dict]:
    """Return quadrant stats for (L,S) split by thresholds."""
    masks = {
        "Q1 (L low, S high)": (L < L_th) & (S >= S_th),
        "Q2 (L high, S high)": (L >= L_th) & (S >= S_th),
        "Q3 (L low, S low)": (L < L_th) & (S < S_th),
        "Q4 (L high, S low)": (L >= L_th) & (S < S_th),
    }
    out = []
    total = L.size if L is not None else 0
    for name, m in masks.items():
        s = score[m]
        st = _describe(s)
        out.append(
            {
                "name": name,
                "count": int(m.sum()),
                "ratio": float(m.sum() / max(1, total)),
                "score_stats": st,
            }
        )
    return out


def _print_quadrants(qs: list[dict], *, title: str) -> None:
    print(f"\n=== {title} ===")
    for q in qs:
        st = q["score_stats"]
        if st.get("n", 0) == 0:
            print(f"{q['name']}: count={q['count']} ({q['ratio'] * 100:.2f}%), score: n=0")
        else:
            print(
                f"{q['name']}: count={q['count']} ({q['ratio'] * 100:.2f}%), "
                f"score mean={st['mean']:.4f}, p25={st['p25']:.4f}, p50={st['p50']:.4f}, p75={st['p75']:.4f}, p90={st['p90']:.4f}"
            )


def _load_shape(npz_path: str) -> tuple[int, int]:
    data = np.load(npz_path)
    if "logits" not in data:
        raise ValueError("npz must contain logits to infer shape.")
    logits = data["logits"]
    if logits.ndim != 3:
        raise ValueError("npz['logits'] must be shape (epochs, num_samples, num_classes)")
    return int(logits.shape[0]), int(logits.shape[1])


def main() -> None:
    # ---- Compute StabilityScore using the main implementation
    ss_kwargs = {}
    if BETA_OVERRIDE is not None:
        ss_kwargs["beta"] = float(BETA_OVERRIDE)
    if THETA_OVERRIDE is not None:
        ss_kwargs["theta"] = float(THETA_OVERRIDE)
    if LAMBDA_OVERRIDE is not None:
        ss_kwargs["lam"] = float(LAMBDA_OVERRIDE)

    result = StabilityScoreV5(NPZ_PATH, **ss_kwargs).compute()

    scores = result.scores
    L = result.learn_time_normalized
    S = result.post_stability
    g_learn = result.g_learn
    score_learn = result.score_learn
    score_unlearn = result.score_unlearn
    learnable_mask = result.learnable_mask.astype(bool)

    # ---- Basic info
    meta = _parse_npz_name(NPZ_PATH)
    E, N = _load_shape(NPZ_PATH)

    tag = f"{meta['dataset']}_{meta['model']}_E{E}"
    _ensure_dir(OUT_DIR)

    print(f"Loaded: {NPZ_PATH}")
    print(f"Parsed: dataset={meta['dataset']}, model={meta['model']}")
    print(f"Epochs={E}, Samples={N}")
    print(f"Learnable (hard threshold) samples: {int(learnable_mask.sum())} ({learnable_mask.mean() * 100:.2f}%)")
    print(f"Non-learnable samples: {int((~learnable_mask).sum())} ({(~learnable_mask).mean() * 100:.2f}%)")

    # =========================
    # Quantiles / ratios required by v5
    # =========================
    print("\n=== Quantiles (v5) ===")
    _print_quantiles("S (post_stability)", S)
    _print_quantiles("L (learn_time_normalized)", L)
    _print_quantiles("g_learn (learnability gate)", g_learn)
    _print_quantiles("scores", scores)
    print(f"learnable_mask ratio: {float(learnable_mask.mean()):.4f}")
    print(f"scores>0.8 ratio: {float((scores > 0.8).mean()):.4f}")
    if learnable_mask.any():
        print(f"learnable scores>0.8 ratio: {float((scores[learnable_mask] > 0.8).mean()):.4f}")
    else:
        print("learnable scores>0.8 ratio: n=0")

    # =========================
    # Top-k report
    # =========================
    topk = min(TOPK, scores.size)
    top_idx = np.argsort(-scores)[:topk]
    top_L = L[top_idx]
    top_S = S[top_idx]
    print(f"\n=== Top-{topk} StabilityScore samples (by score) ===")
    _print_stats("Top-k LearnTime L_i", _describe(top_L))
    _print_stats("Top-k PostLearnStability S_i", _describe(top_S))

    # =========================
    # Core diagnostics 1: split learnable vs non-learnable
    # =========================
    L_learn = L[learnable_mask]
    S_learn = S[learnable_mask]
    score_learnable = scores[learnable_mask]

    L_non = L[~learnable_mask]
    S_non = S[~learnable_mask]
    score_non = scores[~learnable_mask]

    print("\n=== Stats: learnable subset (hard threshold) ===")
    _print_stats("L_i", _describe(L_learn))
    _print_stats("S_i", _describe(S_learn))
    _print_stats("Score", _describe(score_learnable))

    print("\n=== Stats: non-learnable subset (hard threshold) ===")
    _print_stats("L_i", _describe(L_non))
    _print_stats("S_i", _describe(S_non))
    _print_stats("Score", _describe(score_non))

    # =========================
    # Plots: distributions
    # =========================
    # Original/global distributions
    _save_hist(
        scores,
        title=f"StabilityScore distribution ({tag})",
        xlabel="StabilityScore",
        out_path=OUT_DIR / f"{tag}__v5__score_all.png",
        bins=40,
    )
    _save_hist(
        L,
        title=f"LearnTime L_i distribution ({tag})",
        xlabel="L_i",
        out_path=OUT_DIR / f"{tag}__v5__L_all.png",
        bins=40,
    )
    _save_hist(
        S,
        title=f"PostLearnStability S_i distribution ({tag})",
        xlabel="S_i",
        out_path=OUT_DIR / f"{tag}__v5__S_all.png",
        bins=40,
    )

    # Split distributions
    _save_hist(
        score_learnable,
        title=f"Score distribution - learnable ({tag})",
        xlabel="StabilityScore",
        out_path=OUT_DIR / f"{tag}__v5__score_learnable.png",
        bins=40,
    )
    _save_hist(
        score_non,
        title=f"Score distribution - non-learnable ({tag})",
        xlabel="StabilityScore",
        out_path=OUT_DIR / f"{tag}__v5__score_nonlearnable.png",
        bins=40,
    )
    _save_hist(
        L_learn,
        title=f"L_i distribution - learnable ({tag})",
        xlabel="L_i",
        out_path=OUT_DIR / f"{tag}__v5__L_learnable.png",
        bins=40,
    )
    _save_hist(
        L_non,
        title=f"L_i distribution - non-learnable ({tag})",
        xlabel="L_i",
        out_path=OUT_DIR / f"{tag}__v5__L_nonlearnable.png",
        bins=40,
    )
    _save_hist(
        S_learn,
        title=f"S_i distribution - learnable ({tag})",
        xlabel="S_i",
        out_path=OUT_DIR / f"{tag}__v5__S_learnable.png",
        bins=40,
    )
    _save_hist(
        S_non,
        title=f"S_i distribution - non-learnable ({tag})",
        xlabel="S_i",
        out_path=OUT_DIR / f"{tag}__v5__S_nonlearnable.png",
        bins=40,
    )

    # v5-specific histograms
    _save_hist(
        g_learn,
        title=f"g_learn distribution ({tag})",
        xlabel="g_learn",
        out_path=OUT_DIR / f"{tag}__v5__g_learn_all.png",
        bins=40,
        vlines=[(0.5, "g_learn=0.5")],
    )
    _save_hist(
        score_learn,
        title=f"Score_learn distribution ({tag})",
        xlabel="Score_learn",
        out_path=OUT_DIR / f"{tag}__v5__score_learn_all.png",
        bins=40,
    )
    _save_hist(
        score_unlearn,
        title=f"Score_unlearn distribution ({tag})",
        xlabel="Score_unlearn",
        out_path=OUT_DIR / f"{tag}__v5__score_unlearn_all.png",
        bins=40,
    )

    # =========================
    # Core diagnostics 2: scatter L vs S colored by score
    # =========================
    # Learnable-only scatter
    _save_scatter(
        L_learn,
        S_learn,
        score_learnable,
        title=f"L vs S (learnable only), color=score ({tag})",
        xlabel="L_i",
        ylabel="S_i",
        out_path=OUT_DIR / f"{tag}__v5__scatter_LS_learnable.png",
    )
    # All-samples scatter
    _save_scatter(
        L,
        S,
        scores,
        title=f"L vs S (all), color=score ({tag})",
        xlabel="L_i",
        ylabel="S_i",
        out_path=OUT_DIR / f"{tag}__v5__scatter_LS_all.png",
    )
    # v5 scatter: S vs g_learn
    _save_scatter(
        S,
        g_learn,
        scores,
        title=f"S vs g_learn (all), color=score ({tag})",
        xlabel="S_i",
        ylabel="g_learn",
        out_path=OUT_DIR / f"{tag}__v5__scatter_S_glearn_all.png",
    )

    # =========================
    # Optional: quadrant stats
    # =========================
    if L_learn.size > 0 and S_learn.size > 0:
        L_th_val = float(L_TH) if L_TH is not None else float(np.quantile(L_learn, L_TH_QUANTILE))
        S_th_val = float(S_TH) if S_TH is not None else float(np.quantile(S_learn, S_TH_QUANTILE))
        print(f"\nQuadrant thresholds (learnable subset): L_th={L_th_val:.4f}, S_th={S_th_val:.4f}")
        qs = _quadrant_stats(L_learn, S_learn, score_learnable, L_th_val, S_th_val)
        _print_quadrants(qs, title="Quadrant stats on learnable subset (L vs S)")

    # =========================
    # Final summary
    # =========================
    print("\n=== Summary ===")
    _print_stats("Score [all]", _describe(scores))
    _print_stats("Score [learnable]", _describe(score_learnable))
    _print_stats("Score [non-learnable]", _describe(score_non))
    print(f"Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
