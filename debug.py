import matplotlib.pyplot as plt
import numpy as np

from weights import StabilityScore

NPZ_PATH = r"weights/proxy_logs/22/cifar10_resnet18_2026_01_20_11_42.npz"
TOPK = 10000


def main():
    result = StabilityScore(NPZ_PATH).compute()
    scores = result.scores
    learn_time = result.learn_time_normalized
    post_stability = result.post_stability

    print(f"Loaded {NPZ_PATH}")
    print(f"Total samples: {scores.size}")
    print(f"Stable-learnable samples: {result.learnable_mask.sum()}")

    topk = min(TOPK, scores.size)
    top_idx = np.argsort(-scores)[:topk]
    top_l = learn_time[top_idx]
    top_s = post_stability[top_idx]
    print(f"\n=== Top-{topk} StabilityScore samples ===")
    print(
        f"LearnTime L_i: mean={top_l.mean():.4f}, std={top_l.std():.4f}, "
        f"min={top_l.min():.4f}, p25={np.percentile(top_l, 25):.4f}, "
        f"median={np.median(top_l):.4f}, p75={np.percentile(top_l, 75):.4f}, max={top_l.max():.4f}"
    )
    print(
        f"Post-stability S_i: mean={top_s.mean():.4f}, std={top_s.std():.4f}, "
        f"min={top_s.min():.4f}, p25={np.percentile(top_s, 25):.4f}, "
        f"median={np.median(top_s):.4f}, p75={np.percentile(top_s, 75):.4f}, max={top_s.max():.4f}"
    )

    plt.figure()
    plt.hist(scores, bins=30)
    plt.title("StabilityScore distribution")
    plt.xlabel("StabilityScore")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.figure()
    plt.hist(learn_time, bins=30)
    plt.title("LearnTime L_i distribution")
    plt.xlabel("L_i")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.figure()
    plt.hist(post_stability, bins=30)
    plt.title("PostLearnStability S_i distribution")
    plt.xlabel("S_i")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.show()


if __name__ == "__main__":
    main()
