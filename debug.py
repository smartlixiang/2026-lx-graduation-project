# viz_forgetting_top200.py
import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = r"weights/proxy_logs/22/cifar10_resnet18_2026_01_20_11_42.npz"
TOPK = 30000


def _pick_key(keys, candidates):
    """Pick the first key in keys that contains any candidate substr."""
    for c in candidates:
        for k in keys:
            if c.lower() in k.lower():
                return k
    return None


def load_proxy_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    # Try to locate logits/probs/preds and labels
    # We prefer logits or probs over preds because forgetting requires correctness per epoch.
    logits_k = _pick_key(keys, ["logits", "epoch_logits"])
    probs_k = _pick_key(keys, ["probs", "prob", "epoch_probs"])
    preds_k = _pick_key(keys, ["preds", "pred", "epoch_preds"])
    labels_k = _pick_key(keys, ["labels", "targets", "y_true", "gt", "y"])

    if labels_k is None:
        print("❌ Cannot find labels key. Available keys:", keys)
        raise KeyError("labels key not found in npz")

    labels = data[labels_k]

    # Ensure labels shape: (N,)
    labels = np.asarray(labels).astype(int).reshape(-1)

    if logits_k is not None:
        arr = data[logits_k]
        src = f"logits:{logits_k}"
        # Expected shape: (E, N, C) or (N, E, C)
        logits = np.asarray(arr)
        return logits, labels, src, keys

    if probs_k is not None:
        arr = data[probs_k]
        src = f"probs:{probs_k}"
        probs = np.asarray(arr)
        return probs, labels, src, keys

    if preds_k is not None:
        arr = data[preds_k]
        src = f"preds:{preds_k}"
        preds = np.asarray(arr)
        return preds, labels, src, keys

    print("❌ Cannot find logits/probs/preds key. Available keys:", keys)
    raise KeyError("no prediction history key found in npz")


def to_epoch_pred(history):
    """
    Convert prediction history to epoch-wise predicted class indices.
    Input history can be:
      - logits/probs: shape (E, N, C) or (N, E, C)
      - preds: shape (E, N) or (N, E)
    Returns preds: shape (E, N)
    """
    h = np.asarray(history)
    if h.ndim == 3:
        # logits/probs
        # Detect which axis is epoch
        # Heuristic: if first axis is small (<=500) treat as epoch.
        if h.shape[0] <= 500:
            # (E, N, C)
            preds = np.argmax(h, axis=-1)  # (E, N)
        elif h.shape[1] <= 500:
            # (N, E, C) -> transpose
            preds = np.argmax(h, axis=-1).transpose(1, 0)  # (E, N)
        else:
            raise ValueError(f"Ambiguous 3D shape {h.shape}, cannot infer epoch axis.")
        return preds

    if h.ndim == 2:
        # preds
        if h.shape[0] <= 500:
            return h  # (E, N)
        if h.shape[1] <= 500:
            return h.transpose(1, 0)  # (E, N)
        raise ValueError(f"Ambiguous 2D shape {h.shape}, cannot infer epoch axis.")

    raise ValueError(f"Unsupported history ndim={h.ndim}, shape={h.shape}")


def compute_accuracy_and_forgetting(epoch_preds, labels):
    """
    epoch_preds: (E, N), labels: (N,)
    Returns:
      correct: (E, N) bool
      r: (N,) accuracy over epochs
      forgetting_counts: (N,) number of 1->0 transitions across epochs
      forgetting_normalized: (N,) min-max normalized forgetting counts
    """
    labels = labels.reshape(1, -1)
    correct = (epoch_preds == labels)  # (E, N) bool

    # accuracy per sample across epochs
    r = correct.mean(axis=0)  # (N,)

    # forgetting events: correct at t-1 then wrong at t
    prev = correct[:-1]
    curr = correct[1:]
    forgetting_events = (prev & (~curr)).astype(np.int32)
    forgetting_counts = forgetting_events.sum(axis=0)  # (N,)

    # min-max normalize (as in your code)
    fmin = forgetting_counts.min()
    fmax = forgetting_counts.max()
    if fmax == fmin:
        forgetting_normalized = np.zeros_like(forgetting_counts, dtype=np.float32)
    else:
        forgetting_normalized = (forgetting_counts - fmin) / (fmax - fmin)

    return correct, r, forgetting_counts, forgetting_normalized


def compute_forgetting_score(r, Fnorm,
                             score_00=0.9, score_10=0.7, score_01=0.1, score_11=0.2):
    """
    Bilinear interpolation exactly like the formula:
    (1-r)(1-F)*s00 + r(1-F)*s10 + (1-r)F*s01 + rF*s11
    """
    r = np.asarray(r, dtype=np.float32)
    F = np.asarray(Fnorm, dtype=np.float32)
    return (1 - r) * (1 - F) * score_00 + r * (1 - F) * score_10 + (1 - r) * F * score_01 + r * F * score_11


def print_counts_hist(forgetting_counts):
    vals, cnts = np.unique(forgetting_counts, return_counts=True)
    total = forgetting_counts.size
    print("\n=== forgetting_counts histogram (value: count / ratio) ===")
    for v, c in zip(vals.tolist(), cnts.tolist()):
        print(f"{v:>3}: {c:>6}  ({c / total:.4f})")
    print(f"Total N={total}, max={forgetting_counts.max()}, min={forgetting_counts.min()}")


def main():
    history, labels, src, keys = load_proxy_npz(NPZ_PATH)
    print(f"Loaded {NPZ_PATH}")
    print(f"Using prediction history from: {src}")
    print(f"Available keys: {keys}")

    epoch_preds = to_epoch_pred(history)
    E, N = epoch_preds.shape
    print(f"Epoch preds shape: (E={E}, N={N})")

    _, r, fcnt, fnorm = compute_accuracy_and_forgetting(epoch_preds, labels)
    fscore = compute_forgetting_score(r, fnorm)

    # (A) top200 ForgettingScore accuracy distribution
    topk = min(TOPK, N)
    top_idx = np.argsort(-fscore)[:topk]
    top_r = r[top_idx]
    print(f"\n=== Top-{topk} ForgettingScore samples ===")
    print(f"Accuracy r_i: mean={top_r.mean():.4f}, std={top_r.std():.4f}, "
          f"min={top_r.min():.4f}, p25={np.percentile(top_r, 25):.4f}, "
          f"median={np.median(top_r):.4f}, p75={np.percentile(top_r, 75):.4f}, max={top_r.max():.4f}")

    # (B) forgetting_counts histogram
    print_counts_hist(fcnt)

    # (C) forgetting_normalized distribution
    print("\n=== forgetting_normalized distribution ===")
    print(f"mean={fnorm.mean():.4f}, std={fnorm.std():.4f}, "
          f"min={fnorm.min():.4f}, p25={np.percentile(fnorm, 25):.4f}, "
          f"median={np.median(fnorm):.4f}, p75={np.percentile(fnorm, 75):.4f}, max={fnorm.max():.4f}")

    # ---- Plots ----
    # 1) top200 accuracy histogram
    plt.figure()
    plt.hist(top_r, bins=20)
    plt.title(f"Accuracy distribution of Top-{topk} ForgettingScore samples")
    plt.xlabel("Accuracy r_i")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)

    # 2) forgetting_counts bar chart (ratio)
    vals, cnts = np.unique(fcnt, return_counts=True)
    ratios = cnts / N
    plt.figure()
    plt.bar(vals, ratios)
    plt.title("forgetting_counts histogram (ratio)")
    plt.xlabel("forgetting_counts")
    plt.ylabel("ratio")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    # 3) forgetting_normalized histogram
    plt.figure()
    plt.hist(fnorm, bins=30)
    plt.title("forgetting_normalized distribution")
    plt.xlabel("forgetting_normalized")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.show()


if __name__ == "__main__":
    main()
