"""End-to-end debug script for CIFAR-10 selection + training using proxy dynamics.

This version implements the user's intended baseline:
- Compute utility score u = A + B + C - R (AbsorptionEfficiency, Informativeness,
  CoverageGain, RiskScore)
- Normalize u with a global quantile-based min-max
- Select top-k within each true class proportional to cut_ratio
- Train downstream model on the selected subset (true labels)
"""

# Key interfaces:
# - _compute_u_scores_from_proxy_log(): builds A/B/C/R/u from proxy logs for selection.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.seed import set_seed

# NEW: import dynamic scoring terms (must exist in your repo)
from utils.proxy_log_utils import load_proxy_log
from utils.score_utils import quantile_minmax
from weights.AbsorptionEfficiencyScore import AbsorptionEfficiencyScore
from weights.CoverageGainScore import CoverageGainScore
from weights.InformativenessScore import InformativenessScore
from weights.RiskScore import RiskScore

# =========================
# User config
# =========================
DATASET = "cifar10"
MODEL_NAME = "resnet50"
SEED = 22
CUT_RATIOS = [30, 40, 50]
PROXY_EPOCHS = 50
PROXY_LOG_DIR = (
    Path("weights") / "proxy_logs" / DATASET / "resnet18" / str(SEED) / str(PROXY_EPOCHS)
)
OUT_DIR = Path("debug")

EPOCHS = 200
BATCH_SIZE = 128
NUM_WORKERS = 4
INIT_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


# =========================
# Utilities
# =========================

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_u_scores_from_proxy_log(log_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compute u = A + B + C - R, then global quantile normalize.

    Returns:
        u_norm: shape (N,) float32 in [0,1], aligned to dataset index order
        labels: shape (N,) int64 true labels, aligned to dataset index order
    """
    proxy_data = load_proxy_log(log_path, DATASET, str(CONFIG.data_root))
    labels_proxy = proxy_data["labels"]
    indices_proxy = proxy_data["indices"]

    # Compute three dynamic scores (each should return per-sample scores + indices)
    absorption_res = AbsorptionEfficiencyScore(log_path).compute(proxy_logs=proxy_data)
    informativeness_res = InformativenessScore(log_path).compute(proxy_logs=proxy_data)
    cov_res = CoverageGainScore(log_path).compute(proxy_logs=proxy_data)
    risk_res = RiskScore(log_path).compute(proxy_logs=proxy_data)

    # Build map from index -> position for each result to align robustly
    def _to_full(scores: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
        full = np.full((n,), np.nan, dtype=np.float32)
        if indices.shape[0] != scores.shape[0]:
            raise ValueError("indices and scores length mismatch in a score result")
        if np.min(indices) < 0 or np.max(indices) >= n:
            raise ValueError("score result indices out of range")
        full[indices.astype(np.int64)] = scores.astype(np.float32)
        return full

    n = labels_proxy.shape[0]
    a_full = _to_full(absorption_res.scores, absorption_res.indices, n)
    b_full = _to_full(informativeness_res.scores, informativeness_res.indices, n)
    c_full = _to_full(cov_res.scores, cov_res.indices, n)
    r_full = _to_full(risk_res.scores, risk_res.indices, n)

    if (
        np.any(np.isnan(a_full))
        or np.any(np.isnan(b_full))
        or np.any(np.isnan(c_full))
        or np.any(np.isnan(r_full))
    ):
        raise RuntimeError("Failed to align some dynamic scores to full index space (NaNs found).")

    u_raw = a_full + b_full + c_full - r_full

    # Ensure labels are aligned to dataset index order 0..N-1
    labels_full = np.full((n,), -1, dtype=np.int64)
    labels_full[indices_proxy.astype(np.int64)] = labels_proxy
    if np.any(labels_full < 0):
        raise RuntimeError("Failed to align proxy labels to full index space.")

    u_norm = quantile_minmax(u_raw.astype(np.float32), q_low=0.1, q_high=0.99)
    components = {
        "A": a_full.astype(np.float32),
        "B": b_full.astype(np.float32),
        "C": c_full.astype(np.float32),
        "R": r_full.astype(np.float32),
        "u": u_norm.astype(np.float32),
    }
    return u_norm, labels_full, components


def _select_topk_indices(
    true_labels: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    cut_ratio: int,
) -> np.ndarray:
    """Select class-proportional top-k within each TRUE class by scores."""
    if not 0 < cut_ratio <= 100:
        raise ValueError("cut_ratio must be in (0, 100]")
    if true_labels.shape[0] != scores.shape[0]:
        raise ValueError("true_labels and scores must have the same length")

    selected: list[int] = []
    ratio = cut_ratio / 100.0

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(true_labels == class_id)
        if class_indices.size == 0:
            continue
        if cut_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        class_scores = scores[class_indices]
        topk = class_indices[np.argsort(-class_scores)[:num_select]]
        selected.extend(topk.tolist())

    return np.sort(np.asarray(selected, dtype=np.int64))


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=INIT_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    history = {
        "train_loss": [],
        "test_acc": [],
    }

    start_eval_epoch = max(1, epochs - 9)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        history["train_loss"].append(epoch_loss)
        if epoch >= start_eval_epoch:
            test_acc = _evaluate(model, test_loader, device)
            history["test_acc"].append(test_acc)
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}, test_acc={test_acc:.4f}")
        else:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f}")

    return history


def main() -> None:
    set_seed(SEED)
    device = CONFIG.global_device

    _ensure_dir(OUT_DIR)

    proxy_log_path = PROXY_LOG_DIR
    if not proxy_log_path.exists():
        raise FileNotFoundError(f"Proxy log not found: {proxy_log_path}")
    print(f"Using proxy log: {proxy_log_path}")

    # Compute the intended utility score u (aligned to dataset order)
    u_scores, labels_full, components = _compute_u_scores_from_proxy_log(proxy_log_path)

    # Data
    data_loader = BaseDataLoader(
        DATASET,
        data_path=CONFIG.data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=0.0,
        seed=SEED,
    )
    train_loader, _, test_loader = data_loader.load()
    train_dataset = train_loader.dataset

    # Basic alignment sanity check (optional but helps catch index mismatch early)
    if len(train_dataset) != u_scores.shape[0]:
        raise RuntimeError(
            f"Dataset length mismatch: len(train_dataset)={len(train_dataset)} vs u_scores={u_scores.shape[0]}. "
            "Your proxy log indices likely do not correspond to this training dataset ordering."
        )

    results = {
        "dataset": DATASET,
        "model": MODEL_NAME,
        "seed": SEED,
        "proxy_log": str(proxy_log_path),
        "cut_ratios": CUT_RATIOS,
        "epochs": EPOCHS,
        "metrics": {},
    }

    num_classes = data_loader.num_classes

    for cut_ratio in CUT_RATIOS:
        selected_indices = _select_topk_indices(labels_full, u_scores, num_classes, cut_ratio)
        subset = Subset(train_dataset, selected_indices.tolist())
        generator = torch.Generator().manual_seed(SEED)
        train_subset_loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=NUM_WORKERS,
            pin_memory=CONFIG.pin_memory,
            drop_last=False,
        )

        model_fn = get_model(MODEL_NAME)
        model = model_fn(num_classes=num_classes).to(device)

        print(f"\nTraining with cut_ratio={cut_ratio} (selected={len(subset)})")
        history = _train_model(model, train_subset_loader, test_loader, device, EPOCHS)

        test_acc_samples = history["test_acc"]
        metrics = {
            "selected": int(len(subset)),
            "final_test_acc": float(test_acc_samples[-1]) if test_acc_samples else 0.0,
            "best_test_acc": float(np.max(test_acc_samples)) if test_acc_samples else 0.0,
            "avg_test_acc": float(np.mean(test_acc_samples)) if test_acc_samples else 0.0,
        }
        results["metrics"][str(cut_ratio)] = metrics

        selection_path = OUT_DIR / f"selection_cr{cut_ratio}.npz"
        np.savez(
            selection_path,
            indices=selected_indices,
            labels=labels_full[selected_indices],
            u_scores=u_scores[selected_indices],
        )

        history_path = OUT_DIR / f"history_cr{cut_ratio}.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
