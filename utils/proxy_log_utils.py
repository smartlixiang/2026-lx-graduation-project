"""Utilities for loading proxy training logs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dataset.dataset import BaseDataLoader
from utils.global_config import CONFIG


def extract_labels(dataset) -> np.ndarray:
    num_samples = len(dataset)
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if len(values) == num_samples:
                return np.asarray(values, dtype=np.int64)
    labels = np.empty(num_samples, dtype=np.int64)
    for idx in range(num_samples):
        _, label = dataset[idx]
        if hasattr(label, "item"):
            label = label.item()
        labels[idx] = int(label)
    return labels


def load_dataset_labels(dataset_name: str, data_root: str) -> np.ndarray:
    loader = BaseDataLoader(
        dataset_name,
        data_path=Path(data_root),
        batch_size=CONFIG.default_batch_size,
        num_workers=CONFIG.num_workers,
        val_split=0.0,
        seed=CONFIG.global_seed,
    )
    train_loader, _, _ = loader.load()
    return extract_labels(train_loader.dataset)


def compute_loss_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if logits.ndim != 3:
        raise ValueError("logits must have shape (epochs, num_samples, num_classes)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (num_samples,)")
    if logits.shape[1] != labels.shape[0]:
        raise ValueError("labels length must match logits samples")
    logits_max = np.max(logits, axis=2, keepdims=True)
    shifted = logits - logits_max
    logsumexp = np.log(np.exp(shifted).sum(axis=2)) + logits_max.squeeze(2)
    label_idx = labels.reshape(1, -1, 1)
    correct_logits = np.take_along_axis(logits, label_idx, axis=2).squeeze(2)
    loss = logsumexp - correct_logits
    return loss.astype(np.float32)


def _assemble_logits_from_folds(log_dir: Path, labels_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta_path = log_dir / "meta.json"
    num_samples = labels_all.shape[0]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_samples = int(meta.get("num_samples", num_samples))

    fold_paths = sorted(log_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {log_dir}")

    logits_full: np.ndarray | None = None
    filled = np.zeros(num_samples, dtype=bool)
    for fold_path in fold_paths:
        data = np.load(fold_path)
        if "val_logits" not in data or "val_indices" not in data:
            raise ValueError(f"Fold file missing val_logits/val_indices: {fold_path}")
        val_logits = data["val_logits"].astype(np.float32)
        val_indices = data["val_indices"].astype(np.int64)
        if logits_full is None:
            epochs, _, num_classes = val_logits.shape
            logits_full = np.empty((epochs, num_samples, num_classes), dtype=np.float32)
        if val_indices.max(initial=-1) >= num_samples:
            raise ValueError(f"val_indices out of range in {fold_path}")
        if np.any(filled[val_indices]):
            raise ValueError(f"Duplicate val_indices found in {fold_path}")
        logits_full[:, val_indices, :] = val_logits
        filled[val_indices] = True

    if logits_full is None:
        raise RuntimeError("Failed to assemble logits from folds.")
    if not np.all(filled):
        missing = np.where(~filled)[0]
        raise RuntimeError(f"Missing logits for {missing.size} samples.")

    indices = np.arange(num_samples, dtype=np.int64)
    return logits_full, indices


def resolve_proxy_log_path(
    proxy_log_root: str | Path,
    dataset_name: str,
    seed: int,
    proxy_model: str = "resnet18",
    max_epoch: int | None = None,
) -> Path:
    candidate = Path(proxy_log_root)
    if candidate.exists():
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and any(candidate.glob("fold_*.npz")):
            return candidate

    base_dir = candidate / dataset_name / proxy_model / str(seed)
    if max_epoch is not None:
        epoch_dir = base_dir / str(max_epoch)
        if epoch_dir.exists():
            return epoch_dir
        raise FileNotFoundError(f"未找到代理训练日志路径: {epoch_dir}")

    if base_dir.exists() and base_dir.is_dir():
        epoch_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if epoch_dirs:
            epoch_dirs.sort(key=lambda p: int(p.name))
            return epoch_dirs[-1]

    legacy_dir = candidate / str(seed)
    if legacy_dir.exists() and legacy_dir.is_dir():
        matches = sorted(legacy_dir.glob("*.npz"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(f"未找到代理训练日志路径: {proxy_log_root}")


def load_proxy_log(proxy_log_path: str | Path, dataset_name: str, data_root: str) -> dict[str, np.ndarray]:
    path = Path(proxy_log_path)
    labels_all = load_dataset_labels(dataset_name, data_root)

    if path.is_dir():
        logits, indices = _assemble_logits_from_folds(path, labels_all)
        labels = labels_all
        loss = compute_loss_from_logits(logits, labels)
        return {"logits": logits, "labels": labels, "indices": indices, "loss": loss}

    data = np.load(path)
    if "logits_over_epochs" in data:
        logits = data["logits_over_epochs"].astype(np.float32)
    elif "logits" in data:
        logits = data["logits"].astype(np.float32)
    elif "val_logits" in data:
        logits = data["val_logits"].astype(np.float32)
    elif "train_logits" in data:
        logits = data["train_logits"].astype(np.float32)
    else:
        raise ValueError(f"Proxy log missing logits: {path}")

    if "val_indices" in data:
        indices = data["val_indices"].astype(np.int64)
    elif "train_indices" in data:
        indices = data["train_indices"].astype(np.int64)
    else:
        indices = data["indices"].astype(np.int64) if "indices" in data else np.arange(logits.shape[1])

    if "labels" in data:
        labels = data["labels"].astype(np.int64)
    else:
        labels = labels_all[indices]

    loss = data["loss"].astype(np.float32) if "loss" in data else compute_loss_from_logits(logits, labels)
    return {"logits": logits, "labels": labels, "indices": indices, "loss": loss}


__all__ = ["load_proxy_log", "resolve_proxy_log_path"]
