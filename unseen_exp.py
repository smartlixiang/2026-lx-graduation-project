from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_adapter as train_adapter_module
import train_proxy as train_proxy_module
import weights.dynamic_utils as dynamic_utils

from dataset.dataset_config import CIFAR10, CIFAR100
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import build_class_prompts, resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.seed import set_seed
from weights import (
    AbsorptionGainScore,
    ConfusionComplementarityScore,
    PersistentDifficultyScore,
    TransferabilityAlignmentScore,
)


VALID_DATASETS = (CIFAR10, CIFAR100)
KEEP_RATIOS = (60, 70, 80, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unseen-sample generalization experiment for the proposed selection framework."
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=VALID_DATASETS)
    parser.add_argument("--seed", type=int, default=int(CONFIG.global_seed))
    parser.add_argument("--data-root", type=str, default=str(CONFIG.data_root))
    parser.add_argument("--result-root", type=str, default="unseen_result")
    parser.add_argument("--known-root", type=str, default="known_dataset")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args()


def build_raw_dataset(dataset_name: str, data_root: str | Path, transform=None):
    data_root = str(data_root)
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    raise ValueError(f"unseen_exp only supports cifar10/cifar100, got {dataset_name}")


def get_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"), dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(getattr(dataset, "labels"), dtype=np.int64)

    labels = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels[i] = int(y.item() if hasattr(y, "item") else y)
    return labels


class KnownSubsetDataset(Dataset):
    """
    A local-index dataset view over the known subset.

    In this experiment:
    - known/unknown split is saved using original full-dataset indices;
    - adapter and proxy CV are trained only on known samples;
    - proxy logs use local known-subset indices 0..len(known)-1;
    - final selection mask is still defined over the full training set.
    """

    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.classes = getattr(base_dataset, "classes", None)

        base_targets = get_targets(base_dataset)
        self.targets = base_targets[self.indices].astype(np.int64).tolist()

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        return self.base_dataset[int(self.indices[idx])]


@contextlib.contextmanager
def patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def save_known_split(
    dataset_name: str,
    seed: int,
    known_root: Path,
    num_samples: int,
    skip_saved: bool,
) -> tuple[np.ndarray, np.ndarray, Path]:
    out_dir = known_root / dataset_name / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split.npz"

    if skip_saved and split_path.exists():
        data = np.load(split_path, allow_pickle=False)
        required = {"dataset", "seed", "num_samples", "known_indices", "unknown_indices"}
        if required.issubset(set(data.files)):
            if (
                str(data["dataset"].item()) == dataset_name
                and int(data["seed"].item()) == seed
                and int(data["num_samples"].item()) == num_samples
            ):
                known = np.asarray(data["known_indices"], dtype=np.int64)
                unknown = np.asarray(data["unknown_indices"], dtype=np.int64)
                print(f"[Skip] known/unknown split loaded: {split_path}")
                return known, unknown, split_path

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    half = num_samples // 2
    known = np.sort(perm[:half]).astype(np.int64)
    unknown = np.sort(perm[half:]).astype(np.int64)

    np.savez_compressed(
        split_path,
        dataset=np.asarray(dataset_name),
        seed=np.asarray(seed, dtype=np.int64),
        num_samples=np.asarray(num_samples, dtype=np.int64),
        known_indices=known,
        unknown_indices=unknown,
        known_ratio=np.asarray(0.5, dtype=np.float32),
    )
    np.save(out_dir / "known_indices.npy", known)
    np.save(out_dir / "unknown_indices.npy", unknown)

    print(f"[Split] saved known/unknown split: {split_path}")
    return known, unknown, split_path


def adapter_cache_valid(adapter_dir: Path, dataset_name: str, seed: int, known_count: int) -> bool:
    meta_path = adapter_dir / "meta.json"
    image_path = adapter_dir / "adapter_image.pt"
    text_path = adapter_dir / "adapter_context.pt"

    if not (meta_path.exists() and image_path.exists() and text_path.exists()):
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        meta.get("dataset") == dataset_name
        and int(meta.get("seed")) == int(seed)
        and int(meta.get("num_samples")) == int(known_count)
        and meta.get("adapter_type") == "linear"
        and meta.get("training_objective") == "InfoNCE"
        and meta.get("unseen_known_subset") is True
    )


def train_adapter_on_known(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    result_root: Path,
) -> Path:
    adapter_dir = result_root / "adapter" / args.dataset / str(args.seed)

    if args.skip_saved and adapter_cache_valid(adapter_dir, args.dataset, args.seed, len(known_indices)):
        print(f"[Skip] known-subset adapter exists: {adapter_dir}")
        return adapter_dir

    def patched_build_dataset(dataset_name: str, data_root: str, transform):
        base = build_raw_dataset(dataset_name, data_root, transform=transform)
        return KnownSubsetDataset(base, known_indices)

    def patched_resolve_adapter_dir(dataset_name: str, seed: int) -> Path:
        out = result_root / "adapter" / dataset_name / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        return out

    adapter_args = SimpleNamespace(
        dataset=args.dataset,
        data_root=str(args.data_root),
        clip_model=args.clip_model,
        prompt_template="a photo of a {}",
        batch_size=None,
        num_workers=4,
        epochs=30,
        lr=1e-4,
        weight_decay=0.0,
        hidden_dim=256,
        temperature=0.07,
        step_size=30,
        gamma=0.1,
        device=args.device,
        seed=str(args.seed),
        debug_prompts=args.debug_prompts,
    )

    print("[Adapter] train adapter on known subset with train_adapter.py defaults")
    with patch_attr(train_adapter_module, "_build_dataset", patched_build_dataset):
        with patch_attr(train_adapter_module, "resolve_adapter_dir", patched_resolve_adapter_dir):
            train_adapter_module.train_for_seed(adapter_args, args.seed, multi_seed=False)

    meta_path = adapter_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["unseen_known_subset"] = True
    meta["known_ratio"] = 0.5
    meta["known_num_samples"] = int(len(known_indices))
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return adapter_dir


def proxy_log_dir_for(
    result_root: Path,
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> Path:
    return (
        result_root
        / "weights"
        / "proxy_logs"
        / dataset_name
        / proxy_model
        / str(seed)
        / str(int(epochs))
    )


def proxy_cache_valid(proxy_dir: Path, known_count: int, seed: int, epochs: int) -> bool:
    meta_path = proxy_dir / "meta.json"

    if not meta_path.exists():
        return False

    fold_paths = sorted(proxy_dir.glob("fold_*.npz"))
    if len(fold_paths) < 5:
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if int(meta.get("num_samples", -1)) != int(known_count):
        return False
    if int(meta.get("seed", -1)) != int(seed):
        return False
    if int(meta.get("epochs", -1)) != int(epochs):
        return False
    if meta.get("unseen_known_subset") is not True:
        return False
    if meta.get("unseen_proxy_log_seed_specific") is not True:
        return False

    return True


def train_proxy_on_known(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    result_root: Path,
) -> tuple[Path, int]:
    proxy_args = SimpleNamespace(
        dataset=args.dataset,
        data_root=str(args.data_root),
        model=args.proxy_model,
        epochs=None,
        batch_size=None,
        num_workers=4,
        lr=None,
        momentum=None,
        weight_decay=None,
        lr_milestones=None,
        lr_gamma=None,
        device=args.device or "",
        k_folds=5,
        seed=str(args.seed),
    )
    proxy_args = train_proxy_module.apply_dataset_defaults(proxy_args)
    resolved_epochs = int(proxy_args.epochs)

    proxy_dir = proxy_log_dir_for(
        result_root=result_root,
        dataset_name=args.dataset,
        proxy_model=args.proxy_model,
        seed=args.seed,
        epochs=resolved_epochs,
    )

    if args.skip_saved and proxy_cache_valid(proxy_dir, len(known_indices), args.seed, resolved_epochs):
        print(f"[Skip] seed-specific known-subset proxy logs exist: {proxy_dir}")
        return proxy_dir, resolved_epochs

    OriginalBaseDataLoader = train_proxy_module.BaseDataLoader

    class KnownBaseDataLoader:
        def __init__(
            self,
            dataset_name: str,
            data_path: Path,
            batch_size: int,
            num_workers: int,
            val_split: float,
            seed: int,
        ) -> None:
            self.inner = OriginalBaseDataLoader(
                dataset_name,
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                val_split=val_split,
                seed=seed,
            )
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.num_classes = None

        def load(self):
            train_loader, val_loader, test_loader = self.inner.load()
            self.num_classes = self.inner.num_classes

            known_dataset = KnownSubsetDataset(train_loader.dataset, known_indices)
            known_loader = DataLoader(
                known_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            )
            return known_loader, val_loader, test_loader

    def patched_resolve_proxy_log_dir(dataset: str, proxy_model: str, epochs: int) -> Path:
        out = proxy_log_dir_for(
            result_root=result_root,
            dataset_name=dataset,
            proxy_model=proxy_model,
            seed=args.seed,
            epochs=int(epochs),
        )
        out.mkdir(parents=True, exist_ok=True)
        return out

    print(
        "[Proxy] train seed-specific proxy CV on known subset "
        f"-> {args.dataset}/{args.proxy_model}/{args.seed}/{resolved_epochs}"
    )
    with patch_attr(train_proxy_module, "BaseDataLoader", KnownBaseDataLoader):
        with patch_attr(train_proxy_module, "resolve_proxy_log_dir", patched_resolve_proxy_log_dir):
            train_proxy_module.run_for_seed(proxy_args, args.seed)

    meta_path = proxy_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["unseen_known_subset"] = True
    meta["known_ratio"] = 0.5
    meta["known_num_samples"] = int(len(known_indices))
    meta["unseen_proxy_log_seed_specific"] = True
    meta["unseen_proxy_log_layout"] = "unseen_result/weights/proxy_logs/[dataset]/[model]/[seed]/[max_epoch]"
    meta["unseen_proxy_log_seed"] = int(args.seed)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return proxy_dir, resolved_epochs


def build_score_loader(dataset_name: str, data_root: str | Path, preprocess, batch_size: int = 128) -> DataLoader:
    dataset = build_raw_dataset(dataset_name, data_root, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=CONFIG.global_device.type == "cuda",
    )


def build_known_score_loader(
    dataset_name: str,
    data_root: str | Path,
    preprocess,
    known_indices: np.ndarray,
    batch_size: int = 128,
) -> DataLoader:
    base = build_raw_dataset(dataset_name, data_root, transform=preprocess)
    known_dataset = KnownSubsetDataset(base, known_indices)
    return DataLoader(
        known_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=CONFIG.global_device.type == "cuda",
    )


def quantile_minmax_ref(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ref_values = np.asarray(ref_values, dtype=np.float32)

    if ref_values.size == 0:
        return np.full_like(values, 0.5, dtype=np.float32)

    lo = float(np.quantile(ref_values, 0.002))
    hi = float(np.quantile(ref_values, 0.998))

    if abs(hi - lo) <= 1e-12:
        return np.full_like(values, 0.5, dtype=np.float32)

    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def classwise_ref_normalize(
    raw: np.ndarray,
    labels: np.ndarray,
    known_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    known_mask = np.zeros(labels.shape[0], dtype=bool)
    known_mask[known_indices] = True

    out = np.zeros_like(raw, dtype=np.float32)
    for c in range(num_classes):
        class_mask = labels == c
        ref_mask = class_mask & known_mask

        if not np.any(class_mask):
            continue

        ref_values = raw[ref_mask] if np.any(ref_mask) else raw[class_mask]
        out[class_mask] = quantile_minmax_ref(raw[class_mask], ref_values)

    return out


@torch.no_grad()
def encode_all_images(metric, loader: DataLoader, image_adapter, device: torch.device):
    image_adapter.eval()
    features, labels = metric._encode_images(tqdm(loader, desc="[Static] encode all images"), image_adapter)
    return features.to(device), labels.to(device)


def compute_static_scores_for_known_weight_learning(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    adapter_paths: dict[str, Path],
    class_names: list[str],
    device: torch.device,
    result_root: Path,
) -> dict[str, np.ndarray]:
    cache_path = (
        result_root
        / "static_scores"
        / "known_only"
        / args.dataset
        / str(args.seed)
        / "static_scores.npz"
    )

    if args.skip_saved and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        required = {"sa", "div", "dds", "labels", "dataset", "seed", "known_num_samples"}
        if required.issubset(set(data.files)):
            if (
                str(data["dataset"].item()) == args.dataset
                and int(data["seed"].item()) == args.seed
                and int(data["known_num_samples"].item()) == len(known_indices)
            ):
                print(f"[Skip] known static scores loaded: {cache_path}")
                return {
                    "sa": np.asarray(data["sa"], dtype=np.float32),
                    "div": np.asarray(data["div"], dtype=np.float32),
                    "dds": np.asarray(data["dds"], dtype=np.float32),
                    "labels": np.asarray(data["labels"], dtype=np.int64),
                }

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=str(args.data_root),
        debug_prompts=args.debug_prompts,
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
        adapter_image_path=adapter_paths["image_path"],
        adapter_text_path=adapter_paths["text_path"],
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    dds_loader = build_known_score_loader(
        args.dataset, args.data_root, dds_metric.extractor.preprocess, known_indices
    )
    div_loader = build_known_score_loader(
        args.dataset, args.data_root, div_metric.extractor.preprocess, known_indices
    )
    sa_loader = build_known_score_loader(
        args.dataset, args.data_root, sa_metric.extractor.preprocess, known_indices
    )

    dds = dds_metric.score_dataset(
        tqdm(dds_loader, desc="[Known static] DDS"),
        adapter=image_adapter,
    ).scores.numpy()
    div = div_metric.score_dataset(
        tqdm(div_loader, desc="[Known static] Div"),
        adapter=image_adapter,
    ).scores.numpy()
    sa = sa_metric.score_dataset(
        tqdm(sa_loader, desc="[Known static] SA"),
        adapter_image=image_adapter,
        adapter_text=text_adapter,
    ).scores.numpy()

    labels = np.asarray(
        KnownSubsetDataset(build_raw_dataset(args.dataset, args.data_root, None), known_indices).targets,
        dtype=np.int64,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sa=sa.astype(np.float32),
        div=div.astype(np.float32),
        dds=dds.astype(np.float32),
        labels=labels.astype(np.int64),
        dataset=np.asarray(args.dataset),
        seed=np.asarray(args.seed, dtype=np.int64),
        known_num_samples=np.asarray(len(known_indices), dtype=np.int64),
    )

    print(f"[Known static] saved: {cache_path}")
    return {"sa": sa, "div": div, "dds": dds, "labels": labels}


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)

    sorted_vec = np.sort(vector)[::-1]
    cumulative_sum = np.cumsum(sorted_vec)
    rho_candidates = sorted_vec - (cumulative_sum - 1) / np.arange(1, vector.size + 1)
    rho_indices = np.where(rho_candidates > 0)[0]

    if rho_indices.size == 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)

    rho = rho_indices[-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    denom = projected.sum()

    if denom <= 0:
        return np.full_like(vector, 1.0 / vector.size, dtype=np.float64)

    return projected / denom


def fit_ridge_regression_nonnegative(
    features: np.ndarray,
    targets: np.ndarray,
    l2_lambda: float = 1e-2,
    learning_rate: float = 1e-2,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    features = features.astype(np.float64)
    targets = targets.astype(np.float64)

    n, d = features.shape
    weights = np.full(d, 1.0 / d, dtype=np.float64)
    bias = float(targets.mean())

    for _ in tqdm(range(max_iter), desc="[Weight] fitting constrained ridge", leave=False):
        preds = features @ weights + bias
        errors = preds - targets
        grad_w = (features.T @ errors) / n + l2_lambda * weights
        grad_b = errors.mean()

        next_weights = project_to_simplex(weights - learning_rate * grad_w)
        next_bias = bias - learning_rate * grad_b

        if np.linalg.norm(next_weights - weights) < tol and abs(next_bias - bias) < tol:
            weights = next_weights
            bias = next_bias
            break

        weights = next_weights
        bias = next_bias

    return weights.astype(np.float32), float(bias)


def learn_weights_on_known(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    proxy_dir: Path,
    proxy_epochs: int,
    adapter_paths: dict[str, Path],
    class_names: list[str],
    device: torch.device,
    result_root: Path,
) -> dict[str, float]:
    weight_path = result_root / "weights" / "scoring_weights.json"

    if args.skip_saved and weight_path.exists():
        try:
            data = json.loads(weight_path.read_text(encoding="utf-8"))
            entry = data.get(args.dataset, {}).get(str(args.seed))
            if isinstance(entry, dict) and all(k in entry for k in ("sa", "div", "dds")):
                if (
                    entry.get("unseen_known_subset") is True
                    and entry.get("unseen_proxy_log_seed_specific") is True
                    and str(entry.get("proxy_log_dir")) == str(proxy_dir)
                ):
                    print(f"[Skip] learned weights loaded: {weight_path}")
                    return {
                        "sa": float(entry["sa"]),
                        "div": float(entry["div"]),
                        "dds": float(entry["dds"]),
                    }
        except Exception:
            pass

    known_labels = np.asarray(
        KnownSubsetDataset(build_raw_dataset(args.dataset, args.data_root, None), known_indices).targets,
        dtype=np.int64,
    )

    old_load_dataset_labels = dynamic_utils.load_dataset_labels
    dynamic_utils.load_dataset_labels = lambda dataset_name, data_root: known_labels.copy()

    try:
        folds, labels_all = dynamic_utils.load_cv_fold_logs(proxy_dir, args.dataset, str(args.data_root))
    finally:
        dynamic_utils.load_dataset_labels = old_load_dataset_labels

    a_result = AbsorptionGainScore().compute(folds=folds, labels_all=labels_all)
    c_result = ConfusionComplementarityScore().compute(folds=folds, labels_all=labels_all)
    t_result = TransferabilityAlignmentScore().compute(folds=folds, labels_all=labels_all)
    p_result = PersistentDifficultyScore().compute(folds=folds, labels_all=labels_all)

    dynamic_target = np.clip(
        (
            a_result.final_normalized
            + c_result.final_normalized
            + 0.5 * t_result.final_normalized
            + 0.5 * p_result.final_normalized
        ).astype(np.float32)
        / 3.0,
        0.0,
        1.0,
    )

    static_scores = compute_static_scores_for_known_weight_learning(
        args=args,
        known_indices=known_indices,
        adapter_paths=adapter_paths,
        class_names=class_names,
        device=device,
        result_root=result_root,
    )

    if not np.array_equal(labels_all.astype(np.int64), static_scores["labels"].astype(np.int64)):
        raise ValueError("Known-subset dynamic labels and known-subset static labels are inconsistent.")

    features = np.stack(
        [static_scores["sa"], static_scores["div"], static_scores["dds"]],
        axis=1,
    ).astype(np.float64)

    weights, bias = fit_ridge_regression_nonnegative(
        features=features,
        targets=dynamic_target.astype(np.float64),
        l2_lambda=1e-2,
        learning_rate=1e-2,
        max_iter=1000,
        tol=1e-6,
    )

    weight_payload: dict[str, Any] = {}
    if weight_path.exists():
        try:
            loaded = json.loads(weight_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                weight_payload = loaded
        except Exception:
            weight_payload = {}

    dataset_entry = weight_payload.get(args.dataset, {})
    dataset_entry[str(args.seed)] = {
        "sa": float(weights[0]),
        "div": float(weights[1]),
        "dds": float(weights[2]),
        "bias": float(bias),
        "ridge_lambda": 1e-2,
        "dynamic_target": "(A+C+0.5*T+0.5*P)/3",
        "constraint": "w >= 0 and sum(w)=1 solved jointly with ridge objective",
        "unseen_known_subset": True,
        "known_ratio": 0.5,
        "known_num_samples": int(len(known_indices)),
        "unseen_proxy_log_seed_specific": True,
        "proxy_log_layout": "unseen_result/weights/proxy_logs/[dataset]/[model]/[seed]/[max_epoch]",
        "proxy_log_dir": str(proxy_dir),
        "proxy_model": args.proxy_model,
        "proxy_seed": int(args.seed),
        "proxy_epochs": int(proxy_epochs),
    }
    weight_payload[args.dataset] = dataset_entry

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_text(json.dumps(weight_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    weights_dict = {
        "sa": float(weights[0]),
        "div": float(weights[1]),
        "dds": float(weights[2]),
    }
    print(f"[Weight] learned on known subset and seed-specific proxy logs: {weights_dict}, bias={bias:.6f}")
    return weights_dict


def compute_all_static_scores_with_known_base(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    adapter_paths: dict[str, Path],
    class_names: list[str],
    device: torch.device,
    result_root: Path,
) -> dict[str, np.ndarray]:
    cache_path = (
        result_root
        / "static_scores"
        / "all_with_known_base"
        / args.dataset
        / str(args.seed)
        / "static_scores.npz"
    )

    if args.skip_saved and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        required = {"sa", "div", "dds", "labels", "dataset", "seed", "known_indices"}
        if required.issubset(set(data.files)):
            if (
                str(data["dataset"].item()) == args.dataset
                and int(data["seed"].item()) == args.seed
                and np.array_equal(np.asarray(data["known_indices"], dtype=np.int64), known_indices)
            ):
                print(f"[Skip] all-sample static scores with known base loaded: {cache_path}")
                return {
                    "sa": np.asarray(data["sa"], dtype=np.float32),
                    "div": np.asarray(data["div"], dtype=np.float32),
                    "dds": np.asarray(data["dds"], dtype=np.float32),
                    "labels": np.asarray(data["labels"], dtype=np.int64),
                }

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=str(args.data_root),
        debug_prompts=args.debug_prompts,
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
        adapter_image_path=adapter_paths["image_path"],
        adapter_text_path=adapter_paths["text_path"],
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    all_dataset = build_raw_dataset(args.dataset, args.data_root, transform=None)
    labels_np = get_targets(all_dataset)
    num_classes = len(class_names)

    all_loader = build_score_loader(args.dataset, args.data_root, dds_metric.extractor.preprocess)
    image_features, labels_t = encode_all_images(dds_metric, all_loader, image_adapter, device)
    labels = labels_t.detach().cpu().numpy().astype(np.int64)

    if not np.array_equal(labels, labels_np):
        labels_np = labels

    # SA: adapter is trained on known subset. Normalization reference is also known subset.
    _, prompts = build_class_prompts(
        dataset_name=args.dataset,
        data_root=str(args.data_root),
        class_names=getattr(all_dataset, "classes"),
        prompt_template="a photo of a {}",
        debug=args.debug_prompts,
    )
    with torch.no_grad():
        text_features = sa_metric.extractor.encode_text(prompts).to(device)
        text_features = F.normalize(text_adapter(text_features), dim=-1)
        feats_norm = F.normalize(image_features.to(device), dim=-1)
        label_tensor = torch.as_tensor(labels_np, dtype=torch.long, device=device)

        sims = feats_norm @ text_features.T
        target_sim = sims.gather(1, label_tensor.view(-1, 1)).squeeze(1)
        neg_mask = torch.ones_like(sims, dtype=torch.bool)
        neg_mask.scatter_(1, label_tensor.view(-1, 1), False)
        neg_max = sims.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
        sa_raw = (target_sim - neg_max).detach().cpu().numpy().astype(np.float32)

    sa = classwise_ref_normalize(sa_raw, labels_np, known_indices, num_classes)

    # Div: query candidates are all samples, while the reference/base is known samples of the same class.
    div_raw = np.zeros(len(labels_np), dtype=np.float32)
    for c in tqdm(range(num_classes), desc="[Static] Div with known base", unit="class"):
        class_idx = np.flatnonzero(labels_np == c)
        ref_idx = np.intersect1d(class_idx, known_indices, assume_unique=False)

        if class_idx.size == 0:
            continue
        if ref_idx.size == 0:
            div_raw[class_idx] = 0.0
            continue

        q_feat = image_features[class_idx].to(device)
        r_feat = image_features[ref_idx].to(device)

        if hasattr(div_metric, "_knn_mean_distance_to_reference"):
            raw = div_metric._knn_mean_distance_to_reference(
                query_features=q_feat,
                reference_features=r_feat,
                k=div_metric.k,
                query_indices=torch.as_tensor(class_idx, dtype=torch.long, device=device),
                reference_indices=torch.as_tensor(ref_idx, dtype=torch.long, device=device),
            )
        else:
            dist = torch.cdist(q_feat.float(), r_feat.float(), p=2)
            k = max(1, min(r_feat.shape[0], int(math.ceil(float(div_metric.k) * r_feat.shape[0]))))
            raw = dist.topk(k=k, largest=False, dim=1).values.mean(dim=1)

        div_raw[class_idx] = raw.detach().cpu().numpy().astype(np.float32)

    div = classwise_ref_normalize(div_raw, labels_np, known_indices, num_classes)

    # DDS: PCA directions are fitted on known samples; all samples are scored by projection to known-base directions.
    dds_raw = np.zeros(len(labels_np), dtype=np.float32)
    for c in tqdm(range(num_classes), desc="[Static] DDS with known base", unit="class"):
        class_idx = np.flatnonzero(labels_np == c)
        ref_idx = np.intersect1d(class_idx, known_indices, assume_unique=False)

        if class_idx.size == 0:
            continue

        q_feat = image_features[class_idx].to(device)
        r_feat = image_features[ref_idx].to(device) if ref_idx.size > 0 else q_feat
        raw = dds_metric._dds_from_reference_pca(q_feat, r_feat)
        dds_raw[class_idx] = raw.detach().cpu().numpy().astype(np.float32)

    dds = classwise_ref_normalize(dds_raw, labels_np, known_indices, num_classes)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sa=sa.astype(np.float32),
        div=div.astype(np.float32),
        dds=dds.astype(np.float32),
        labels=labels_np.astype(np.int64),
        known_indices=known_indices.astype(np.int64),
        dataset=np.asarray(args.dataset),
        seed=np.asarray(args.seed, dtype=np.int64),
        known_num_samples=np.asarray(len(known_indices), dtype=np.int64),
        num_samples=np.asarray(len(labels_np), dtype=np.int64),
        meta_json=np.asarray(
            json.dumps(
                {
                    "stage": "all_samples_static_scored_with_known_base",
                    "normalization": "classwise quantile normalization uses known subset as reference",
                    "candidate_scope": "full training set",
                    "base_scope": "known subset",
                },
                ensure_ascii=False,
            )
        ),
    )

    print(f"[Static] saved all-sample scores with known base: {cache_path}")
    return {"sa": sa, "div": div, "dds": dds, "labels": labels_np}


def allocate_class_budgets(labels: np.ndarray, num_classes: int, keep_ratio: int) -> np.ndarray:
    sr = keep_ratio / 100.0
    class_sizes = np.asarray([np.sum(labels == c) for c in range(num_classes)], dtype=np.int64)
    target_size = int(round(sr * labels.shape[0]))

    raw = class_sizes.astype(np.float64) * sr
    budgets = np.floor(raw).astype(np.int64)
    budgets = np.minimum(budgets, class_sizes)

    need = int(target_size - budgets.sum())
    if need > 0:
        frac = raw - budgets.astype(np.float64)
        order = np.lexsort((np.arange(num_classes), -frac))
        for c in order:
            if need <= 0:
                break
            if budgets[c] < class_sizes[c]:
                budgets[c] += 1
                need -= 1

    if budgets.sum() != target_size:
        raise RuntimeError(f"class budget allocation failed: {budgets.sum()} != {target_size}")

    return budgets


def select_group_mask_known_base(
    static_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    known_indices: np.ndarray,
    keep_ratio: int,
    num_classes: int,
    group_candidate_pool_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(static_scores["labels"], dtype=np.int64)
    sa = np.asarray(static_scores["sa"], dtype=np.float32)
    div = np.asarray(static_scores["div"], dtype=np.float32)
    dds = np.asarray(static_scores["dds"], dtype=np.float32)

    n = labels.shape[0]
    budgets = allocate_class_budgets(labels, num_classes, keep_ratio)

    selected = np.zeros(n, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    score = weights["sa"] * sa + weights["div"] * div + weights["dds"] * dds

    selected_count_by_class = np.zeros(num_classes, dtype=np.int64)

    # Same spirit as calculate_my_mask.py group mode: small high-score initialization.
    # Candidates are all samples, not only known samples.
    for c in range(num_classes):
        class_idx = np.flatnonzero(labels == c)
        budget = int(budgets[c])

        if class_idx.size == 0 or budget <= 0:
            continue

        init_count = min(3, budget, class_idx.size)
        order = np.argsort(-score[class_idx], kind="mergesort")
        init_idx = class_idx[order[:init_count]]

        selected[init_idx] = 1
        selected_count_by_class[c] = init_count

    total_target = int(round(n * keep_ratio / 100.0))
    pbar = tqdm(
        total=total_target - int(selected.sum()),
        desc=f"[Group kr={keep_ratio}] greedy add",
        unit="sample",
    )

    while int(selected.sum()) < total_target:
        active_classes = np.flatnonzero(selected_count_by_class < budgets)
        if active_classes.size == 0:
            break

        for c in active_classes:
            if int(selected.sum()) >= total_target:
                break
            if selected_count_by_class[c] >= budgets[c]:
                continue

            class_idx = np.flatnonzero(labels == c)
            candidates = class_idx[selected[class_idx] == 0]

            if candidates.size == 0:
                continue

            candidate_score = score[candidates].copy()
            pool_n = max(1, min(int(group_candidate_pool_size), candidates.size))

            rank = np.argsort(-candidate_score, kind="mergesort")
            pool = candidates[rank[:pool_n]]
            picked = int(pool[0] if pool_n == 1 else rng.choice(pool, size=1)[0])

            selected[picked] = 1
            selected_count_by_class[c] += 1
            pbar.update(1)

    pbar.close()

    selected_by_class = {str(c): int(selected[labels == c].sum()) for c in range(num_classes)}

    known_mask = np.zeros(n, dtype=bool)
    known_mask[known_indices] = True
    known_selected = int(selected[known_indices].sum())
    unknown_selected = int(selected.sum()) - known_selected

    stats = {
        "solver": "unseen_known_base_group_greedy",
        "keep_ratio": int(keep_ratio),
        "target_size": int(total_target),
        "selected": int(selected.sum()),
        "known_selected": known_selected,
        "unknown_selected": unknown_selected,
        "selected_by_class": selected_by_class,
        "class_budgets": {str(c): int(budgets[c]) for c in range(num_classes)},
        "weights": weights,
        "candidate_scope": "full training set",
        "base_scope": "known subset",
        "note": "kr is defined relative to the full training set, while static/dynamic bases are fitted on known subset.",
    }

    return selected.astype(np.uint8), stats


def save_mask(
    result_root: Path,
    dataset_name: str,
    model_name: str,
    seed: int,
    keep_ratio: int,
    mask: np.ndarray,
    stats: dict[str, Any],
) -> Path:
    out_path = (
        result_root
        / "mask"
        / "learned_group"
        / dataset_name
        / model_name
        / str(seed)
        / f"mask_{keep_ratio}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        mask=mask.astype(np.uint8),
        dataset=np.asarray(dataset_name),
        seed=np.asarray(seed, dtype=np.int64),
        keep_ratio=np.asarray(keep_ratio, dtype=np.int64),
        method=np.asarray("learned_group"),
        meta_json=np.asarray(json.dumps(stats, ensure_ascii=False)),
    )

    return out_path


def mask_cache_valid(path: Path, num_samples: int, dataset: str, seed: int, keep_ratio: int) -> bool:
    if not path.exists():
        return False

    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return False

    if "mask" not in data.files:
        return False

    mask = np.asarray(data["mask"])
    if mask.shape != (num_samples,):
        return False

    if "dataset" in data.files and str(data["dataset"].item()) != dataset:
        return False
    if "seed" in data.files and int(data["seed"].item()) != seed:
        return False
    if "keep_ratio" in data.files and int(data["keep_ratio"].item()) != keep_ratio:
        return False

    return True


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.strip().lower()
    args.data_root = str(Path(args.data_root))

    result_root = PROJECT_ROOT / args.result_root
    known_root = PROJECT_ROOT / args.known_root

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device

    start = time.perf_counter()

    full_dataset = build_raw_dataset(args.dataset, args.data_root, transform=None)
    labels_full = get_targets(full_dataset)

    class_names = resolve_class_names_for_prompts(
        dataset_name=args.dataset,
        data_root=args.data_root,
        class_names=full_dataset.classes,  # type: ignore[attr-defined]
    )
    num_classes = len(class_names)

    known_indices, unknown_indices, split_path = save_known_split(
        dataset_name=args.dataset,
        seed=args.seed,
        known_root=known_root,
        num_samples=len(full_dataset),
        skip_saved=args.skip_saved,
    )

    print(
        f"[Init] dataset={args.dataset}, seed={args.seed}, "
        f"full={len(full_dataset)}, known={len(known_indices)}, unknown={len(unknown_indices)}, "
        f"classes={num_classes}, device={device}"
    )

    adapter_dir = train_adapter_on_known(args, known_indices, result_root)
    adapter_paths = {
        "image_path": adapter_dir / "adapter_image.pt",
        "text_path": adapter_dir / "adapter_context.pt",
        "meta_path": adapter_dir / "meta.json",
    }

    proxy_dir, proxy_epochs = train_proxy_on_known(args, known_indices, result_root)

    weights = learn_weights_on_known(
        args=args,
        known_indices=known_indices,
        proxy_dir=proxy_dir,
        proxy_epochs=proxy_epochs,
        adapter_paths=adapter_paths,
        class_names=class_names,
        device=device,
        result_root=result_root,
    )

    static_scores = compute_all_static_scores_with_known_base(
        args=args,
        known_indices=known_indices,
        adapter_paths=adapter_paths,
        class_names=class_names,
        device=device,
        result_root=result_root,
    )

    if static_scores["labels"].shape[0] != labels_full.shape[0]:
        raise RuntimeError("Final static scores must cover the full training set.")

    for kr in KEEP_RATIOS:
        mask_path = (
            result_root
            / "mask"
            / "learned_group"
            / args.dataset
            / args.model_name
            / str(args.seed)
            / f"mask_{kr}.npz"
        )

        if args.skip_saved and mask_cache_valid(mask_path, len(full_dataset), args.dataset, args.seed, kr):
            print(f"[Skip] mask exists: {mask_path}")
            continue

        mask, stats = select_group_mask_known_base(
            static_scores=static_scores,
            weights=weights,
            known_indices=known_indices,
            keep_ratio=kr,
            num_classes=num_classes,
            group_candidate_pool_size=args.group_candidate_pool_size,
            seed=args.seed,
        )

        expected = int(round(len(full_dataset) * kr / 100.0))
        if int(mask.sum()) != expected:
            raise RuntimeError(f"mask selected {int(mask.sum())}, expected {expected}")

        out_path = save_mask(
            result_root=result_root,
            dataset_name=args.dataset,
            model_name=args.model_name,
            seed=args.seed,
            keep_ratio=kr,
            mask=mask,
            stats=stats,
        )

        print(
            f"[Mask] kr={kr}, selected={int(mask.sum())}, "
            f"known_selected={stats['known_selected']}, "
            f"unknown_selected={stats['unknown_selected']}, "
            f"saved={out_path}"
        )

    elapsed = time.perf_counter() - start
    print(f"[Done] unseen experiment finished in {elapsed:.2f}s")
    print(f"[Output] known split: {split_path}")
    print(f"[Output] seed-specific proxy logs: {proxy_dir}")
    print(f"[Output] result root: {result_root}")


if __name__ == "__main__":
    main()