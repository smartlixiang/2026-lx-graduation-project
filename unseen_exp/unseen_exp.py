from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_adapter as train_adapter_module
import train_proxy as train_proxy_module
import weights.dynamic_utils as dynamic_utils

from dataset.dataset_config import CIFAR10, CIFAR100
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.seed import set_seed
from calculate_my_mask import select_group_mask
from utils.path_rules import resolve_mask_path, resolve_proxy_log_dir
from utils.score_utils import standard_zscore_by_class
from weights.calibration import build_dynamic_target, fit_softplus_ratio_regression
from weights import (
    AbsorptionGainScore,
    ConfusionComplementarityScore,
    TransferabilityScore,
)


VALID_DATASETS = (CIFAR10, CIFAR100)
KEEP_RATIOS = (60, 70, 80, 90)
COMPONENT_NAMES = ("A", "C", "T")
METHOD = "unseen_learned_group"
KNOWN_RATIO = 0.5
K_FOLDS = 5
RATIO_LAMBDA = 1e-3


def _atomic_savez(path: Path, **values: Any) -> None:
    """Write an npz only after it has been completely assembled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(tmp, **values)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _integer_vector(value: np.ndarray) -> bool:
    return value.ndim == 1 and np.issubdtype(value.dtype, np.integer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unseen-sample generalization experiment for the proposed selection framework."
    )
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=VALID_DATASETS)
    parser.add_argument("--seed", type=int, default=int(CONFIG.global_seed))
    parser.add_argument("--data-root", type=str, default=str(CONFIG.data_root))
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--dist-weight-factor", type=float, default=1.0)
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
    - known/unseen split is saved using original full-dataset indices;
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

    if skip_saved:
        try:
            with np.load(split_path, allow_pickle=False) as data:
                required = {"dataset", "seed", "num_samples", "known_ratio", "known_indices", "unseen_indices"}
                known_raw, unseen_raw = data["known_indices"], data["unseen_indices"]
                valid = required.issubset(data.files) and (
                    str(data["dataset"].item()) == dataset_name
                    and int(data["seed"].item()) == seed
                    and int(data["num_samples"].item()) == num_samples
                    and float(data["known_ratio"].item()) == KNOWN_RATIO
                    and _integer_vector(known_raw) and _integer_vector(unseen_raw)
                )
                known = np.asarray(known_raw, dtype=np.int64); unseen = np.asarray(unseen_raw, dtype=np.int64)
                valid = valid and len(known) == num_samples // 2 and len(unseen) == num_samples - num_samples // 2
                valid = valid and np.unique(known).size == len(known) and np.unique(unseen).size == len(unseen)
                valid = valid and np.array_equal(np.sort(np.concatenate((known, unseen))), np.arange(num_samples))
            if valid:
                print(f"[Skip] known/unseen split loaded: {split_path}")
                return known, unseen, split_path
        except Exception:
            pass
        print(f"[Recompute] known/unseen split: {split_path}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    half = num_samples // 2
    known = np.sort(perm[:half]).astype(np.int64)
    unseen = np.sort(perm[half:]).astype(np.int64)

    _atomic_savez(
        split_path,
        dataset=np.asarray(dataset_name),
        seed=np.asarray(seed, dtype=np.int64),
        num_samples=np.asarray(num_samples, dtype=np.int64),
        known_indices=known,
        unseen_indices=unseen,
        known_ratio=np.asarray(0.5, dtype=np.float32),
    )

    print(f"[Save] known/unseen split: {split_path}")
    return known, unseen, split_path


def adapter_cache_valid(adapter_dir: Path, dataset_name: str, seed: int, known_count: int, clip_model: str) -> bool:
    meta_path = adapter_dir / "meta.json"
    image_path = adapter_dir / "adapter_image.pt"
    text_path = adapter_dir / "adapter_context.pt"

    if not (meta_path.exists() and image_path.exists() and text_path.exists()):
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    try:
        return (
        meta.get("dataset") == dataset_name and int(meta.get("seed")) == int(seed)
        and meta.get("clip_model") == clip_model and int(meta.get("num_samples")) == int(known_count)
        and meta.get("adapter_type") == "linear"
        and meta.get("training_objective") == "InfoNCE"
        and meta.get("unseen_known_subset") is True
        and float(meta.get("known_ratio")) == KNOWN_RATIO)
    except (TypeError, ValueError):
        return False


def train_adapter_on_known(
    args: argparse.Namespace,
    known_indices: np.ndarray,
    result_root: Path,
) -> Path:
    adapter_dir = result_root / "adapter" / args.dataset / str(args.seed)

    if args.skip_saved and adapter_cache_valid(adapter_dir, args.dataset, args.seed, len(known_indices), args.clip_model):
        print(f"[Skip] known-subset adapter exists: {adapter_dir}")
        return adapter_dir
    if args.skip_saved:
        print(f"[Recompute] known-subset adapter: {adapter_dir}")
    # Metadata is the completion marker; never leave an old valid marker while retraining.
    (adapter_dir / "meta.json").unlink(missing_ok=True)

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
    _atomic_json(meta_path, meta)

    return adapter_dir


def proxy_log_dir_for(
    result_root: Path,
    dataset_name: str,
    proxy_model: str,
    seed: int,
    epochs: int,
) -> Path:
    return resolve_proxy_log_dir(dataset_name, seed=seed, proxy_model=proxy_model,
        epochs=epochs, root=SCRIPT_DIR / "proxy_logs")


def proxy_cache_valid(proxy_dir: Path, dataset: str, model: str, known_count: int, seed: int, epochs: int) -> bool:
    meta_path = proxy_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not (meta.get("dataset") == dataset and meta.get("model") == model
                and int(meta.get("num_samples")) == known_count and int(meta.get("seed")) == seed
                and int(meta.get("epochs")) == epochs and int(meta.get("k_folds")) == K_FOLDS
                and meta.get("unseen_known_subset") is True
                and meta.get("unseen_proxy_log_seed_specific") is True):
            return False
        num_classes = int(meta["num_classes"])
        validation = []
        for fold_no in range(1, K_FOLDS + 1):
            with np.load(proxy_dir / f"fold_{fold_no}.npz", allow_pickle=False) as fold:
                required = {"train_indices", "val_indices", "train_logits", "val_logits"}
                if not required.issubset(fold.files): return False
                train, val = fold["train_indices"], fold["val_indices"]
                train_logits, val_logits = fold["train_logits"], fold["val_logits"]
                if not (_integer_vector(train) and _integer_vector(val)): return False
                if (np.any(train < 0) or np.any(train >= known_count) or np.any(val < 0)
                        or np.any(val >= known_count) or np.intersect1d(train, val).size): return False
                if (np.unique(train).size != len(train) or np.unique(val).size != len(val)
                        or not np.array_equal(np.sort(np.concatenate((train, val))), np.arange(known_count))):
                    return False
                if train_logits.shape != (epochs, len(train), num_classes): return False
                if val_logits.shape != (epochs, len(val), num_classes): return False
                if not (np.isfinite(train_logits).all() and np.isfinite(val_logits).all()): return False
                validation.append(np.asarray(val, dtype=np.int64))
        return np.array_equal(np.sort(np.concatenate(validation)), np.arange(known_count))
    except Exception:
        return False


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

    if args.skip_saved and proxy_cache_valid(proxy_dir, args.dataset, args.proxy_model,
                                              len(known_indices), args.seed, resolved_epochs):
        print(f"[Skip] seed-specific known-subset proxy logs exist: {proxy_dir}")
        return proxy_dir, resolved_epochs
    if args.skip_saved:
        print(f"[Recompute] known-subset proxy logs: {proxy_dir}")
    # A proxy cache is all-or-nothing.  Remove the old set so an interrupted run
    # cannot be mistaken for five mutually consistent folds.
    (proxy_dir / "meta.json").unlink(missing_ok=True)
    for fold_no in range(1, K_FOLDS + 1):
        (proxy_dir / f"fold_{fold_no}.npz").unlink(missing_ok=True)

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

    def patched_resolve_proxy_log_dir(dataset: str, seed: int | None = None, *, proxy_model: str = "resnet18", epochs: int, root=None) -> Path:
        out = resolve_proxy_log_dir(dataset, seed=args.seed, proxy_model=proxy_model,
            epochs=int(epochs), root=SCRIPT_DIR / "proxy_logs")
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
    meta["unseen_proxy_log_seed"] = int(args.seed)
    _atomic_json(meta_path, meta)

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

    known_labels = get_targets(build_raw_dataset(args.dataset, args.data_root))[known_indices]
    if args.skip_saved:
        cached = load_known_static_cache(cache_path, args.dataset, args.seed, known_labels)
        if cached is not None:
            print(f"[Skip] known static scores loaded: {cache_path}")
            return cached
        print(f"[Recompute] known static scores: {cache_path}")

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
    _atomic_savez(
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


def load_known_static_cache(path: Path, dataset: str, seed: int,
                            known_labels: np.ndarray) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"sa", "div", "dds", "labels", "dataset", "seed", "known_num_samples"}
            if not required.issubset(data.files): return None
            result = {key: np.asarray(data[key]) for key in ("sa", "div", "dds", "labels")}
            n = len(known_labels)
            if (str(data["dataset"].item()) != dataset or int(data["seed"].item()) != seed
                    or int(data["known_num_samples"].item()) != n): return None
            if any(result[key].shape != (n,) for key in result): return None
            if not all(np.isfinite(result[key]).all() for key in ("sa", "div", "dds")): return None
            if not np.array_equal(result["labels"].astype(np.int64), known_labels): return None
            return {key: result[key].astype(np.int64 if key == "labels" else np.float32) for key in result}
    except Exception:
        return None


def load_dynamic_component(path: Path, known_labels: np.ndarray) -> SimpleNamespace | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"labels", "raw_foldwise", "fold_normalized", "aggregated", "final_normalized"}
            if not required.issubset(data.files): return None
            labels = np.asarray(data["labels"]); raw = np.asarray(data["raw_foldwise"])
            normalized = np.asarray(data["fold_normalized"]); aggregated = np.asarray(data["aggregated"])
            final = np.asarray(data["final_normalized"]); n = len(known_labels)
            if not np.array_equal(labels.astype(np.int64), known_labels): return None
            if aggregated.shape != (n,) or final.shape != (n,): return None
            if raw.ndim < 2 or normalized.ndim < 2 or raw.shape[0] != K_FOLDS or normalized.shape[0] != K_FOLDS: return None
            if raw.shape[1] != n or normalized.shape[1] != n: return None
            if not (np.isfinite(aggregated).all() and np.isfinite(final).all()): return None
            if np.isinf(raw).any() or np.isinf(normalized).any(): return None
            return SimpleNamespace(raw_foldwise=raw, fold_normalized=normalized,
                                   aggregated=aggregated, final_normalized=final)
    except Exception:
        return None


def get_dynamic_components(args: argparse.Namespace, proxy_dir: Path, proxy_epochs: int,
                           known_labels: np.ndarray, result_root: Path) -> dict[str, Any]:
    cache_dir = result_root / "dynamic_cache" / args.dataset / args.proxy_model / str(args.seed) / str(proxy_epochs)
    results: dict[str, Any] = {}
    if args.skip_saved:
        for name in COMPONENT_NAMES:
            cached = load_dynamic_component(cache_dir / f"{name}.npz", known_labels)
            if cached is not None:
                results[name] = cached
                print(f"[Skip] dynamic {name}: {cache_dir / f'{name}.npz'}")
    missing = [name for name in COMPONENT_NAMES if name not in results]
    if not missing: return results
    for name in missing: print(f"[Recompute] dynamic {name}: {cache_dir / f'{name}.npz'}")
    old_loader = dynamic_utils.load_dataset_labels
    dynamic_utils.load_dataset_labels = lambda dataset_name, data_root: known_labels.copy()
    try:
        folds, labels_all = dynamic_utils.load_cv_fold_logs(proxy_dir, args.dataset, str(args.data_root))
    finally:
        dynamic_utils.load_dataset_labels = old_loader
    calculators = {"A": AbsorptionGainScore, "C": ConfusionComplementarityScore, "T": TransferabilityScore}
    for name in missing:
        value = calculators[name]().compute(folds=folds, labels_all=labels_all)
        results[name] = value
        _atomic_savez(cache_dir / f"{name}.npz", labels=labels_all, raw_foldwise=value.raw_foldwise,
                      fold_normalized=value.fold_normalized, aggregated=value.aggregated,
                      final_normalized=value.final_normalized)
        print(f"[Save] dynamic {name}: {cache_dir / f'{name}.npz'}")
    return results


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
    known_labels = np.asarray(KnownSubsetDataset(build_raw_dataset(args.dataset, args.data_root), known_indices).targets, dtype=np.int64)
    if args.skip_saved:
        cached = load_scoring_weights(weight_path, args, len(known_indices), proxy_epochs)
        if cached is not None:
            print(f"[Skip] scoring weights loaded: {weight_path}")
            return cached
        print(f"[Recompute] scoring weights: {weight_path}")
    results = get_dynamic_components(args, proxy_dir, proxy_epochs, known_labels, result_root)
    labels_all = known_labels
    dynamic_target, _ = build_dynamic_target(results)
    static = compute_static_scores_for_known_weight_learning(
        args, known_indices, adapter_paths, class_names, device, result_root
    )
    if not np.array_equal(labels_all.astype(np.int64), static["labels"]):
        raise ValueError("Known-subset dynamic and static labels are inconsistent.")
    features = np.stack([
        standard_zscore_by_class(static["sa"], known_labels),
        standard_zscore_by_class(static["div"], known_labels),
        standard_zscore_by_class(static["dds"], known_labels),
    ], axis=1)
    fit = fit_softplus_ratio_regression(features, dynamic_target, ratio_lambda=RATIO_LAMBDA,
        learning_rate=2e-3, max_iter=10000, tol=1e-6, device=device)
    normalized = np.asarray(fit["normalized_weights"], dtype=np.float64)
    entry = {"sa": float(normalized[0]), "div": float(normalized[1]), "dds": float(normalized[2]),
             "bias": float(fit["bias"]), "ratio_lambda": RATIO_LAMBDA, "dataset": args.dataset,
             "seed": args.seed, "known_ratio": 0.5, "known_num_samples": len(known_indices),
             "proxy_model": args.proxy_model, "proxy_epochs": proxy_epochs}
    try:
        payload = json.loads(weight_path.read_text(encoding="utf-8")) if weight_path.exists() else {}
        if not isinstance(payload, dict): payload = {}
    except Exception:
        payload = {}
    payload.setdefault(args.dataset, {})[str(args.seed)] = entry
    _atomic_json(weight_path, payload)
    print(f"[Save] scoring weights: {weight_path}")
    return {key: entry[key] for key in ("sa", "div", "dds")}


def load_scoring_weights(path: Path, args: argparse.Namespace, known_count: int,
                         proxy_epochs: int) -> dict[str, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload[args.dataset][str(args.seed)]
        required = {"sa", "div", "dds", "bias", "ratio_lambda", "dataset", "seed", "known_ratio",
                    "known_num_samples", "proxy_model", "proxy_epochs"}
        if not required.issubset(entry): return None
        if not (entry["dataset"] == args.dataset and int(entry["seed"]) == args.seed
                and float(entry["known_ratio"]) == KNOWN_RATIO
                and int(entry["known_num_samples"]) == known_count
                and entry["proxy_model"] == args.proxy_model and int(entry["proxy_epochs"]) == proxy_epochs
                and float(entry["ratio_lambda"]) == RATIO_LAMBDA): return None
        values = np.asarray([entry[k] for k in ("sa", "div", "dds", "bias")], dtype=np.float64)
        weights = values[:3]
        if (not np.isfinite(values).all() or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-4, rtol=0.0)):
            return None
        return {key: float(entry[key]) for key in ("sa", "div", "dds")}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def compute_all_static_scores_with_known_base(args, known_indices, adapter_paths, class_names, device, result_root):
    cache_path = result_root / "static_scores" / "all_with_known_base" / args.dataset / str(args.seed) / "static_scores.npz"
    labels_expected = get_targets(build_raw_dataset(args.dataset, args.data_root))
    if args.skip_saved:
        cached = load_all_static_cache(cache_path, args.dataset, args.seed, labels_expected, known_indices)
        if cached is not None:
            print(f"[Skip] all-sample static scores loaded: {cache_path}")
            return cached
        print(f"[Recompute] all-sample static scores: {cache_path}")
    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device,
        dataset_name=args.dataset, data_root=str(args.data_root), debug_prompts=args.debug_prompts)
    image_adapter, text_adapter, _ = load_trained_adapters(dataset_name=args.dataset, clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim, seed=args.seed, map_location=device,
        adapter_image_path=adapter_paths["image_path"], adapter_text_path=adapter_paths["text_path"])
    image_adapter.to(device).eval(); text_adapter.to(device).eval()
    all_dataset = build_raw_dataset(args.dataset, args.data_root)
    labels = get_targets(all_dataset)
    sa_loader = build_score_loader(args.dataset, args.data_root, sa_metric.extractor.preprocess)
    sa = sa_metric.score_dataset(sa_loader, adapter_image=image_adapter, adapter_text=text_adapter).scores.numpy()
    dds_loader = build_score_loader(args.dataset, args.data_root, dds_metric.extractor.preprocess)
    features, encoded_labels = encode_all_images(dds_metric, dds_loader, image_adapter, device)
    if not np.array_equal(labels, encoded_labels.cpu().numpy()):
        raise ValueError("Full-dataset labels are inconsistent.")
    dds = np.zeros(len(labels), dtype=np.float32)
    known_mask = np.zeros(len(labels), dtype=bool); known_mask[known_indices] = True
    for class_id in tqdm(range(len(class_names)), desc="[Static] DDS with known base", unit="class"):
        query = np.flatnonzero(labels == class_id); reference = np.flatnonzero((labels == class_id) & known_mask)
        dds[query] = dds_metric._dds_from_reference_pca(features[query], features[reference]).cpu().numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(cache_path, sa=sa.astype(np.float32), dds=dds, labels=labels,
                  known_indices=known_indices, dataset=np.asarray(args.dataset), seed=np.asarray(args.seed))
    print(f"[Static] saved: {cache_path}")
    return {"sa": sa, "dds": dds, "labels": labels}


def load_all_static_cache(path: Path, dataset: str, seed: int, labels: np.ndarray,
                          known_indices: np.ndarray) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"sa", "dds", "labels", "known_indices", "dataset", "seed"}
            if not required.issubset(data.files): return None
            sa, dds, saved_labels = (np.asarray(data[key]) for key in ("sa", "dds", "labels"))
            if (str(data["dataset"].item()) != dataset or int(data["seed"].item()) != seed
                    or sa.shape != labels.shape or dds.shape != labels.shape or saved_labels.shape != labels.shape
                    or not np.isfinite(sa).all() or not np.isfinite(dds).all()
                    or not np.array_equal(saved_labels.astype(np.int64), labels)
                    or not np.array_equal(np.asarray(data["known_indices"], dtype=np.int64), known_indices)):
                return None
            return {"sa": sa.astype(np.float32), "dds": dds.astype(np.float32),
                    "labels": saved_labels.astype(np.int64)}
    except Exception:
        return None


def mask_path_for(dataset: str, seed: int, keep_ratio: int) -> Path:
    return resolve_mask_path(METHOD, dataset, "", seed, keep_ratio, root=SCRIPT_DIR / "mask")


def save_mask(dataset, seed, keep_ratio, mask, known_indices, unseen_indices) -> Path:
    out_path = mask_path_for(dataset, seed, keep_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    known_selected = int(mask[known_indices].sum())
    unseen_selected = int(mask[unseen_indices].sum())
    _atomic_savez(out_path, mask=mask.astype(np.uint8), dataset=np.asarray(dataset),
        seed=np.asarray(seed), keep_ratio=np.asarray(keep_ratio), method=np.asarray(METHOD),
        known_indices=known_indices, unseen_indices=unseen_indices,
        known_selected=np.asarray(known_selected), unseen_selected=np.asarray(unseen_selected))
    return out_path


def mask_cache_valid(path: Path, num_samples: int, dataset: str, seed: int, keep_ratio: int,
                     known_indices: np.ndarray, unseen_indices: np.ndarray) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"mask", "dataset", "seed", "keep_ratio", "method", "known_indices",
                        "unseen_indices", "known_selected", "unseen_selected"}
            if not required.issubset(data.files): return False
            mask = np.asarray(data["mask"])
            known_selected, unseen_selected = int(data["known_selected"].item()), int(data["unseen_selected"].item())
            expected = round(num_samples * keep_ratio / 100)
            return (mask.shape == (num_samples,) and set(np.unique(mask).tolist()).issubset({0, 1})
                    and int(mask.sum()) == expected and str(data["dataset"].item()) == dataset
                    and int(data["seed"].item()) == seed and int(data["keep_ratio"].item()) == keep_ratio
                    and str(data["method"].item()) == METHOD
                    and np.array_equal(np.asarray(data["known_indices"], dtype=np.int64), known_indices)
                    and np.array_equal(np.asarray(data["unseen_indices"], dtype=np.int64), unseen_indices)
                    and known_selected == int(mask[known_indices].sum())
                    and unseen_selected == int(mask[unseen_indices].sum())
                    and known_selected + unseen_selected == int(mask.sum()))
    except Exception:
        return False


def main() -> None:
    args = parse_args(); args.dataset = args.dataset.strip().lower()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    full_dataset = build_raw_dataset(args.dataset, args.data_root)
    labels_full = get_targets(full_dataset)
    known_indices, unseen_indices, _ = save_known_split(args.dataset, args.seed,
        SCRIPT_DIR / "known_dataset", len(full_dataset), args.skip_saved)
    valid_masks = {
        keep_ratio: args.skip_saved and mask_cache_valid(
            mask_path_for(args.dataset, args.seed, keep_ratio), len(full_dataset), args.dataset,
            args.seed, keep_ratio, known_indices, unseen_indices)
        for keep_ratio in KEEP_RATIOS
    }
    for keep_ratio, valid in valid_masks.items():
        if valid: print(f"[Skip] mask: {mask_path_for(args.dataset, args.seed, keep_ratio)}")
        elif args.skip_saved: print(f"[Recompute] mask: {mask_path_for(args.dataset, args.seed, keep_ratio)}")
    if all(valid_masks.values()):
        return
    class_names = list(resolve_class_names_for_prompts(args.dataset, args.data_root, full_dataset.classes))
    adapter_dir = train_adapter_on_known(args, known_indices, SCRIPT_DIR)
    adapter_paths = {"image_path": adapter_dir / "adapter_image.pt", "text_path": adapter_dir / "adapter_context.pt",
                     "meta_path": adapter_dir / "meta.json"}
    proxy_dir, proxy_epochs = train_proxy_on_known(args, known_indices, SCRIPT_DIR)
    weights = learn_weights_on_known(args, known_indices, proxy_dir, proxy_epochs,
        adapter_paths, class_names, device, SCRIPT_DIR)
    static = compute_all_static_scores_with_known_base(args, known_indices, adapter_paths,
        class_names, device, SCRIPT_DIR)
    if static["labels"].shape != labels_full.shape: raise RuntimeError("Static scores must cover the full training set.")
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    image_adapter, _, _ = load_trained_adapters(dataset_name=args.dataset, clip_model=args.clip_model,
        input_dim=div_metric.extractor.embed_dim, seed=args.seed, map_location=device,
        adapter_image_path=adapter_paths["image_path"], adapter_text_path=adapter_paths["text_path"])
    image_adapter.to(device).eval()
    div_loader = build_score_loader(args.dataset, args.data_root, div_metric.extractor.preprocess)
    for keep_ratio in KEEP_RATIOS:
        if valid_masks[keep_ratio]: continue
        mask, _, _ = select_group_mask(sa_raw_scores=static["sa"], div_metric=div_metric,
            div_loader=div_loader, image_adapter=image_adapter, labels=static["labels"], weights=weights,
            num_classes=len(class_names), keep_ratio=keep_ratio, device=device, seed=args.seed,
            weight_group="learned", dds_static_scores=static["dds"],
            group_candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count,
            dist_weight_factor=args.dist_weight_factor)
        expected = round(len(full_dataset) * keep_ratio / 100)
        if int(mask.sum()) != expected: raise RuntimeError(f"mask selected {mask.sum()}, expected {expected}")
        out = save_mask(args.dataset, args.seed, keep_ratio, mask, known_indices, unseen_indices)
        print(f"[Save] mask: {out}")


if __name__ == "__main__":
    main()
