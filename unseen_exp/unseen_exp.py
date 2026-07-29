from __future__ import annotations

import argparse
import contextlib
import json
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

    if skip_saved and split_path.exists():
        data = np.load(split_path, allow_pickle=False)
        required = {"dataset", "seed", "num_samples", "known_indices", "unseen_indices"}
        if required.issubset(set(data.files)):
            if (
                str(data["dataset"].item()) == dataset_name
                and int(data["seed"].item()) == seed
                and int(data["num_samples"].item()) == num_samples
            ):
                known = np.asarray(data["known_indices"], dtype=np.int64)
                unseen = np.asarray(data["unseen_indices"], dtype=np.int64)
                print(f"[Skip] known/unseen split loaded: {split_path}")
                return known, unseen, split_path

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    half = num_samples // 2
    known = np.sort(perm[:half]).astype(np.int64)
    unseen = np.sort(perm[half:]).astype(np.int64)

    np.savez_compressed(
        split_path,
        dataset=np.asarray(dataset_name),
        seed=np.asarray(seed, dtype=np.int64),
        num_samples=np.asarray(num_samples, dtype=np.int64),
        known_indices=known,
        unseen_indices=unseen,
        known_ratio=np.asarray(0.5, dtype=np.float32),
    )

    print(f"[Split] saved known/unseen split: {split_path}")
    return known, unseen, split_path


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
    return resolve_proxy_log_dir(dataset_name, seed=seed, proxy_model=proxy_model,
        epochs=epochs, root=SCRIPT_DIR / "proxy_logs")


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
    old_loader = dynamic_utils.load_dataset_labels
    dynamic_utils.load_dataset_labels = lambda dataset_name, data_root: known_labels.copy()
    try:
        folds, labels_all = dynamic_utils.load_cv_fold_logs(proxy_dir, args.dataset, str(args.data_root))
    finally:
        dynamic_utils.load_dataset_labels = old_loader
    results = {
        "A": AbsorptionGainScore().compute(folds=folds, labels_all=labels_all),
        "C": ConfusionComplementarityScore().compute(folds=folds, labels_all=labels_all),
        "T": TransferabilityScore().compute(folds=folds, labels_all=labels_all),
    }
    cache_dir = result_root / "dynamic_cache" / args.dataset / args.proxy_model / str(args.seed) / str(proxy_epochs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in COMPONENT_NAMES:
        value = results[name]
        np.savez_compressed(cache_dir / f"{name}.npz", labels=labels_all, raw_foldwise=value.raw_foldwise,
                            fold_normalized=value.fold_normalized, aggregated=value.aggregated,
                            final_normalized=value.final_normalized)
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
    fit = fit_softplus_ratio_regression(features, dynamic_target, ratio_lambda=1e-3,
        learning_rate=2e-3, max_iter=10000, tol=1e-6, device=device)
    normalized = np.asarray(fit["normalized_weights"], dtype=np.float64)
    entry = {"sa": float(normalized[0]), "div": float(normalized[1]), "dds": float(normalized[2]),
             "bias": float(fit["bias"]), "ratio_lambda": 1e-3, "dataset": args.dataset,
             "seed": args.seed, "known_ratio": 0.5, "known_num_samples": len(known_indices),
             "proxy_model": args.proxy_model, "proxy_epochs": proxy_epochs}
    payload = json.loads(weight_path.read_text()) if weight_path.exists() else {}
    payload.setdefault(args.dataset, {})[str(args.seed)] = entry
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: entry[key] for key in ("sa", "div", "dds")}


def compute_all_static_scores_with_known_base(args, known_indices, adapter_paths, class_names, device, result_root):
    cache_path = result_root / "static_scores" / "all_with_known_base" / args.dataset / str(args.seed) / "static_scores.npz"
    if args.skip_saved and cache_path.exists():
        with np.load(cache_path) as data:
            return {key: np.asarray(data[key]) for key in ("sa", "dds", "labels")}
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
    np.savez_compressed(cache_path, sa=sa.astype(np.float32), dds=dds, labels=labels,
                        known_indices=known_indices, dataset=np.asarray(args.dataset), seed=np.asarray(args.seed))
    print(f"[Static] saved: {cache_path}")
    return {"sa": sa, "dds": dds, "labels": labels}


def mask_path_for(dataset: str, seed: int, keep_ratio: int) -> Path:
    return resolve_mask_path(METHOD, dataset, "", seed, keep_ratio, root=SCRIPT_DIR / "mask")


def save_mask(dataset, seed, keep_ratio, mask, known_indices, unseen_indices) -> Path:
    out_path = mask_path_for(dataset, seed, keep_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    known_selected = int(mask[known_indices].sum())
    unseen_selected = int(mask[unseen_indices].sum())
    np.savez_compressed(out_path, mask=mask.astype(np.uint8), dataset=np.asarray(dataset),
        seed=np.asarray(seed), keep_ratio=np.asarray(keep_ratio), method=np.asarray(METHOD),
        known_indices=known_indices, unseen_indices=unseen_indices,
        known_selected=np.asarray(known_selected), unseen_selected=np.asarray(unseen_selected))
    return out_path


def mask_cache_valid(path: Path, num_samples: int, dataset: str, seed: int, keep_ratio: int) -> bool:
    if not path.exists(): return False
    with np.load(path) as data:
        return ("mask" in data and data["mask"].shape == (num_samples,)
                and str(data["dataset"].item()) == dataset and int(data["seed"].item()) == seed
                and int(data["keep_ratio"].item()) == keep_ratio and str(data["method"].item()) == METHOD)


def main() -> None:
    args = parse_args(); args.dataset = args.dataset.strip().lower()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    full_dataset = build_raw_dataset(args.dataset, args.data_root)
    labels_full = get_targets(full_dataset)
    class_names = list(resolve_class_names_for_prompts(args.dataset, args.data_root, full_dataset.classes))
    known_indices, unseen_indices, _ = save_known_split(args.dataset, args.seed,
        SCRIPT_DIR / "known_dataset", len(full_dataset), args.skip_saved)
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
        path = mask_path_for(args.dataset, args.seed, keep_ratio)
        if args.skip_saved and mask_cache_valid(path, len(full_dataset), args.dataset, args.seed, keep_ratio):
            print(f"[Skip] mask exists: {path}"); continue
        mask, _, _ = select_group_mask(sa_raw_scores=static["sa"], div_metric=div_metric,
            div_loader=div_loader, image_adapter=image_adapter, labels=static["labels"], weights=weights,
            num_classes=len(class_names), keep_ratio=keep_ratio, device=device, seed=args.seed,
            weight_group="learned", dds_static_scores=static["dds"],
            group_candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count,
            dist_weight_factor=args.dist_weight_factor)
        expected = round(len(full_dataset) * keep_ratio / 100)
        if int(mask.sum()) != expected: raise RuntimeError(f"mask selected {mask.sum()}, expected {expected}")
        out = save_mask(args.dataset, args.seed, keep_ratio, mask, known_indices, unseen_indices)
        print(f"[Mask] saved: {out}")


if __name__ == "__main__":
    main()
