#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate fixed image-corruption masks for CIFAR-100 and Tiny-ImageNet.

Examples:

CUDA_VISIBLE_DEVICES=0 python corruption_exp/cal_corruption_mask.py \
    --dataset cifar100 \
    --seed 22 \
    --weight-group learned \
    --kr 30,50,70

CUDA_VISIBLE_DEVICES=0 python corruption_exp/cal_corruption_mask.py \
    --dataset cifar100 \
    --seed 22 \
    --weight-group naive \
    --kr 30,50,70

CUDA_VISIBLE_DEVICES=1 python corruption_exp/cal_corruption_mask.py \
    --dataset tiny-imagenet \
    --seed 42 \
    --weight-group learned \
    --kr 30,50,70
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
CORRUPTION_EXP_ROOT = THIS_FILE.parent
PROJECT_ROOT = CORRUPTION_EXP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
CORRUPTION_DATA_ROOT = PROJECT_ROOT / "corruption_data"
ADAPTER_ROOT = CORRUPTION_EXP_ROOT / "adapters"
WEIGHTS_ROOT = CORRUPTION_EXP_ROOT / "weights"
PROXY_LOG_ROOT = WEIGHTS_ROOT / "proxy_logs"
DYNAMIC_CACHE_ROOT = WEIGHTS_ROOT / "dynamic_cache"
STATIC_SCORE_ROOT = CORRUPTION_EXP_ROOT / "static_scores"
MASK_ROOT = CORRUPTION_EXP_ROOT / "mask"
WEIGHTS_PATH = WEIGHTS_ROOT / "scoring_weights.json"
SUPPORTED_DATASETS = ("cifar100", "tiny-imagenet")
DATASET_NUMERIC_ID = {"cifar100": 100, "tiny-imagenet": 200}
EXPECTED_TRAIN_SIZES = {"cifar100": 50_000, "tiny-imagenet": 100_000}
from corruption_exp import corruption_opt  # noqa: E402
from dataset.dataset_config import CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import CLIPFeatureExtractor, load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.proxy_log_utils import resolve_seed_epoch_proxy_log_dir  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores, resolve_static_score_cache_dir  # noqa: E402
from utils.training_defaults import get_proxy_training_config  # noqa: E402
import calculate_my_mask as mask_mod  # noqa: E402
import learn_scoring_weights as learn_weights_mod  # noqa: E402
import train_adapter as train_adapter_mod  # noqa: E402
import train_proxy as train_proxy_mod  # noqa: E402


@dataclass(frozen=True)
class CorruptionInfo:
    dataset: str
    seed: int
    list_path: Path
    num_samples: int
    corruption_types: np.ndarray
    is_corrupted: np.ndarray
    type_counts: dict[str, int]


def parse_ratio_list(text: str) -> list[int]:
    ratios = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ratios:
        raise ValueError("--kr cannot be empty")
    if any(r <= 0 or r > 100 for r in ratios):
        raise ValueError(f"invalid keep ratios: {ratios}")
    return ratios


def tiny_train_root(data_root: Path = DATA_ROOT) -> Path:
    return data_root / "tiny-imagenet-200" / "train"


def is_tiny_train_root(root: str | Path) -> bool:
    return Path(root).expanduser().resolve() == tiny_train_root().resolve()


def build_raw_train_dataset(dataset_name: str, data_root: Path = DATA_ROOT):
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=None, target_transform=None)
    if dataset_name == TINY_IMAGENET:
        root = tiny_train_root(data_root)
        if not root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {root}")
        return datasets.ImageFolder(root=str(root), transform=None, target_transform=None)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def extract_labels(dataset: Dataset) -> np.ndarray:
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr) and len(getattr(dataset, attr)) == len(dataset):
            return np.asarray(getattr(dataset, attr), dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([y for _, y in dataset.samples], dtype=np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


def load_corruption_info(dataset_name: str, seed: int, *, num_samples: int | None = None, strict_expected_size: bool = True) -> CorruptionInfo:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError("corruption masks support only cifar100 and tiny-imagenet")
    if num_samples is None:
        num_samples = EXPECTED_TRAIN_SIZES[dataset_name]
    if strict_expected_size and num_samples != EXPECTED_TRAIN_SIZES[dataset_name]:
        raise ValueError(f"{dataset_name} train size {num_samples}, expected {EXPECTED_TRAIN_SIZES[dataset_name]}")
    path = CORRUPTION_DATA_ROOT / dataset_name / f"corruption_list_{int(seed)}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"corruption list not found: {path}")
    rows = np.loadtxt(path, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(f"corruption list must have exactly two integer columns, got shape={rows.shape}")
    sample_ids = rows[:, 0]
    type_ids = rows[:, 1]
    expected_total = int(round(num_samples * 0.2))
    expected_per_type = expected_total // corruption_opt.NUM_CORRUPTION_TYPES
    if rows.shape[0] != expected_total:
        raise ValueError(f"corruption ratio is not exactly 20%: rows={rows.shape[0]}, expected={expected_total}")
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("duplicate sample_id / one sample assigned multiple corruptions")
    if np.any(sample_ids < 0) or np.any(sample_ids >= num_samples):
        raise ValueError("sample_id out of range")
    if np.any(type_ids < 0) or np.any(type_ids >= corruption_opt.NUM_CORRUPTION_TYPES):
        raise ValueError("corruption_type must be one of 0,1,2,3,4")
    counts = {corruption_opt.CORRUPTION_ID_TO_NAME[i]: int(np.sum(type_ids == i)) for i in range(corruption_opt.NUM_CORRUPTION_TYPES)}
    bad = {k: v for k, v in counts.items() if v != expected_per_type}
    if bad:
        raise ValueError(f"corruption type counts must be equal ({expected_per_type} each), got {counts}")
    corruption_types = np.full(num_samples, -1, dtype=np.int16)
    corruption_types[sample_ids] = type_ids.astype(np.int16)
    return CorruptionInfo(dataset_name, int(seed), path, num_samples, corruption_types, corruption_types >= 0, counts)


class FixedCorruptionDataset(Dataset):
    """Apply manifest-defined image corruptions before caller transforms."""

    _FORWARD_ATTRS = ("classes", "class_to_idx", "targets", "data", "samples", "imgs", "labels")

    def __init__(self, raw_dataset: Dataset, transform=None, target_transform=None, *, corruption_info: CorruptionInfo):
        if len(raw_dataset) != corruption_info.num_samples:
            raise ValueError(f"raw dataset length {len(raw_dataset)} != corruption list length {corruption_info.num_samples}")
        self.raw_dataset = raw_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.corruption_info = corruption_info
        for attr in self._FORWARD_ATTRS:
            if hasattr(raw_dataset, attr):
                setattr(self, attr, getattr(raw_dataset, attr))

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.raw_dataset, name)

    def __getitem__(self, idx: int):
        image, target = self.raw_dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        ctype = int(self.corruption_info.corruption_types[int(idx)])
        if ctype >= 0:
            ss = np.random.SeedSequence([int(self.corruption_info.seed), int(idx), ctype, DATASET_NUMERIC_ID[self.corruption_info.dataset]])
            rng = np.random.default_rng(ss)
            image = corruption_opt.apply_corruption(image, ctype, rng=rng)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


@contextlib.contextmanager
def patched_training_corruption(dataset_name: str, corruption_info: CorruptionInfo, *, data_root: Path = DATA_ROOT) -> Iterator[None]:
    orig_cifar100 = datasets.CIFAR100
    orig_imagefolder = datasets.ImageFolder

    class PatchedCIFAR100(orig_cifar100):  # type: ignore[misc, valid-type]
        def __new__(cls, *args, **kwargs):
            transform = kwargs.pop("transform", None)
            target_transform = kwargs.pop("target_transform", None)
            train = bool(kwargs.get("train", False)) if "train" in kwargs else (bool(args[1]) if len(args) > 1 else False)
            raw = orig_cifar100(*args, transform=None, target_transform=None, **kwargs)
            if dataset_name == CIFAR100 and train:
                return FixedCorruptionDataset(raw, transform, target_transform, corruption_info=corruption_info)
            raw.transform = transform; raw.target_transform = target_transform
            return raw

    class PatchedImageFolder(orig_imagefolder):  # type: ignore[misc, valid-type]
        def __new__(cls, root, *args, **kwargs):
            transform = kwargs.pop("transform", None)
            target_transform = kwargs.pop("target_transform", None)
            raw = orig_imagefolder(root, *args, transform=None, target_transform=None, **kwargs)
            if dataset_name == TINY_IMAGENET and is_tiny_train_root(root):
                return FixedCorruptionDataset(raw, transform, target_transform, corruption_info=corruption_info)
            raw.transform = transform; raw.target_transform = target_transform
            return raw

    datasets.CIFAR100 = PatchedCIFAR100  # type: ignore[assignment]
    datasets.ImageFolder = PatchedImageFolder  # type: ignore[assignment]
    try:
        yield
    finally:
        datasets.CIFAR100 = orig_cifar100  # type: ignore[assignment]
        datasets.ImageFolder = orig_imagefolder  # type: ignore[assignment]


@contextlib.contextmanager
def patched_project_paths(corruption_info: CorruptionInfo | None = None) -> Iterator[None]:
    old_adapter_resolve = train_adapter_mod.resolve_adapter_dir
    old_proxy_resolve = train_proxy_mod.resolve_proxy_log_dir
    old_learn_project = learn_weights_mod.PROJECT_ROOT
    old_dyn_dir = learn_weights_mod.resolve_dynamic_component_cache_dir
    old_dyn_path = learn_weights_mod.resolve_dynamic_component_cache_path
    old_static_cache = getattr(learn_weights_mod, "get_or_compute_static_scores", None)

    def adapter_dir(dataset_name: str, seed: int) -> Path:
        p = ADAPTER_ROOT / dataset_name / str(int(seed)); p.mkdir(parents=True, exist_ok=True); return p
    def proxy_dir(dataset: str, seed: int | None = None, *, proxy_model: str = "resnet18", epochs: int, root=None) -> Path:
        if seed is None: raise ValueError("seed required")
        return PROXY_LOG_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    def dyn_dir(dataset: str, proxy_model: str, seed: int, epochs: int) -> Path:
        return DYNAMIC_CACHE_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    def dyn_path(dataset: str, proxy_model: str, seed: int, epochs: int, component_name: str) -> Path:
        return dyn_dir(dataset, proxy_model, seed, epochs) / f"{component_name.strip().upper()}.npz"

    def static_cache_wrapper(**kwargs):
        kwargs["cache_root"] = STATIC_SCORE_ROOT
        kwargs["use_file_hashes"] = False
        return get_or_compute_static_scores(**kwargs)


    train_adapter_mod.resolve_adapter_dir = adapter_dir
    train_proxy_mod.resolve_proxy_log_dir = proxy_dir
    learn_weights_mod.PROJECT_ROOT = CORRUPTION_EXP_ROOT
    learn_weights_mod.resolve_dynamic_component_cache_dir = dyn_dir
    learn_weights_mod.resolve_dynamic_component_cache_path = dyn_path
    if old_static_cache is not None:
        learn_weights_mod.get_or_compute_static_scores = static_cache_wrapper
    try:
        yield
    finally:
        train_adapter_mod.resolve_adapter_dir = old_adapter_resolve
        train_proxy_mod.resolve_proxy_log_dir = old_proxy_resolve
        learn_weights_mod.PROJECT_ROOT = old_learn_project
        learn_weights_mod.resolve_dynamic_component_cache_dir = old_dyn_dir
        learn_weights_mod.resolve_dynamic_component_cache_path = old_dyn_path
        if old_static_cache is not None:
            learn_weights_mod.get_or_compute_static_scores = old_static_cache


def build_context(info: CorruptionInfo, args: argparse.Namespace) -> dict[str, Any]:
    cfg = get_proxy_training_config(info.dataset)
    return {
        "dataset": info.dataset,
        "seed": int(info.seed),
        "num_samples": int(info.num_samples),
        "num_corrupted": int(info.is_corrupted.sum()),
        "corruption_ratio": float(info.is_corrupted.mean()),
        "corruption_type_counts": info.type_counts,
        "proxy_model": args.proxy_model,
        "proxy_epochs": int(cfg["epochs"]),
        "clip_model": args.clip_model,
        "ratio_lambda": float(args.ratio_lambda),
        "transferability_component": "TransferabilityScore",
    }


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except Exception:
        return None


def adapter_paths(dataset: str, seed: int) -> tuple[Path, Path]:
    d = ADAPTER_ROOT / dataset / str(int(seed)); return d / "adapter_image.pt", d / "adapter_context.pt"


def stage_adapter(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> None:
    del ctx
    image_p, text_p = adapter_paths(info.dataset, info.seed)
    valid = image_p.is_file() and text_p.is_file()
    reason = "missing adapter files"
    if valid:
        try:
            extractor = CLIPFeatureExtractor(model_name=args.clip_model, device=torch.device(args.device) if args.device else CONFIG.global_device)
            load_trained_adapters(info.dataset, args.clip_model, extractor.embed_dim, info.seed, adapter_image_path=image_p, adapter_text_path=text_p, map_location="cpu")
            tqdm.write(f"Adapter cache HIT: {image_p.parent}")
        except Exception as exc:
            valid = False
            reason = f"adapter load failed: {exc}"
    if args.force:
        valid = False
        reason = "forced by --force"
    if not valid:
        tqdm.write(f"Adapter cache MISS→TRAIN ({reason})")
        ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), clip_model=args.clip_model, prompt_template="a photo of a {}", batch_size=args.batch_size, num_workers=args.num_workers, epochs=30, lr=1e-4, weight_decay=0.0, hidden_dim=256, temperature=0.07, step_size=30, gamma=0.1, device=args.device, seed=str(info.seed), debug_prompts=args.debug_prompts)
        with patched_project_paths(), patched_training_corruption(info.dataset, info):
            train_adapter_mod.train_for_seed(ns, info.seed, False)


def build_class_names(dataset: str, data_root: Path = DATA_ROOT):
    ds = build_raw_train_dataset(dataset, data_root)
    return resolve_class_names_for_prompts(dataset_name=dataset, data_root=str(data_root), class_names=ds.classes)



def _cleanup_empty_dirs(path: Path, stop: Path) -> None:
    cur = path
    stop = stop.resolve()
    while cur.exists() and cur.resolve() != stop:
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def migrate_dynamic_component_cache(dataset: str, proxy_model: str, seed: int, epochs: int, component_name: str, labels: np.ndarray) -> None:
    if component_name not in {"A", "C"}:
        return
    base = DYNAMIC_CACHE_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    canonical = base / f"{component_name}.npz"
    if canonical.is_file() or not base.exists():
        return
    valid: list[Path] = []
    for candidate in base.rglob(f"{component_name}.npz"):
        if candidate == canonical:
            continue
        result, cached_labels, reason = learn_weights_mod._load_dynamic_component_cache_with_reason(
            cache_path=candidate,
            component_name=component_name,
            dataset=dataset,
            proxy_model=proxy_model,
            proxy_training_seed=seed,
            epochs=epochs,
        )
        if result is not None and cached_labels is not None and np.array_equal(cached_labels, labels):
            valid.append(candidate)
        else:
            tqdm.write(f"Dynamic cache migration skip {candidate}: {reason}")
    if not valid:
        return
    if len(valid) > 1:
        raise RuntimeError(f"multiple valid {component_name} dynamic cache candidates under {base}: {valid}")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(valid[0]), str(canonical))
    tqdm.write(f"Dynamic cache migrated: {valid[0]} -> {canonical}")
    _cleanup_empty_dirs(valid[0].parent, base)


def migrate_dynamic_caches(args: argparse.Namespace, info: CorruptionInfo) -> None:
    raw = build_raw_train_dataset(info.dataset)
    labels = extract_labels(raw)
    epochs = int(get_proxy_training_config(info.dataset)["epochs"])
    for component_name in ("A", "C"):
        migrate_dynamic_component_cache(info.dataset, args.proxy_model, info.seed, epochs, component_name, labels)


def _static_bundle_valid(cache_dir: Path, expected: dict[str, object], labels: np.ndarray) -> bool:
    required = ("SA_cache.npz", "Div_cache.npz", "DDS_cache.npz")
    if not all((cache_dir / name).is_file() for name in required):
        return False
    for filename in required:
        try:
            with np.load(cache_dir / filename, allow_pickle=False) as data:
                if not {"scores", "labels", "indices", "meta"}.issubset(set(data.files)):
                    return False
                meta = json.loads(str(data["meta"]))
                for key, value in expected.items():
                    if meta.get(key) != value:
                        return False
                cached_labels = np.asarray(data["labels"], dtype=np.int64)
                indices = np.asarray(data["indices"])
                scores = np.asarray(data["scores"])
                if not np.array_equal(cached_labels, labels):
                    return False
                if scores.shape != labels.shape or indices.shape != labels.shape:
                    return False
                if not np.array_equal(indices, np.arange(labels.shape[0], dtype=indices.dtype)):
                    return False
        except Exception:
            return False
    return True


def migrate_static_score_cache(args: argparse.Namespace, info: CorruptionInfo, div: Div, dds: DifficultyDirection, sa: SemanticAlignment, labels: np.ndarray, img_p: Path, txt_p: Path) -> Path:
    canonical = resolve_static_score_cache_dir(STATIC_SCORE_ROOT, info.dataset, info.seed, div.k, dds.eigval_lower_bound, dds.eigval_upper_bound)
    required = ("SA_cache.npz", "Div_cache.npz", "DDS_cache.npz")
    if all((canonical / name).is_file() for name in required):
        return canonical
    expected = {
        "dataset": info.dataset,
        "seed": int(info.seed),
        "clip_model": args.clip_model,
        "adapter_image_path": str(img_p),
        "adapter_text_path": str(txt_p),
        "div_k": float(div.k),
        "dds_k": int(dds.k),
        "dds_eigval_lower_bound": float(dds.eigval_lower_bound),
        "dds_eigval_upper_bound": float(dds.eigval_upper_bound),
        "prompt_template": sa.prompt_template,
        "num_samples": int(info.num_samples),
        "score_storage": "raw_static_scores_v1",
        "score_version": "raw_static_scores_v1",
    }
    root = STATIC_SCORE_ROOT / info.dataset / str(info.seed)
    if not root.exists():
        return canonical
    valid = []
    for sa_file in root.rglob("SA_cache.npz"):
        candidate = sa_file.parent
        if candidate == canonical:
            continue
        if _static_bundle_valid(candidate, expected, labels):
            valid.append(candidate)
    if not valid:
        return canonical
    if len(valid) > 1:
        raise RuntimeError(f"multiple valid static cache bundles under {root}: {valid}")
    canonical.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.move(str(valid[0] / name), str(canonical / name))
    tqdm.write(f"Static score cache migrated: {valid[0]} -> {canonical}")
    _cleanup_empty_dirs(valid[0], root)
    return canonical

def stage_static_scores(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    device = torch.device(args.device) if args.device else CONFIG.global_device
    class_names = build_class_names(info.dataset)
    dds = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device, dataset_name=info.dataset, data_root=str(DATA_ROOT), debug_prompts=args.debug_prompts)
    img_p, txt_p = adapter_paths(info.dataset, info.seed)
    image_adapter, text_adapter, _ = load_trained_adapters(info.dataset, args.clip_model, dds.extractor.embed_dim, info.seed, map_location=device, adapter_image_path=img_p, adapter_text_path=txt_p)
    image_adapter.to(device).eval(); text_adapter.to(device).eval()
    raw = build_raw_train_dataset(info.dataset); labels = extract_labels(raw)

    def loader(preprocess):
        ds = FixedCorruptionDataset(build_raw_train_dataset(info.dataset), preprocess, None, corruption_info=info)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    def compute():
        return {
            "dds": np.asarray(dds.score_dataset(tqdm(loader(dds.extractor.preprocess), desc="Scoring DDS", unit="batch"), adapter=image_adapter).scores),
            "div": np.asarray(div.score_dataset(tqdm(loader(div.extractor.preprocess), desc="Scoring Div", unit="batch"), adapter=image_adapter).scores),
            "sa": np.asarray(sa.score_dataset(tqdm(loader(sa.extractor.preprocess), desc="Scoring SA", unit="batch"), adapter_image=image_adapter, adapter_text=text_adapter).scores),
            "labels": labels,
        }
    cache_dir = migrate_static_score_cache(args, info, div, dds, sa, labels, img_p, txt_p)
    before = sorted(cache_dir.rglob("*.npz")) if cache_dir.exists() else []
    scores = get_or_compute_static_scores(cache_root=STATIC_SCORE_ROOT, dataset=info.dataset, seed=info.seed, clip_model=args.clip_model, adapter_image_path=str(img_p), adapter_text_path=str(txt_p), div_k=div.k, dds_k=dds.k, dds_eigval_lower_bound=dds.eigval_lower_bound, dds_eigval_upper_bound=dds.eigval_upper_bound, prompt_template=sa.prompt_template, num_samples=info.num_samples, compute_fn=compute, use_file_hashes=False)
    if not np.array_equal(np.asarray(scores["labels"], dtype=np.int64), labels):
        raise ValueError("static score labels do not match original clean labels")
    after = sorted(cache_dir.rglob("*.npz")) if cache_dir.exists() else []
    tqdm.write(f"Static scores cache {'HIT' if before and before == after else 'MISS→COMPUTED'}: {cache_dir}")
    return scores




def validate_proxy_logs(log_dir: Path, args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> tuple[bool, str]:
    meta = load_json(log_dir / "meta.json")
    if meta is None:
        return False, "missing/invalid proxy meta.json"
    target_epochs = int(ctx["proxy_epochs"])
    if meta.get("dataset") != info.dataset or meta.get("model") != args.proxy_model or int(meta.get("seed", -1)) != info.seed:
        return False, "proxy meta dataset/model/seed mismatch"
    if int(meta.get("epochs", -1)) < target_epochs or int(meta.get("k_folds", -1)) != int(args.k_folds):
        return False, "proxy meta epochs/folds mismatch"
    for fold in range(1, int(args.k_folds) + 1):
        p = log_dir / f"fold_{fold}.npz"
        if not p.is_file():
            return False, f"missing {p.name}"
        try:
            with np.load(p, allow_pickle=False) as data:
                required = {"train_indices", "val_indices", "train_logits", "val_logits"}
                if not required.issubset(set(data.files)):
                    return False, f"invalid {p.name}"
                train_indices = np.asarray(data["train_indices"])
                val_indices = np.asarray(data["val_indices"])
                if train_indices.ndim != 1 or val_indices.ndim != 1:
                    return False, f"indices must be 1D in {p.name}"
                train_shape = data["train_logits"].shape
                val_shape = data["val_logits"].shape
                if len(train_shape) != 3 or len(val_shape) != 3:
                    return False, f"invalid logits rank in {p.name}"
                if train_shape[0] != val_shape[0]:
                    return False, f"train/val epoch mismatch in {p.name}"
                if train_shape[2] != val_shape[2] or train_shape[2] <= 0:
                    return False, f"train/val class dimension mismatch in {p.name}"
                if train_shape[1] != train_indices.shape[0] or val_shape[1] != val_indices.shape[0]:
                    return False, f"logits/index length mismatch in {p.name}"
                if train_shape[0] < target_epochs or val_shape[0] < target_epochs:
                    return False, f"insufficient epochs in {p.name}"
        except Exception as exc:
            return False, f"cannot load {p.name}: {exc}"
    return True, "ok"


def stage_proxy(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> Path:
    seed_dir = PROXY_LOG_ROOT / info.dataset / args.proxy_model / str(info.seed)
    target_epochs = int(ctx["proxy_epochs"])
    candidates: list[Path] = []
    exact = seed_dir / str(target_epochs)
    if exact.exists():
        candidates.append(exact)
    if seed_dir.exists():
        longer = sorted((p for p in seed_dir.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) > target_epochs), key=lambda p: int(p.name))
        candidates.extend([p for p in longer if p not in candidates])
    reason = "no candidate proxy logs"
    if not args.force:
        for log_dir in candidates:
            valid, reason = validate_proxy_logs(log_dir, args, info, ctx)
            if valid:
                source_epochs = int(log_dir.name) if log_dir.name.isdigit() else target_epochs
                if source_epochs == target_epochs:
                    tqdm.write(f"[proxy] exact log hit: epochs={target_epochs}, path={log_dir}")
                else:
                    tqdm.write(f"[proxy] reuse longer log: source_epochs={source_epochs}, target_epochs={target_epochs}, path={log_dir}")
                tqdm.write(f"Proxy CV cache HIT: {log_dir}")
                return log_dir
            tqdm.write(f"Proxy CV candidate invalid: {log_dir} ({reason})")
    else:
        reason = "forced by --force"
    log_dir = seed_dir / str(target_epochs)
    tqdm.write(f"Proxy CV cache MISS→TRAIN ({reason})")
    ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), model=args.proxy_model, epochs=target_epochs, batch_size=args.batch_size, num_workers=args.num_workers, lr=None, momentum=None, weight_decay=None, lr_milestones=None, lr_gamma=None, device=args.device or "", k_folds=args.k_folds, seed=info.seed)
    ns = train_proxy_mod.apply_dataset_defaults(ns)
    with patched_project_paths(), patched_training_corruption(info.dataset, info):
        train_proxy_mod.run_for_seed(ns, info.seed)
    return log_dir




def load_valid_weights(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> dict[str, float] | None:
    data = load_json(WEIGHTS_PATH) or {}
    entry = ((data.get(info.dataset) or {}).get(str(info.seed)) if isinstance(data.get(info.dataset), dict) else None)
    if not isinstance(entry, dict):
        return None
    vals = {k: float(entry.get(k, np.nan)) for k in ("sa", "div", "dds")}
    if not all(np.isfinite(v) and v > 0 for v in vals.values()) or abs(sum(vals.values()) - 1.0) > 1e-4:
        return None
    if entry.get("transferability_component") != "TransferabilityScore":
        return None
    expected_ratio_lambda = float(args.ratio_lambda)
    try:
        saved_ratio_lambda = float(entry.get("ratio_lambda", np.nan))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(saved_ratio_lambda) or abs(saved_ratio_lambda - expected_ratio_lambda) > 1e-12:
        return None
    ectx = entry.get("corruption_context")
    if isinstance(ectx, dict):
        for key in ("dataset", "seed", "proxy_model", "proxy_epochs", "clip_model", "num_samples", "ratio_lambda", "transferability_component"):
            if ectx.get(key) != ctx.get(key):
                return None
    return vals


def stage_weights(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any], proxy_log_dir: Path | None = None) -> dict[str, float]:
    if args.weight_group == "naive":
        tqdm.write("skip proxy/dynamic/weight-learning for naive weights")
        return {"sa": 1/3, "div": 1/3, "dds": 1/3}
    if proxy_log_dir is None:
        raise ValueError("proxy_log_dir is required for learned corruption weights.")
    if not proxy_log_dir.is_dir():
        raise FileNotFoundError(f"validated proxy log directory not found: {proxy_log_dir}")
    if not args.force:
        valid = load_valid_weights(args, info, ctx)
        if valid is not None:
            tqdm.write(f"Learned weights cache HIT: {WEIGHTS_PATH}")
            return valid
    tqdm.write("Learned weights cache MISS→LEARN")
    migrate_dynamic_caches(args, info)
    img_p, txt_p = adapter_paths(info.dataset, info.seed)
    ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), proxy_log=str(proxy_log_dir), proxy_model=args.proxy_model, proxy_epochs=int(ctx["proxy_epochs"]), adapter_image_path=str(img_p), adapter_text_path=str(txt_p), clip_model=args.clip_model, batch_size=args.batch_size, num_workers=args.num_workers, div_k=0.05, dds_k=5, dds_important_eigval_ratio=0.8, coverage_tau_g=0.15, coverage_s_g=0.07, coverage_k_pct=0.05, coverage_q_low=0.002, coverage_q_high=0.998, ridge_lambda=0.01, learning_rate=1e-2, max_iter=10000, tol=1e-6, ratio_lambda=args.ratio_lambda, regression_learning_rate=2e-3, regression_max_iter=10000, regression_tol=1e-8, output=str(WEIGHTS_PATH), device=args.device, debug_prompts=args.debug_prompts, seed=str(info.seed), proxy_training_seed=None)
    tqdm.write(f"[weights] ratio_lambda={float(args.ratio_lambda):.12g}")
    with patched_project_paths(), patched_training_corruption(info.dataset, info):
        learn_weights_mod.run_once(ns, [info.seed])
    data = load_json(WEIGHTS_PATH) or {}
    entry = data.setdefault(info.dataset, {}).setdefault(str(info.seed), {})
    entry["corruption_context"] = {k: ctx[k] for k in ("dataset", "seed", "proxy_model", "proxy_epochs", "clip_model", "num_samples", "corruption_type_counts", "ratio_lambda", "transferability_component") if k in ctx}
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = load_valid_weights(args, info, ctx)
    if valid is None:
        raise RuntimeError("learned weights were saved but failed validation")
    return valid


def slim_group_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keep = ("solver", "final_rate", "subset_comprehensive_score", "distribution_shift", "candidate_pool_size", "group_init_count")
    out = {k: stats[k] for k in keep if k in stats}
    if "candidate_pool_size" not in out and "candidate_pool_size" in stats: out["candidate_pool_size"] = int(stats["candidate_pool_size"])
    return out


def mask_path(mode_name: str, dataset: str, seed: int, keep_ratio: int) -> Path:
    return MASK_ROOT / mode_name / dataset / str(int(seed)) / f"mask_{int(keep_ratio)}.npz"


def validate_mask_file(path: Path, info: CorruptionInfo, keep_ratio: int, weight_group: str, weights: dict[str, float]) -> tuple[bool, str]:
    if not path.is_file(): return False, "missing mask file"
    try:
        with np.load(path, allow_pickle=False) as data:
            mask = np.asarray(data["mask"], dtype=np.uint8); selected = np.asarray(data["selected_indices"], dtype=np.int64)
            if mask.shape != (info.num_samples,): return False, "mask length mismatch"
            if not set(np.unique(mask).tolist()).issubset({0, 1}): return False, "mask contains non-binary values"
            expected = int(round(info.num_samples * keep_ratio / 100.0))
            if int(mask.sum()) != expected or selected.size != expected: return False, "selected count mismatch"
            if not np.array_equal(np.asarray(data["corruption_types"], dtype=np.int16), info.corruption_types): return False, "corruption_types mismatch"
            if str(np.asarray(data["dataset"]).item()) != info.dataset or str(np.asarray(data["weight_group"]).item()) != weight_group: return False, "dataset/weight_group mismatch"
            if int(np.asarray(data["seed"]).item()) != info.seed or int(np.asarray(data["keep_ratio"]).item()) != int(keep_ratio): return False, "seed/keep_ratio mismatch"
            saved_weights = np.asarray(data["weights"], dtype=np.float64)
            if saved_weights.shape != (3,): return False, "weights shape mismatch"
            expected_weights = np.array([weights["sa"], weights["div"], weights["dds"]], dtype=np.float64)
            if not np.allclose(saved_weights, expected_weights): return False, "weights mismatch"
    except Exception as exc:
        return False, f"mask load failed: {exc}"
    return True, "ok"


def _print_corruption_type_counts(path: Path, mask: np.ndarray, corruption_types: np.ndarray) -> None:
    selected_types = np.asarray(corruption_types, dtype=np.int16)[np.asarray(mask, dtype=np.uint8).astype(bool)]
    names = ["gaussian_noise", "partial_occlusion", "resolution_degradation", "fog", "motion_blur"]
    parts = [f"{name}={int(np.sum(selected_types == idx))}" for idx, name in enumerate(names)]
    tqdm.write(f"[mask] saved {path} | " + ", ".join(parts))


def save_mask(path: Path, mask: np.ndarray, selected_by_class: dict[int, int], stats: dict[str, Any], weights: dict[str, float], args: argparse.Namespace, info: CorruptionInfo, keep_ratio: int) -> None:
    m = np.asarray(mask, dtype=np.uint8)
    ok, reason = validate_in_memory_mask(m, info.num_samples, keep_ratio)
    if not ok: raise ValueError(reason)
    selected = np.flatnonzero(m).astype(np.int64)
    csel = info.corruption_types[selected]
    scalar = dict(dataset=info.dataset, method=args.mode_name, weight_group=args.weight_group, seed=int(info.seed), keep_ratio=int(keep_ratio), num_selected=int(selected.size), num_corrupted_total=int(info.is_corrupted.sum()), num_corrupted_selected=int(np.sum(csel >= 0)), corruption_ratio_total=float(info.is_corrupted.mean()), corruption_ratio_in_mask=float(np.mean(csel >= 0)) if selected.size else 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_stats = slim_group_stats(stats); group_stats["group_init_count"] = int(args.group_init_count); group_stats["candidate_pool_size"] = int(args.group_candidate_pool_size)
    np.savez_compressed(path, mask=m, selected_indices=selected, corruption_types=info.corruption_types, is_corrupted=info.is_corrupted, weights=np.array([weights["sa"], weights["div"], weights["dds"]], dtype=np.float64), selected_by_class=np.array(json.dumps(selected_by_class), dtype=np.str_), group_stats=np.array(json.dumps(group_stats), dtype=np.str_), **{k: np.array(v) for k, v in scalar.items()})
    _print_corruption_type_counts(path, m, info.corruption_types)


def validate_in_memory_mask(mask: np.ndarray, num_samples: int, keep_ratio: int) -> tuple[bool, str]:
    if mask.shape != (num_samples,): return False, "mask length mismatch"
    if not set(np.unique(mask).tolist()).issubset({0, 1}): return False, "mask must be binary"
    expected = int(round(num_samples * keep_ratio / 100.0))
    if int(mask.sum()) != expected: return False, f"mask selected {int(mask.sum())}, expected {expected}"
    return True, "ok"


def stage_masks(args: argparse.Namespace, info: CorruptionInfo, scores: dict[str, np.ndarray], weights: dict[str, float], keep_ratios: list[int]) -> None:
    device = torch.device(args.device) if args.device else CONFIG.global_device
    labels = np.asarray(scores["labels"], dtype=np.int64)
    class_names = build_class_names(info.dataset)
    div = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    img_p, _ = adapter_paths(info.dataset, info.seed)
    image_adapter, _, _ = load_trained_adapters(info.dataset, args.clip_model, div.extractor.embed_dim, info.seed, map_location=device, adapter_image_path=img_p, adapter_text_path=adapter_paths(info.dataset, info.seed)[1])
    image_adapter.to(device).eval()
    ds = FixedCorruptionDataset(build_raw_train_dataset(info.dataset), div.extractor.preprocess, None, corruption_info=info)
    div_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    for kr in keep_ratios:
        out = mask_path(args.mode_name, info.dataset, info.seed, kr)
        valid, reason = validate_mask_file(out, info, kr, args.weight_group, weights)
        if args.skip_saved and valid:
            with np.load(out, allow_pickle=False) as data:
                cached_mask = np.asarray(data["mask"], dtype=np.uint8)
            _print_corruption_type_counts(out, cached_mask, info.corruption_types); continue
        if args.skip_saved and not valid:
            tqdm.write(f"Mask cache MISS→COMPUTE ({reason})")
        mask, selected_by_class, stats = mask_mod.select_group_mask_by_center_repair(np.asarray(scores["sa"], dtype=np.float32), div_metric=div, div_loader=div_loader, image_adapter=image_adapter, labels=labels, weights=weights, num_classes=len(class_names), keep_ratio=kr, device=device, seed=info.seed, dds_static_scores=np.asarray(scores["dds"], dtype=np.float32), group_candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count)
        save_mask(out, mask, selected_by_class, stats, weights, args, info, kr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate fixed image-corruption group masks.")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument("--seed", required=True)
    p.add_argument("--kr", default="30,50,70")
    p.add_argument("--weight-group", default="learned", choices=("naive", "learned"))
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--proxy-model", default="resnet18")
    p.add_argument("--model-name", default="resnet50", help="Only used for compatibility/path naming context.")
    p.add_argument("--mode-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--group-candidate-pool-size", type=int, default=10)
    p.add_argument("--group-init-count", type=int, default=2)
    p.add_argument("--debug-prompts", action="store_true")
    p.add_argument("--skip-saved", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--ratio-lambda", type=float, default=1e-3)
    return p.parse_args()


def run_seed(args: argparse.Namespace, seed: int, keep_ratios: list[int]) -> None:
    set_seed(seed)
    raw = build_raw_train_dataset(args.dataset)
    info = load_corruption_info(args.dataset, seed, num_samples=len(raw), strict_expected_size=True)
    ctx = build_context(info, args)
    args.mode_name = args.mode_name or f"corruption_{args.weight_group}_group"
    stages = ["Adapter", "Static scores", "Group masks"] if args.weight_group == "naive" else ["Adapter", "Static scores", "Proxy CV", "Dynamic supervision and weights", "Group masks"]
    with tqdm(total=len(stages), desc=f"seed={seed} stages", unit="stage") as bar:
        tqdm.write(f"[1/{len(stages)}] Adapter")
        stage_adapter(args, info, ctx); bar.update(1)
        tqdm.write(f"[2/{len(stages)}] Static scores")
        scores = stage_static_scores(args, info, ctx); bar.update(1)
        if args.weight_group == "learned":
            tqdm.write(f"[3/{len(stages)}] Proxy CV")
            proxy_log_dir = stage_proxy(args, info, ctx); bar.update(1)
            tqdm.write(f"[4/{len(stages)}] Dynamic supervision and weights")
            weights = stage_weights(args, info, ctx, proxy_log_dir=proxy_log_dir); bar.update(1)
        else:
            weights = stage_weights(args, info, ctx, proxy_log_dir=None)
        tqdm.write(f"[{len(stages)}/{len(stages)}] Group masks")
        stage_masks(args, info, scores, weights, keep_ratios); bar.update(1)


def main() -> None:
    args = parse_args()
    keep_ratios = parse_ratio_list(args.kr)
    seeds = parse_seed_list(args.seed)
    if not seeds: raise ValueError("--seed cannot be empty")
    for seed in seeds:
        run_seed(args, int(seed), keep_ratios)


if __name__ == "__main__":
    main()
