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
import hashlib
import json
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
CORRUPTION_PRIOR_RATIO = 0.2
CORRUPTION_RISK_FACTOR = float(1.0 - np.sqrt(CORRUPTION_PRIOR_RATIO))

from corruption_exp import corruption_opt  # noqa: E402
from dataset.dataset_config import CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import CLIPFeatureExtractor, load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.training_defaults import get_default_training_config  # noqa: E402
import calculate_my_mask as mask_mod  # noqa: E402
import learn_scoring_weights as learn_weights_mod  # noqa: E402
import train_adapter as train_adapter_mod  # noqa: E402
import train_proxy as train_proxy_mod  # noqa: E402


@dataclass(frozen=True)
class CorruptionInfo:
    dataset: str
    seed: int
    list_path: Path
    list_hash: str
    num_samples: int
    corruption_types: np.ndarray
    is_corrupted: np.ndarray
    type_counts: dict[str, int]


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    return CorruptionInfo(dataset_name, int(seed), path, sha1_file(path), num_samples, corruption_types, corruption_types >= 0, counts)


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
    old_gate_path = learn_weights_mod._noise_gate_cache_path
    old_mask_mean = mask_mod._mean_stats_cache_path

    def adapter_dir(dataset_name: str, seed: int) -> Path:
        p = ADAPTER_ROOT / dataset_name / str(int(seed)); p.mkdir(parents=True, exist_ok=True); return p
    def proxy_dir(dataset: str, seed: int | None = None, *, proxy_model: str = "resnet18", epochs: int, root=None) -> Path:
        if seed is None: raise ValueError("seed required")
        return PROXY_LOG_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))
    def dyn_dir(dataset: str, proxy_model: str, seed: int, epochs: int) -> Path:
        base = DYNAMIC_CACHE_ROOT / dataset / proxy_model / str(int(seed)) / str(int(epochs))
        if corruption_info is not None:
            return base / f"corruption_{corruption_info.list_hash}"
        return base
    def dyn_path(dataset: str, proxy_model: str, seed: int, epochs: int, component_name: str) -> Path:
        return dyn_dir(dataset, proxy_model, seed, epochs) / f"{component_name.strip().upper()}.npz"
    def gate_path(dataset: str, proxy_model: str, seed: int, epochs: int) -> Path:
        return dyn_dir(dataset, proxy_model, seed, epochs) / "noise_gate.npz"
    def mean_path(dataset_name: str, clip_model: str, adapter_image_path: str) -> Path:
        tag = clip_model.replace("/", "-").replace(" ", "_")
        h = sha1_file(Path(adapter_image_path))
        return STATIC_SCORE_ROOT / "group_mean_stats" / dataset_name / tag / f"img_adapter_{h}.npz"

    train_adapter_mod.resolve_adapter_dir = adapter_dir
    train_proxy_mod.resolve_proxy_log_dir = proxy_dir
    learn_weights_mod.PROJECT_ROOT = CORRUPTION_EXP_ROOT
    learn_weights_mod.resolve_dynamic_component_cache_dir = dyn_dir
    learn_weights_mod.resolve_dynamic_component_cache_path = dyn_path
    learn_weights_mod._noise_gate_cache_path = gate_path
    mask_mod._mean_stats_cache_path = mean_path
    try:
        yield
    finally:
        train_adapter_mod.resolve_adapter_dir = old_adapter_resolve
        train_proxy_mod.resolve_proxy_log_dir = old_proxy_resolve
        learn_weights_mod.PROJECT_ROOT = old_learn_project
        learn_weights_mod.resolve_dynamic_component_cache_dir = old_dyn_dir
        learn_weights_mod.resolve_dynamic_component_cache_path = old_dyn_path
        learn_weights_mod._noise_gate_cache_path = old_gate_path
        mask_mod._mean_stats_cache_path = old_mask_mean


def context_path(dataset: str, seed: int) -> Path:
    return CORRUPTION_EXP_ROOT / "contexts" / dataset / str(int(seed)) / "context.json"


def build_context(info: CorruptionInfo, args: argparse.Namespace) -> dict[str, Any]:
    cfg = get_default_training_config(info.dataset)
    return {
        "dataset": info.dataset, "seed": int(info.seed), "num_samples": int(info.num_samples),
        "num_corrupted": int(info.is_corrupted.sum()), "corruption_ratio": float(info.is_corrupted.mean()),
        "corruption_list_hash": info.list_hash, "corruption_type_counts": info.type_counts,
        "proxy_model": args.proxy_model, "proxy_epochs": int(cfg["epochs"]), "clip_model": args.clip_model,
        "use_noise_gate": True,
        "noise_gate_cache_version": learn_weights_mod.NOISE_GATE_CACHE_VERSION,
        "learn_window": int(args.learn_window),
        "learn_min_correct": int(args.learn_min_correct),
        "gate_low": float(args.gate_low),
        "gate_high": float(args.gate_high),
    }


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except Exception:
        return None


def save_context(ctx: dict[str, Any]) -> None:
    p = context_path(ctx["dataset"], int(ctx["seed"])); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")


def context_matches(ctx: dict[str, Any] | None, expected: dict[str, Any], *, keys: tuple[str, ...] | None = None) -> tuple[bool, str]:
    if ctx is None: return False, "missing context.json"
    keys = keys or tuple(expected.keys())
    for k in keys:
        if ctx.get(k) != expected.get(k):
            return False, f"context mismatch: {k}"
    return True, "ok"


def adapter_paths(dataset: str, seed: int) -> tuple[Path, Path]:
    d = ADAPTER_ROOT / dataset / str(int(seed)); return d / "adapter_image.pt", d / "adapter_context.pt"


def stage_adapter(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> dict[str, str]:
    image_p, text_p = adapter_paths(info.dataset, info.seed)
    ok_ctx, reason = context_matches(load_json(context_path(info.dataset, info.seed)), ctx, keys=("dataset", "seed", "corruption_list_hash", "clip_model"))
    valid = ok_ctx and image_p.is_file() and text_p.is_file()
    if valid:
        try:
            extractor = CLIPFeatureExtractor(model_name=args.clip_model, device=torch.device(args.device) if args.device else CONFIG.global_device)
            load_trained_adapters(info.dataset, args.clip_model, extractor.embed_dim, info.seed, adapter_image_path=image_p, adapter_text_path=text_p, map_location="cpu")
            tqdm.write(f"Adapter cache HIT: {image_p.parent}")
        except Exception as exc:
            valid = False; reason = f"adapter load failed: {exc}"
    if args.force: valid = False; reason = "forced by --force"
    if not valid:
        tqdm.write(f"Adapter cache MISS→TRAIN ({reason})")
        ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), clip_model=args.clip_model, prompt_template="a photo of a {}", batch_size=args.batch_size, num_workers=args.num_workers, epochs=30, lr=1e-4, weight_decay=0.0, hidden_dim=256, temperature=0.07, step_size=30, gamma=0.1, device=args.device, seed=str(info.seed), debug_prompts=args.debug_prompts)
        with patched_project_paths(), patched_training_corruption(info.dataset, info):
            train_adapter_mod.train_for_seed(ns, info.seed, False)
    adapter_hashes = {"adapter_image_sha1": sha1_file(image_p), "adapter_text_sha1": sha1_file(text_p)}
    ctx.update(adapter_hashes); save_context(ctx)
    return adapter_hashes


def build_class_names(dataset: str, data_root: Path = DATA_ROOT):
    ds = build_raw_train_dataset(dataset, data_root)
    return resolve_class_names_for_prompts(dataset_name=dataset, data_root=str(data_root), class_names=ds.classes)


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
    cache_root = STATIC_SCORE_ROOT / info.dataset / str(info.seed) / f"corruption_{info.list_hash[:12]}"

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
    before = sorted(cache_root.rglob("*.npz")) if cache_root.exists() else []
    scores = get_or_compute_static_scores(cache_root=cache_root, dataset=info.dataset, seed=info.seed, clip_model=args.clip_model, adapter_image_path=str(img_p), adapter_text_path=str(txt_p), div_k=div.k, dds_k=dds.k, dds_eigval_lower_bound=dds.eigval_lower_bound, dds_eigval_upper_bound=dds.eigval_upper_bound, prompt_template=sa.prompt_template, num_samples=info.num_samples, compute_fn=compute)
    if not np.array_equal(np.asarray(scores["labels"], dtype=np.int64), labels):
        raise ValueError("static score labels do not match original clean labels")
    after = sorted(cache_root.rglob("*.npz")) if cache_root.exists() else []
    tqdm.write(f"Static scores cache {'HIT' if before and before == after else 'MISS→COMPUTED'}: {cache_root}")
    return scores


def validate_proxy_logs(log_dir: Path, args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> tuple[bool, str]:
    meta = load_json(log_dir / "meta.json")
    if meta is None: return False, "missing/invalid proxy meta.json"
    if meta.get("dataset") != info.dataset or int(meta.get("seed", -1)) != info.seed: return False, "proxy meta dataset/seed mismatch"
    if int(meta.get("epochs", -1)) != int(ctx["proxy_epochs"]) or int(meta.get("k_folds", -1)) != int(args.k_folds): return False, "proxy meta epochs/folds mismatch"
    side = load_json(log_dir / "context.json")
    ok, reason = context_matches(side, ctx, keys=("dataset", "seed", "corruption_list_hash", "proxy_model", "proxy_epochs"))
    if not ok: return False, reason
    for fold in range(1, int(args.k_folds) + 1):
        p = log_dir / f"fold_{fold}.npz"
        if not p.is_file(): return False, f"missing {p.name}"
        try:
            with np.load(p, allow_pickle=True) as data:
                if "train_indices" not in data or "val_indices" not in data or int(data["train_logits"].shape[0]) != int(ctx["proxy_epochs"]):
                    return False, f"invalid {p.name}"
        except Exception as exc:
            return False, f"cannot load {p.name}: {exc}"
    return True, "ok"


def stage_proxy(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> Path:
    log_dir = PROXY_LOG_ROOT / info.dataset / args.proxy_model / str(info.seed) / str(ctx["proxy_epochs"])
    valid, reason = validate_proxy_logs(log_dir, args, info, ctx)
    if args.force: valid = False; reason = "forced by --force"
    if valid:
        tqdm.write(f"Proxy CV cache HIT: {log_dir}"); return log_dir
    tqdm.write(f"Proxy CV cache MISS→TRAIN ({reason})")
    ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), model=args.proxy_model, epochs=int(ctx["proxy_epochs"]), batch_size=args.batch_size, num_workers=args.num_workers, lr=None, momentum=None, weight_decay=None, lr_milestones=None, lr_gamma=None, device=args.device or "", k_folds=args.k_folds, seed=info.seed)
    ns = train_proxy_mod.apply_dataset_defaults(ns)
    with patched_project_paths(), patched_training_corruption(info.dataset, info):
        train_proxy_mod.run_for_seed(ns, info.seed)
    (log_dir / "context.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_dir


def load_valid_weights(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> dict[str, float] | None:
    data = load_json(WEIGHTS_PATH) or {}
    entry = ((data.get(info.dataset) or {}).get(str(info.seed)) if isinstance(data.get(info.dataset), dict) else None)
    if not isinstance(entry, dict): return None
    vals = {k: float(entry.get(k, np.nan)) for k in ("sa", "div", "dds")}
    if not all(np.isfinite(v) and v > 0 for v in vals.values()) or abs(sum(vals.values()) - 1.0) > 1e-4: return None
    ectx = entry.get("corruption_context")
    ok, _ = context_matches(ectx if isinstance(ectx, dict) else None, ctx, keys=("dataset", "seed", "corruption_list_hash", "proxy_model", "proxy_epochs", "clip_model", "adapter_image_sha1", "adapter_text_sha1", "use_noise_gate", "noise_gate_cache_version", "learn_window", "learn_min_correct", "gate_low", "gate_high"))
    return vals if ok else None


def stage_weights(args: argparse.Namespace, info: CorruptionInfo, ctx: dict[str, Any]) -> dict[str, float]:
    if args.weight_group == "naive":
        tqdm.write("skip proxy/dynamic/weight-learning for naive weights")
        return {"sa": 1/3, "div": 1/3, "dds": 1/3}
    if not args.force:
        valid = load_valid_weights(args, info, ctx)
        if valid is not None:
            tqdm.write(f"Learned weights cache HIT: {WEIGHTS_PATH}"); return valid
    tqdm.write("Learned weights cache MISS→LEARN (with noise gate)")
    img_p, txt_p = adapter_paths(info.dataset, info.seed)
    ns = argparse.Namespace(dataset=info.dataset, data_root=str(DATA_ROOT), proxy_log=str(PROXY_LOG_ROOT), proxy_model=args.proxy_model, proxy_epochs=int(ctx["proxy_epochs"]), adapter_image_path=str(img_p), adapter_text_path=str(txt_p), clip_model=args.clip_model, batch_size=args.batch_size, num_workers=args.num_workers, div_k=0.05, dds_k=5, dds_important_eigval_ratio=0.8, coverage_tau_g=0.15, coverage_s_g=0.07, coverage_k_pct=0.05, coverage_q_low=0.002, coverage_q_high=0.998, ridge_lambda=0.01, learning_rate=1e-2, max_iter=10000, tol=1e-6, learn_window=args.learn_window, learn_min_correct=args.learn_min_correct, gate_low=args.gate_low, gate_high=args.gate_high, ratio_lambda=5e-3, regression_learning_rate=2e-3, regression_max_iter=10000, regression_tol=1e-8, use_noise_gate=True, force_noise_gate=bool(args.force), output=str(WEIGHTS_PATH), device=args.device, debug_prompts=args.debug_prompts, seed=str(info.seed), proxy_training_seed=None)
    with patched_project_paths(info), patched_training_corruption(info.dataset, info):
        learn_weights_mod.run_once(ns, [info.seed])
    data = load_json(WEIGHTS_PATH) or {}; data.setdefault(info.dataset, {}).setdefault(str(info.seed), {})["corruption_context"] = ctx
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True); WEIGHTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = load_valid_weights(args, info, ctx)
    if valid is None: raise RuntimeError("learned weights were saved but failed validation")
    return valid


def slim_group_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keep = ("solver", "final_rate", "dist_weight", "subset_comprehensive_score", "distribution_shift", "candidate_pool_size", "group_init_count")
    out = {k: stats[k] for k in keep if k in stats}
    if "candidate_pool_size" not in out and "candidate_pool_size" in stats: out["candidate_pool_size"] = int(stats["candidate_pool_size"])
    return out


def mask_path(mode_name: str, dataset: str, seed: int, keep_ratio: int) -> Path:
    return MASK_ROOT / mode_name / dataset / str(int(seed)) / f"mask_{int(keep_ratio)}.npz"


def validate_mask_file(path: Path, info: CorruptionInfo, keep_ratio: int, weight_group: str, weights: dict[str, float], dist_weight_factor: float) -> tuple[bool, str]:
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
            if str(np.asarray(data["corruption_list_hash"]).item()) != info.list_hash: return False, "corruption hash mismatch"
            saved_weights = np.asarray(data["weights"], dtype=np.float64)
            if saved_weights.shape != (3,): return False, "weights shape mismatch"
            expected_weights = np.array([weights["sa"], weights["div"], weights["dds"]], dtype=np.float64)
            if not np.allclose(saved_weights, expected_weights): return False, "weights mismatch"
            if "dist_weight_factor" not in data: return False, "missing dist_weight_factor"
            saved_factor = float(np.asarray(data["dist_weight_factor"]).item())
            if not np.allclose(saved_factor, float(dist_weight_factor)): return False, "dist_weight_factor mismatch"
    except Exception as exc:
        return False, f"mask load failed: {exc}"
    return True, "ok"


def save_mask(path: Path, mask: np.ndarray, selected_by_class: dict[int, int], stats: dict[str, Any], weights: dict[str, float], args: argparse.Namespace, info: CorruptionInfo, keep_ratio: int) -> None:
    m = np.asarray(mask, dtype=np.uint8)
    ok, reason = validate_in_memory_mask(m, info.num_samples, keep_ratio)
    if not ok: raise ValueError(reason)
    selected = np.flatnonzero(m).astype(np.int64)
    csel = info.corruption_types[selected]
    counts = {f"num_selected_{corruption_opt.CORRUPTION_ID_TO_NAME[i].replace('partial_', '').replace('resolution_degradation', 'resolution')}": int(np.sum(csel == i)) for i in range(corruption_opt.NUM_CORRUPTION_TYPES)}
    # User requested exact legacy-ish names.
    counts = {
        "num_selected_gaussian": int(np.sum(csel == 0)),
        "num_selected_occlusion": int(np.sum(csel == 1)),
        "num_selected_resolution": int(np.sum(csel == 2)),
        "num_selected_fog": int(np.sum(csel == 3)),
        "num_selected_motion_blur": int(np.sum(csel == 4)),
    }
    scalar = dict(dataset=info.dataset, method=args.mode_name, weight_group=args.weight_group, seed=int(info.seed), keep_ratio=int(keep_ratio), num_selected=int(selected.size), num_corrupted_total=int(info.is_corrupted.sum()), num_corrupted_selected=int(np.sum(csel >= 0)), corruption_ratio_total=float(info.is_corrupted.mean()), corruption_ratio_in_mask=float(np.mean(csel >= 0)) if selected.size else 0.0, corruption_list_hash=info.list_hash, dist_weight_factor=float(CORRUPTION_RISK_FACTOR), **counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_stats = slim_group_stats(stats); group_stats["group_init_count"] = int(args.group_init_count); group_stats["candidate_pool_size"] = int(args.group_candidate_pool_size)
    np.savez_compressed(path, mask=m, selected_indices=selected, corruption_types=info.corruption_types, is_corrupted=info.is_corrupted, weights=np.array([weights["sa"], weights["div"], weights["dds"]], dtype=np.float64), selected_by_class=np.array(json.dumps(selected_by_class), dtype=np.str_), group_stats=np.array(json.dumps(group_stats), dtype=np.str_), **{k: np.array(v) for k, v in scalar.items()})
    summary = {**scalar, "weights": weights, "selected_by_class": {str(k): int(v) for k, v in selected_by_class.items()}, "group_stats": group_stats, "mask_path": str(path)}
    path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tqdm.write(f"mask saved to: {path} | corrupted={scalar['num_corrupted_selected']}/{scalar['num_selected']} | corruption_ratio={scalar['corruption_ratio_in_mask']:.4f} ({scalar['corruption_ratio_in_mask'] * 100:.2f}%)")


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
        valid, reason = validate_mask_file(out, info, kr, args.weight_group, weights, CORRUPTION_RISK_FACTOR)
        if args.skip_saved and valid:
            with np.load(out, allow_pickle=False) as data:
                num_corrupted_selected = int(np.asarray(data["num_corrupted_selected"]).item())
                num_selected = int(np.asarray(data["num_selected"]).item())
                ratio = float(np.asarray(data["corruption_ratio_in_mask"]).item())
            tqdm.write(f"Mask cache HIT (--skip-saved): {out} | corrupted={num_corrupted_selected}/{num_selected} | corruption_ratio={ratio:.4f} ({ratio * 100:.2f}%)"); continue
        if args.skip_saved and not valid:
            tqdm.write(f"Mask cache MISS→COMPUTE ({reason})")
        mask, selected_by_class, stats = mask_mod.select_group_mask(np.asarray(scores["sa"], dtype=np.float32), div_metric=div, div_loader=div_loader, image_adapter=image_adapter, labels=labels, weights=weights, num_classes=len(class_names), keep_ratio=kr, device=device, dataset_name=info.dataset, seed=info.seed, weight_group=args.weight_group, clip_model=args.clip_model, adapter_image_path=str(img_p), div_static_scores=np.asarray(scores["div"], dtype=np.float32), dds_static_scores=np.asarray(scores["dds"], dtype=np.float32), group_candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count, dist_weight_factor=CORRUPTION_RISK_FACTOR)
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
    p.add_argument("--group-candidate-pool-size", type=int, default=5)
    p.add_argument("--group-init-count", type=int, default=10)
    p.add_argument("--debug-prompts", action="store_true")
    p.add_argument("--skip-saved", action="store_true")
    p.add_argument("--learn-window", type=int, default=10)
    p.add_argument("--learn-min-correct", type=int, default=8)
    p.add_argument("--gate-low", type=float, default=0.2)
    p.add_argument("--gate-high", type=float, default=0.95)
    p.add_argument("--force", action="store_true")
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
            stage_proxy(args, info, ctx); bar.update(1)
            tqdm.write(f"[4/{len(stages)}] Dynamic supervision and weights")
            weights = stage_weights(args, info, ctx); bar.update(1)
        else:
            weights = stage_weights(args, info, ctx)
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
