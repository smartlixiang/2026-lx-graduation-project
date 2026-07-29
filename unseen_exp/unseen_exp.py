#!/usr/bin/env python3
"""Prepare scorers and masks for the three explicit unseen-sample protocols.

Experiment 1 calibrates on 50% known data and selects from the clean full
CIFAR-100/Tiny-ImageNet training set (60/70/80/90%).  Experiment 2 calibrates
on 20% known CIFAR-10 data, recomputes every static reference on the disjoint
80% unseen pool, and selects global 20/30/40/60% targets from that pool.
Experiment 3 corrupts the full CIFAR-100 training set first, then makes the
50/50 views, and uses center-repair selection (30/40/50/60%).

This entry point never trains a final classifier; it ends after writing masks.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import calculate_my_mask as mask_solvers
import train_adapter as train_adapter_module
import train_proxy as train_proxy_module
import weights.dynamic_utils as dynamic_utils
from corruption_exp.cal_corruption_mask import (FixedCorruptionDataset, CorruptionInfo,
                                                  load_corruption_info)
from dataset.dataset_config import CIFAR10, CIFAR100, TINY_IMAGENET
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.path_rules import resolve_proxy_log_dir
from utils.score_utils import standard_zscore_by_class
from utils.seed import set_seed
from weights import AbsorptionGainScore, ConfusionComplementarityScore, TransferabilityScore
from weights.calibration import build_dynamic_target, fit_softplus_ratio_regression


@dataclass(frozen=True)
class UnseenExperimentConfig:
    exp_id: int
    allowed_datasets: tuple[str, ...]
    known_ratio: float
    unseen_ratio: int
    default_keep_ratios: tuple[int, ...]
    selection_scope: str
    static_reference_scope: str
    use_corruption: bool
    group_solver: str
    allowed_modes: tuple[str, ...]


EXPERIMENT_CONFIGS = {
    1: UnseenExperimentConfig(1, (CIFAR100, TINY_IMAGENET), .5, 50, (60, 70, 80, 90),
                              "full", "full", False, "standard", ("learned_group", "random")),
    2: UnseenExperimentConfig(2, (CIFAR10,), .2, 80, (20, 30, 40, 60),
                              "unseen", "unseen", False, "standard", ("learned_group", "random")),
    3: UnseenExperimentConfig(3, (CIFAR100,), .5, 50, (30, 40, 50, 60),
                              "full", "full_corrupted", True, "center_repair",
                              ("learned_group", "naive_group", "random")),
}
MODE_METHODS = {"learned_group": "unseen_learned_group",
                "naive_group": "unseen_naive_group", "random": "unseen_random"}
COMPONENTS = {"A": AbsorptionGainScore, "C": ConfusionComplementarityScore,
              "T": TransferabilityScore}
K_FOLDS = 5


def parse_keep_ratios(text: str | None, config: UnseenExperimentConfig) -> tuple[int, ...]:
    if text is None:
        return config.default_keep_ratios
    try:
        ratios = tuple(dict.fromkeys(int(x.strip()) for x in text.split(",") if x.strip()))
    except ValueError as exc:
        raise ValueError("--kr must be comma-separated integers") from exc
    if not ratios or not set(ratios).issubset(config.default_keep_ratios):
        raise ValueError(f"Experiment {config.exp_id} --kr must be a nonempty subset of {config.default_keep_ratios}")
    return ratios


def validate_experiment(exp: int, dataset: str, mode: str, kr: str | None = None):
    config = EXPERIMENT_CONFIGS[exp]
    if dataset not in config.allowed_datasets:
        raise ValueError(f"Experiment {exp} supports datasets {config.allowed_datasets}, not {dataset!r}")
    if mode not in config.allowed_modes:
        raise ValueError(f"Experiment {exp} does not support mode {mode!r}; allowed: {config.allowed_modes}")
    return config, parse_keep_ratios(kr, config)


def parse_args() -> argparse.Namespace:
    details = """Experiment 1: 50/50, CIFAR-100 or Tiny-ImageNet, full clean references and selection.
Experiment 2: 20/80 CIFAR-10, known-only calibration and unseen-only references/selection; kr is global.
Experiment 3: corrupt full CIFAR-100 first, 50/50 views, full corrupted references and center-repair."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=details,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, type=int, choices=(1, 2, 3), help="Protocol number; see details below.")
    parser.add_argument("--dataset", default=CIFAR100, choices=(CIFAR10, CIFAR100, TINY_IMAGENET))
    parser.add_argument("--mode", choices=tuple(MODE_METHODS), default="learned_group")
    parser.add_argument("--kr", help="Comma-separated subset of the protocol keep ratios.")
    parser.add_argument("--seed", type=int, default=int(CONFIG.global_seed))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--device")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--dist-weight-factor", type=float, default=1.0)
    parser.add_argument("--debug-prompts", action="store_true")
    args = parser.parse_args()
    try:
        args.config, args.keep_ratios = validate_experiment(args.exp, args.dataset, args.mode, args.kr)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _atomic_savez(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def build_raw_dataset(dataset_name: str, data_root: str | Path, transform=None) -> Dataset:
    root = Path(data_root)
    if not root.is_absolute(): root = PROJECT_ROOT / root
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(str(root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(str(root), train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train = root / "tiny-imagenet-200" / "train"
        if not train.is_dir(): raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train}")
        return datasets.ImageFolder(str(train), transform=transform)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def get_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"): return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "samples"): return np.asarray([row[1] for row in dataset.samples], dtype=np.int64)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))], dtype=np.int64)


class IndexedSubsetDataset(Dataset):
    """Local-index view; the base may already be a FixedCorruptionDataset."""
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.classes = getattr(base_dataset, "classes", None)
        self.targets = get_targets(base_dataset)[self.indices].tolist()
    def __len__(self): return len(self.indices)
    def __getitem__(self, index): return self.base_dataset[int(self.indices[index])]


KnownSubsetDataset = IndexedSubsetDataset
UnseenSubsetDataset = IndexedSubsetDataset


def load_unseen_split(dataset: str, unseen_ratio: int, seed: int, num_samples: int):
    path = PROJECT_ROOT / "unseen_data" / dataset / str(unseen_ratio) / f"unseen_list_{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"unseen list not found: {path}; run unseen_exp/generate_unseen_list.py first")
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not x.strip() or not x.strip().lstrip("+-").isdigit() for x in lines):
        raise ValueError(f"unseen list must contain exactly one integer per line: {path}")
    unseen = np.asarray([int(x) for x in lines], dtype=np.int64)
    expected = round(num_samples * unseen_ratio / 100)
    if unseen.ndim != 1 or len(unseen) != expected: raise ValueError(f"unseen count {len(unseen)}, expected {expected}")
    if np.unique(unseen).size != len(unseen): raise ValueError("unseen indices contain duplicates")
    if np.any(unseen < 0) or np.any(unseen >= num_samples): raise ValueError("unseen index out of range")
    known = np.setdiff1d(np.arange(num_samples, dtype=np.int64), unseen, assume_unique=False)
    if np.intersect1d(known, unseen).size or not np.array_equal(np.sort(np.r_[known, unseen]), np.arange(num_samples)):
        raise ValueError("known/unseen split is not a disjoint complete partition")
    return known, np.sort(unseen), path


def build_protocol_datasets(args, raw_dataset: Dataset):
    info = None
    full = raw_dataset
    if args.config.use_corruption:
        info = load_corruption_info(args.dataset, args.seed, num_samples=len(raw_dataset), strict_expected_size=True)
        full = FixedCorruptionDataset(raw_dataset, corruption_info=info)
    known, unseen, _ = load_unseen_split(args.dataset, args.config.unseen_ratio, args.seed, len(full))
    return full, IndexedSubsetDataset(full, known), IndexedSubsetDataset(full, unseen), known, unseen, info


def mask_path_for(exp: int, mode: str, dataset: str, seed: int, keep_ratio: int) -> Path:
    return SCRIPT_DIR / "mask" / str(exp) / MODE_METHODS[mode] / dataset / str(seed) / f"mask_{keep_ratio}.npz"


def cache_paths(exp: int, dataset: str, seed: int, proxy_model: str, epochs: int | None = None):
    base = SCRIPT_DIR
    epoch = str(epochs) if epochs is not None else "epochs"
    return {"adapter": base / "adapter" / str(exp) / dataset / str(seed),
            "proxy": base / "proxy_logs" / str(exp) / dataset / proxy_model / str(seed) / epoch,
            "dynamic": base / "dynamic_cache" / str(exp) / dataset / proxy_model / str(seed) / epoch,
            "calibration_static": base / "static_scores" / str(exp) / "calibration" / dataset / str(seed) / "static_scores.npz",
            "selection_static": base / "static_scores" / str(exp) / "selection" / dataset / str(seed) / "static_scores.npz",
            "weights": base / "weights" / str(exp) / "scoring_weights.json"}


def stable_random_seed(seed: int, exp_id: int, keep_ratio: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed), int(exp_id), int(keep_ratio), 0x554E5345])


def generate_random_mask(target_indices, full_num_samples, target_size, seed, exp_id, keep_ratio):
    candidates = np.asarray(target_indices, dtype=np.int64)
    if candidates.ndim != 1 or np.unique(candidates).size != len(candidates): raise ValueError("target indices must be unique")
    if np.any(candidates < 0) or np.any(candidates >= full_num_samples): raise ValueError("target index out of range")
    if not 0 <= target_size <= len(candidates): raise ValueError("target size exceeds candidate pool")
    selected = np.random.default_rng(stable_random_seed(seed, exp_id, keep_ratio)).choice(candidates, target_size, replace=False)
    mask = np.zeros(full_num_samples, dtype=np.uint8); mask[selected] = 1
    return mask


def mask_metadata(args, mask, known, unseen, info: CorruptionInfo | None, keep_ratio):
    selected = np.flatnonzero(mask).astype(np.int64)
    target_pool = unseen if args.config.selection_scope == "unseen" else np.arange(len(mask))
    values = dict(mask=mask.astype(np.uint8), selected_indices=selected, dataset=np.asarray(args.dataset),
        seed=np.asarray(args.seed), exp=np.asarray(args.exp), keep_ratio=np.asarray(keep_ratio),
        method=np.asarray(MODE_METHODS[args.mode]), mode=np.asarray(args.mode),
        selection_scope=np.asarray(args.config.selection_scope), known_ratio=np.asarray(args.config.known_ratio),
        unseen_ratio=np.asarray(args.config.unseen_ratio), known_indices=known, unseen_indices=unseen,
        known_selected=np.asarray(mask[known].sum()), unseen_selected=np.asarray(mask[unseen].sum()),
        target_pool_size=np.asarray(len(target_pool)), target_selected=np.asarray(mask[target_pool].sum()))
    if info is not None:
        total = int(info.is_corrupted.sum()); corrupt_selected = int(mask[info.is_corrupted].sum())
        values.update(corruption_types=info.corruption_types, is_corrupted=info.is_corrupted,
            num_corrupted_total=np.asarray(total), num_corrupted_selected=np.asarray(corrupt_selected),
            corruption_ratio_total=np.asarray(total / len(mask)),
            corruption_ratio_in_mask=np.asarray(corrupt_selected / max(1, int(mask.sum()))))
    return values


def mask_cache_valid(path, args, num_samples, known, unseen, info, keep_ratio):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = set(mask_metadata(args, np.zeros(num_samples, np.uint8), known, unseen, info, keep_ratio))
            if not required.issubset(data.files): return False
            mask = np.asarray(data["mask"]); target = round(num_samples * keep_ratio / 100)
            scalar_checks = (int(data["exp"]) == args.exp and str(data["dataset"]) == args.dataset
                and int(data["seed"]) == args.seed and str(data["mode"]) == args.mode
                and str(data["method"]) == MODE_METHODS[args.mode]
                and str(data["selection_scope"]) == args.config.selection_scope
                and float(data["known_ratio"]) == args.config.known_ratio
                and int(data["unseen_ratio"]) == args.config.unseen_ratio)
            valid = (scalar_checks and mask.shape == (num_samples,) and set(np.unique(mask)).issubset({0, 1})
                and int(mask.sum()) == target and np.array_equal(data["selected_indices"], np.flatnonzero(mask))
                and np.array_equal(data["known_indices"], known) and np.array_equal(data["unseen_indices"], unseen)
                and int(data["known_selected"]) == int(mask[known].sum())
                and int(data["unseen_selected"]) == int(mask[unseen].sum()))
            if args.exp == 2: valid = valid and not mask[known].any() and int(data["target_pool_size"]) == len(unseen)
            if info is not None:
                valid = valid and np.array_equal(data["corruption_types"], info.corruption_types) and np.array_equal(data["is_corrupted"], info.is_corrupted) and np.isclose(float(data["corruption_ratio_total"]), .2)
            return bool(valid)
    except Exception:
        return False


def save_mask(args, mask, known, unseen, info, keep_ratio):
    path = mask_path_for(args.exp, args.mode, args.dataset, args.seed, keep_ratio)
    _atomic_savez(path, **mask_metadata(args, mask, known, unseen, info, keep_ratio))
    return path


@contextlib.contextmanager
def patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(obj, name); setattr(obj, name, value)
    try: yield
    finally: setattr(obj, name, old)


def train_adapter_on_known(args, known_indices, unseen_indices, full_factory) -> Path:
    out = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model)["adapter"]
    meta_path = out / "meta.json"
    if args.skip_saved and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (meta.get("exp") == args.exp and meta.get("dataset") == args.dataset and meta.get("seed") == args.seed
                and meta.get("known_indices") == known_indices.tolist()
                and meta.get("unseen_indices") == unseen_indices.tolist()
                and meta.get("clip_model") == args.clip_model
                and meta.get("static_reference_scope") == args.config.static_reference_scope
                and meta.get("selection_scope") == args.config.selection_scope): return out
    def patched_build(_name, _root, transform): return IndexedSubsetDataset(full_factory(transform), known_indices)
    def patched_dir(_name, _seed): out.mkdir(parents=True, exist_ok=True); return out
    ns = SimpleNamespace(dataset=args.dataset, data_root=str(args.data_root), clip_model=args.clip_model,
        prompt_template="a photo of a {}", batch_size=None, num_workers=4, epochs=30, lr=1e-4,
        weight_decay=0., hidden_dim=256, temperature=.07, step_size=30, gamma=.1,
        device=args.device, seed=str(args.seed), debug_prompts=args.debug_prompts)
    with patch_attr(train_adapter_module, "_build_dataset", patched_build), patch_attr(train_adapter_module, "resolve_adapter_dir", patched_dir):
        train_adapter_module.train_for_seed(ns, args.seed, multi_seed=False)
    meta = json.loads(meta_path.read_text()); meta.update(exp=args.exp, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, known_indices=known_indices.tolist(),
        unseen_indices=unseen_indices.tolist(), clip_model=args.clip_model,
        static_reference_scope=args.config.static_reference_scope, selection_scope=args.config.selection_scope)
    meta_path.write_text(json.dumps(meta, indent=2))
    return out


def train_proxy_on_known(args, known_indices, full_factory):
    """Run the unchanged five-fold proxy implementation on local known indices."""
    proxy_args = SimpleNamespace(dataset=args.dataset, data_root=str(args.data_root), model=args.proxy_model,
        epochs=None, batch_size=None, num_workers=4, lr=None, momentum=None, weight_decay=None,
        lr_milestones=None, lr_gamma=None, device=args.device or "", k_folds=K_FOLDS, seed=str(args.seed))
    proxy_args = train_proxy_module.apply_dataset_defaults(proxy_args)
    epochs = int(proxy_args.epochs)
    out = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model, epochs)["proxy"]
    meta_path = out / "meta.json"
    if args.skip_saved and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (meta.get("exp") == args.exp and meta.get("known_indices") == known_indices.tolist()
                and meta.get("proxy_model") == args.proxy_model):
            return out, epochs
    original_loader = train_proxy_module.BaseDataLoader
    class KnownLoader:
        def __init__(self, dataset_name, data_path, batch_size, num_workers, val_split, seed):
            self.batch_size, self.num_workers = batch_size, num_workers
            self.inner = original_loader(dataset_name, data_path, batch_size, num_workers, val_split, seed)
        def load(self):
            train, val, test = self.inner.load(); self.num_classes = self.inner.num_classes
            known = IndexedSubsetDataset(train.dataset, known_indices)
            return DataLoader(known, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers), val, test
    def patched_dir(_dataset, seed=None, *, proxy_model="resnet18", epochs, root=None):
        result = cache_paths(args.exp, args.dataset, args.seed, proxy_model, int(epochs))["proxy"]
        result.mkdir(parents=True, exist_ok=True); return result
    with patch_attr(train_proxy_module, "BaseDataLoader", KnownLoader), patch_attr(train_proxy_module, "resolve_proxy_log_dir", patched_dir):
        train_proxy_module.run_for_seed(proxy_args, args.seed)
    meta = json.loads(meta_path.read_text()); meta.update(exp=args.exp, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, known_indices=known_indices.tolist(), proxy_model=args.proxy_model)
    meta_path.write_text(json.dumps(meta, indent=2))
    return out, epochs


def prepare_proxy_and_weights(args, known, static, full_factory):
    """Train/load proxy, compute current A/C/T, and fit current softplus-ratio model."""
    proxy_dir, epochs = train_proxy_on_known(args, known, full_factory)
    paths = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model, epochs)
    weight_path = paths["weights"]
    if args.skip_saved and weight_path.is_file():
        payload = json.loads(weight_path.read_text()); entry = payload.get(args.dataset, {}).get(str(args.seed), {})
        if (entry.get("exp") == args.exp and entry.get("known_indices") == known.tolist()
                and entry.get("proxy_model") == args.proxy_model):
            return {key: float(entry[key]) for key in ("sa", "div", "dds")}
    labels = np.asarray(static["labels"], dtype=np.int64)
    old_labels = dynamic_utils.load_dataset_labels
    dynamic_utils.load_dataset_labels = lambda *_: labels.copy()
    try:
        folds, labels_all = dynamic_utils.load_cv_fold_logs(proxy_dir, args.dataset, str(args.data_root))
    finally:
        dynamic_utils.load_dataset_labels = old_labels
    components = {name: calculator().compute(folds=folds, labels_all=labels_all)
                  for name, calculator in COMPONENTS.items()}
    for name, value in components.items():
        _atomic_savez(paths["dynamic"] / f"{name}.npz", labels=labels_all,
            raw_foldwise=value.raw_foldwise, fold_normalized=value.fold_normalized,
            aggregated=value.aggregated, final_normalized=value.final_normalized,
            exp=args.exp, dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio,
            unseen_ratio=args.config.unseen_ratio, known_indices=known, clip_model=args.clip_model,
            proxy_model=args.proxy_model, num_samples=len(known),
            static_reference_scope=args.config.static_reference_scope,
            selection_scope=args.config.selection_scope)
    dynamic_target, _ = build_dynamic_target(components)
    features = np.stack([standard_zscore_by_class(static[key], labels) for key in ("sa", "div", "dds")], axis=1)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    fit = fit_softplus_ratio_regression(features, dynamic_target, ratio_lambda=1e-3,
        learning_rate=2e-3, max_iter=10000, tol=1e-6, device=device)
    weights = np.asarray(fit["normalized_weights"], dtype=np.float64)
    entry = dict(sa=float(weights[0]), div=float(weights[1]), dds=float(weights[2]),
        exp=args.exp, dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, known_indices=known.tolist(), proxy_model=args.proxy_model,
        proxy_epochs=epochs, clip_model=args.clip_model, static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope)
    try: payload = json.loads(weight_path.read_text()) if weight_path.exists() else {}
    except (OSError, json.JSONDecodeError): payload = {}
    payload.setdefault(args.dataset, {})[str(args.seed)] = entry
    weight_path.parent.mkdir(parents=True, exist_ok=True); weight_path.write_text(json.dumps(payload, indent=2))
    return {key: entry[key] for key in ("sa", "div", "dds")}


def compute_static_scores(args, dataset, adapter_dir, cache_kind, known, unseen, info):
    path = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model)[f"{cache_kind}_static"]
    labels = get_targets(dataset)
    if args.skip_saved and path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if (np.array_equal(data["known_indices"], known) and np.array_equal(data["unseen_indices"], unseen)
                    and int(data["exp"]) == args.exp and str(data["dataset"]) == args.dataset
                    and int(data["seed"]) == args.seed and float(data["known_ratio"]) == args.config.known_ratio
                    and int(data["unseen_ratio"]) == args.config.unseen_ratio
                    and str(data["static_reference_scope"]) == args.config.static_reference_scope
                    and str(data["selection_scope"]) == args.config.selection_scope
                    and str(data["clip_model"]) == args.clip_model
                    and str(data["proxy_model"]) == args.proxy_model
                    and int(data["num_samples"]) == len(dataset) and np.array_equal(data["labels"], labels)
                    and (info is None or (np.array_equal(data["corruption_types"], info.corruption_types)
                                          and np.array_equal(data["is_corrupted"], info.is_corrupted)))):
                return {k: np.asarray(data[k]) for k in ("sa", "div", "dds", "labels")}
    device = torch.device(args.device) if args.device else CONFIG.global_device
    classes = resolve_class_names_for_prompts(args.dataset, args.data_root, dataset.classes)
    dds = DifficultyDirection(class_names=classes, clip_model=args.clip_model, device=device)
    div = Div(class_names=classes, clip_model=args.clip_model, device=device)
    sa = SemanticAlignment(class_names=classes, clip_model=args.clip_model, device=device,
                           dataset_name=args.dataset, data_root=str(args.data_root), debug_prompts=args.debug_prompts)
    image, text, _ = load_trained_adapters(args.dataset, args.clip_model, dds.extractor.embed_dim, args.seed,
        map_location=device, adapter_image_path=adapter_dir/"adapter_image.pt", adapter_text_path=adapter_dir/"adapter_context.pt")
    def loader(preprocess):
        # Corruption stays in the base and therefore precedes both transform and index view.
        base = dataset.base_dataset if isinstance(dataset, IndexedSubsetDataset) else dataset
        if isinstance(base, FixedCorruptionDataset): base.transform = preprocess
        elif hasattr(base, "transform"): base.transform = preprocess
        view = IndexedSubsetDataset(base, dataset.indices) if isinstance(dataset, IndexedSubsetDataset) else base
        return DataLoader(view, batch_size=128, shuffle=False, num_workers=4)
    sa_scores = sa.score_dataset(loader(sa.extractor.preprocess), adapter_image=image, adapter_text=text).scores.numpy()
    div_scores = div.score_dataset(loader(div.extractor.preprocess), adapter=image).scores.numpy()
    dds_scores = dds.score_dataset(loader(dds.extractor.preprocess), adapter=image).scores.numpy()
    _atomic_savez(path, sa=sa_scores, div=div_scores, dds=dds_scores, labels=labels, exp=args.exp,
        dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_indices=known, unseen_indices=unseen, clip_model=args.clip_model, proxy_model=args.proxy_model,
        num_samples=len(dataset), static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope,
        corruption_types=info.corruption_types if info else np.empty(0, np.int16),
        is_corrupted=info.is_corrupted if info else np.empty(0, bool))
    return {"sa": sa_scores, "div": div_scores, "dds": dds_scores, "labels": labels}


def run_group_solver(args, static, dataset, adapter_dir, weights, target_size):
    device = torch.device(args.device) if args.device else CONFIG.global_device
    classes = resolve_class_names_for_prompts(args.dataset, args.data_root, dataset.classes)
    div = Div(class_names=classes, clip_model=args.clip_model, device=device)
    image, _, _ = load_trained_adapters(args.dataset, args.clip_model, div.extractor.embed_dim, args.seed,
        map_location=device, adapter_image_path=adapter_dir/"adapter_image.pt", adapter_text_path=adapter_dir/"adapter_context.pt")
    base = dataset.base_dataset if isinstance(dataset, IndexedSubsetDataset) else dataset
    if isinstance(base, FixedCorruptionDataset): base.transform = div.extractor.preprocess
    elif hasattr(base, "transform"): base.transform = div.extractor.preprocess
    scored_dataset = IndexedSubsetDataset(base, dataset.indices) if isinstance(dataset, IndexedSubsetDataset) else base
    loader = DataLoader(scored_dataset, batch_size=128, shuffle=False, num_workers=4)
    pool_ratio = 100.0 * target_size / len(dataset)
    solver = (mask_solvers.select_group_mask_by_center_repair if args.config.group_solver == "center_repair"
              else mask_solvers.select_group_mask)
    kwargs = dict(sa_raw_scores=static["sa"], div_metric=div, div_loader=loader, image_adapter=image,
        labels=static["labels"], weights=weights, num_classes=len(classes), keep_ratio=pool_ratio,
        device=device, seed=args.seed, dds_static_scores=static["dds"],
        group_candidate_pool_size=args.group_candidate_pool_size, group_init_count=args.group_init_count)
    if solver is mask_solvers.select_group_mask: kwargs["dist_weight_factor"] = args.dist_weight_factor
    local, _, _ = solver(**kwargs)
    if int(local.sum()) != target_size: raise RuntimeError(f"group solver selected {local.sum()}, expected {target_size}")
    return local


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    raw = build_raw_dataset(args.dataset, args.data_root)
    full, known_ds, unseen_ds, known, unseen, info = build_protocol_datasets(args, raw)
    valid = {kr: args.skip_saved and mask_cache_valid(mask_path_for(args.exp, args.mode, args.dataset, args.seed, kr),
             args, len(full), known, unseen, info, kr) for kr in args.keep_ratios}
    if all(valid.values()): return
    pending = [kr for kr in args.keep_ratios if not valid[kr]]
    pool_indices = unseen if args.config.selection_scope == "unseen" else np.arange(len(full), dtype=np.int64)
    if args.mode == "random":
        for kr in pending:
            target = round(len(full) * kr / 100)
            save_mask(args, generate_random_mask(pool_indices, len(full), target, args.seed, args.exp, kr), known, unseen, info, kr)
        return
    # Corruption is applied to the full dataset before this factory creates either view.
    def full_factory(transform):
        base = build_raw_dataset(args.dataset, args.data_root, transform=None)
        if info is not None: return FixedCorruptionDataset(base, transform=transform, corruption_info=info)
        base.transform = transform; return base
    adapter = train_adapter_on_known(args, known, unseen, full_factory)
    selection_ds = unseen_ds if args.config.selection_scope == "unseen" else full
    static_selection = compute_static_scores(args, selection_ds, adapter, "selection", known, unseen, info)
    if args.mode == "naive_group":
        weights = {"sa": 1/3, "div": 1/3, "dds": 1/3}
    else:
        calibration_ds = known_ds if args.exp == 2 else selection_ds
        static_calibration = (compute_static_scores(args, calibration_ds, adapter, "calibration", known, unseen, info)
                              if args.exp == 2 else {k: v[known] for k, v in static_selection.items()})
        weights = prepare_proxy_and_weights(args, known, static_calibration, full_factory)
    for kr in pending:
        target = round(len(full) * kr / 100)
        if target > len(selection_ds): raise ValueError(f"global target {target} exceeds selection pool {len(selection_ds)}")
        local = run_group_solver(args, static_selection, selection_ds, adapter, weights, target)
        mask = local if args.config.selection_scope == "full" else np.zeros(len(full), np.uint8)
        if args.config.selection_scope == "unseen": mask[unseen] = local
        save_mask(args, mask, known, unseen, info, kr)


if __name__ == "__main__":
    main()
