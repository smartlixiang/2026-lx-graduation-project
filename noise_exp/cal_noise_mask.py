#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-noise mask calculation for the proposed learned_group method.

Place this file at:
    2026-lx-graduation-project/noise_exp/cal_noise_mask.py

Run from the project root:
    CUDA_VISIBLE_DEVICES=0 python noise_exp/cal_noise_mask.py --dataset cifar10 --seed 22

This script is designed for the label-noise experiment:
    - data/ and noise/ are project-root siblings of noise_exp/.
    - noise file format: noise/{dataset}/noise_list_{seed}.txt
      with two integer columns: sample_id noisy_label.
    - datasets: cifar10, cifar100
    - keep ratios: 30, 50
    - proxy model: resnet18
    - method: learned_group

The pipeline is:
    1. Patch CIFAR training labels according to noise/{dataset}/noise_list_{seed}.txt.
    2. Train CLIP adapters on the noisy training labels.
    3. Train cross-validation proxy models on the noisy training labels.
    4. Learn static-score weights from proxy dynamics and noisy-label static scores.
    5. Compute SA/Div/DDS static scores on noisy labels.
    6. Run group-mode selection and save masks.

Main output masks:
    noise_exp/mask/noise_learned_group/{dataset}/{seed}/mask_{kr}.npz

Intermediate files follow the normal project naming style, but are rooted under
noise_exp/:
    noise_exp/adapters/
    noise_exp/weights/proxy_logs/
    noise_exp/weights/dynamic_cache/
    noise_exp/weights/scoring_weights.json
    noise_exp/static_scores/
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
NOISE_EXP_ROOT = THIS_FILE.parent
PROJECT_ROOT = NOISE_EXP_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
NOISE_ROOT = PROJECT_ROOT / "noise"

ADAPTER_ROOT = NOISE_EXP_ROOT / "adapters"
WEIGHTS_ROOT = NOISE_EXP_ROOT / "weights"
PROXY_LOG_ROOT = WEIGHTS_ROOT / "proxy_logs"
DYNAMIC_CACHE_ROOT = WEIGHTS_ROOT / "dynamic_cache"
STATIC_SCORE_ROOT = NOISE_EXP_ROOT / "static_scores"
MASK_ROOT = NOISE_EXP_ROOT / "mask"

# Project imports. They are intentionally imported after sys.path is set.
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100  # noqa: E402
from model.adapter import AdapterMLP, CLIPFeatureExtractor, load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import build_class_prompts, resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.training_defaults import get_default_training_config  # noqa: E402

import calculate_my_mask as calc_mask  # noqa: E402
import train_adapter as train_adapter_mod  # noqa: E402
import train_proxy as train_proxy_mod  # noqa: E402
import learn_scoring_weights as learn_weights_mod  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate learned_group masks for fixed label-noise experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=(CIFAR10, CIFAR100),
        help="Only cifar10 and cifar100 are used in this label-noise experiment.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="Seed list. Usually one of 22, 42, 96; comma-separated is also supported.",
    )
    parser.add_argument("--kr", type=str, default="30,50", help="Keep ratios. Default: 30,50.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--proxy-model", type=str, default="resnet18")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Only used in mask path rule.")
    parser.add_argument("--mode-name", type=str, default="noise_learned_group")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--skip-saved", action="store_true", help="Skip existing final masks.")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if intermediate files exist.")
    return parser.parse_args()


def parse_ratio_list(text: str) -> list[int]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("--kr cannot be empty.")
    ratios = [int(item) for item in items]
    for ratio in ratios:
        if ratio <= 0 or ratio > 100:
            raise ValueError(f"Invalid keep ratio: {ratio}")
    return ratios


# ---------------------------------------------------------------------------
# Noise utilities
# ---------------------------------------------------------------------------

def read_noise_list(dataset_name: str, seed: int) -> np.ndarray:
    path = NOISE_ROOT / dataset_name / f"noise_list_{seed}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Noise list not found: {path}")

    arr = np.loadtxt(path, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Noise list must have shape (N,2), got {arr.shape}: {path}")

    return arr


def apply_noise_to_dataset_targets(dataset, dataset_name: str, seed: int, *, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutate a torchvision CIFAR train dataset in-place and return clean/noisy/is_noisy arrays."""
    if not hasattr(dataset, "targets"):
        raise AttributeError("Expected a torchvision CIFAR dataset with a .targets attribute.")

    clean_targets = np.asarray(dataset.targets, dtype=np.int64).copy()
    noisy_targets = clean_targets.copy()

    noise_map = read_noise_list(dataset_name, seed)
    sample_ids = noise_map[:, 0].astype(np.int64)
    new_labels = noise_map[:, 1].astype(np.int64)

    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate sample ids in noise list for {dataset_name}, seed={seed}.")
    if np.any(sample_ids < 0) or np.any(sample_ids >= len(clean_targets)):
        raise ValueError(f"Noise sample id out of range for {dataset_name}, seed={seed}.")
    if np.any(new_labels == clean_targets[sample_ids]):
        bad = int(np.sum(new_labels == clean_targets[sample_ids]))
        raise ValueError(f"{bad} noisy labels equal clean labels; this is not allowed.")

    noisy_targets[sample_ids] = new_labels
    dataset.targets = noisy_targets.tolist()

    is_noisy = np.zeros(len(clean_targets), dtype=bool)
    is_noisy[sample_ids] = True

    if verbose:
        print(
            f"[noise] dataset={dataset_name} seed={seed} "
            f"num_noisy={len(sample_ids)}/{len(clean_targets)} "
            f"rate={len(sample_ids) / len(clean_targets):.4f}"
        )

    return clean_targets, noisy_targets, is_noisy


def load_noisy_reference_dataset(dataset_name: str, seed: int, transform=None, *, verbose: bool = False):
    if dataset_name == CIFAR10:
        dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    elif dataset_name == CIFAR100:
        dataset = datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset for noise experiment: {dataset_name}")
    clean, noisy, is_noisy = apply_noise_to_dataset_targets(dataset, dataset_name, seed, verbose=verbose)
    return dataset, clean, noisy, is_noisy


@contextlib.contextmanager
def patched_cifar_noise(dataset_name: str, seed: int, *, verbose_once: bool = False) -> Iterator[None]:
    """
    Patch torchvision.datasets.CIFAR10/CIFAR100 so every training split constructed
    inside the existing project code receives the fixed noisy labels.
    """
    orig_cifar10 = datasets.CIFAR10
    orig_cifar100 = datasets.CIFAR100
    printed = {"value": False}

    def _should_patch_train(args, kwargs) -> bool:
        if "train" in kwargs:
            return bool(kwargs["train"])
        if len(args) >= 2:
            return bool(args[1])
        return True

    def noisy_cifar10(*args, **kwargs):
        ds = orig_cifar10(*args, **kwargs)
        if dataset_name == CIFAR10 and _should_patch_train(args, kwargs):
            do_verbose = verbose_once and not printed["value"]
            apply_noise_to_dataset_targets(ds, CIFAR10, seed, verbose=do_verbose)
            printed["value"] = True
        return ds

    def noisy_cifar100(*args, **kwargs):
        ds = orig_cifar100(*args, **kwargs)
        if dataset_name == CIFAR100 and _should_patch_train(args, kwargs):
            do_verbose = verbose_once and not printed["value"]
            apply_noise_to_dataset_targets(ds, CIFAR100, seed, verbose=do_verbose)
            printed["value"] = True
        return ds

    datasets.CIFAR10 = noisy_cifar10  # type: ignore[assignment]
    datasets.CIFAR100 = noisy_cifar100  # type: ignore[assignment]
    try:
        yield
    finally:
        datasets.CIFAR10 = orig_cifar10  # type: ignore[assignment]
        datasets.CIFAR100 = orig_cifar100  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Path patching for existing project modules
# ---------------------------------------------------------------------------

def noise_adapter_dir(dataset_name: str, seed: int) -> Path:
    path = ADAPTER_ROOT / dataset_name / str(seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def noise_adapter_paths(dataset_name: str, seed: int) -> tuple[Path, Path]:
    base = noise_adapter_dir(dataset_name, seed)
    return base / "adapter_image.pt", base / "adapter_context.pt"


@contextlib.contextmanager
def patched_adapter_output() -> Iterator[None]:
    old_train_resolve = getattr(train_adapter_mod, "resolve_adapter_dir", None)
    import model.adapter as adapter_mod

    old_adapter_resolve = getattr(adapter_mod, "resolve_adapter_dir", None)

    def _resolve(dataset_name: str, seed: int) -> Path:
        return noise_adapter_dir(dataset_name, seed)

    if old_train_resolve is not None:
        train_adapter_mod.resolve_adapter_dir = _resolve  # type: ignore[attr-defined]
    if old_adapter_resolve is not None:
        adapter_mod.resolve_adapter_dir = _resolve  # type: ignore[attr-defined]

    try:
        yield
    finally:
        if old_train_resolve is not None:
            train_adapter_mod.resolve_adapter_dir = old_train_resolve  # type: ignore[attr-defined]
        if old_adapter_resolve is not None:
            adapter_mod.resolve_adapter_dir = old_adapter_resolve  # type: ignore[attr-defined]


@contextlib.contextmanager
def patched_proxy_output() -> Iterator[None]:
    old_resolve = train_proxy_mod.resolve_proxy_log_dir

    def _resolve_proxy_log_dir(dataset: str, seed: int | None = None, *, proxy_model: str = "resnet18", epochs: int, root=None) -> Path:
        del seed, root
        return PROXY_LOG_ROOT / dataset / proxy_model / str(int(epochs))

    train_proxy_mod.resolve_proxy_log_dir = _resolve_proxy_log_dir  # type: ignore[assignment]
    try:
        yield
    finally:
        train_proxy_mod.resolve_proxy_log_dir = old_resolve  # type: ignore[assignment]


@contextlib.contextmanager
def patched_weight_learning_paths() -> Iterator[None]:
    old_cache_dir = getattr(learn_weights_mod, "resolve_dynamic_component_cache_dir", None)
    old_cache_path = getattr(learn_weights_mod, "resolve_dynamic_component_cache_path", None)
    old_static_cache = getattr(learn_weights_mod, "get_or_compute_static_scores", None)

    def _cache_dir(dataset: str, proxy_model: str, epochs: int) -> Path:
        return DYNAMIC_CACHE_ROOT / dataset / proxy_model / str(int(epochs))

    def _cache_path(dataset: str, proxy_model: str, epochs: int, component_name: str) -> Path:
        return _cache_dir(dataset, proxy_model, epochs) / f"{component_name.strip().upper()}.npz"

    def _static_cache_wrapper(**kwargs):
        kwargs["cache_root"] = STATIC_SCORE_ROOT
        return get_or_compute_static_scores(**kwargs)

    if old_cache_dir is not None:
        learn_weights_mod.resolve_dynamic_component_cache_dir = _cache_dir  # type: ignore[attr-defined]
    if old_cache_path is not None:
        learn_weights_mod.resolve_dynamic_component_cache_path = _cache_path  # type: ignore[attr-defined]
    if old_static_cache is not None:
        learn_weights_mod.get_or_compute_static_scores = _static_cache_wrapper  # type: ignore[attr-defined]

    try:
        yield
    finally:
        if old_cache_dir is not None:
            learn_weights_mod.resolve_dynamic_component_cache_dir = old_cache_dir  # type: ignore[attr-defined]
        if old_cache_path is not None:
            learn_weights_mod.resolve_dynamic_component_cache_path = old_cache_path  # type: ignore[attr-defined]
        if old_static_cache is not None:
            learn_weights_mod.get_or_compute_static_scores = old_static_cache  # type: ignore[attr-defined]


@contextlib.contextmanager
def patched_group_mean_cache() -> Iterator[None]:
    old_func = calc_mask._mean_stats_cache_path

    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _mean_stats_cache_path(dataset_name: str, clip_model: str, adapter_image_path: str) -> Path:
        adapter_sha1 = _hash_file(Path(adapter_image_path))
        clip_tag = clip_model.replace("/", "-").replace(" ", "_")
        return (
            STATIC_SCORE_ROOT
            / "group_mean_stats"
            / dataset_name
            / clip_tag
            / f"img_adapter_{adapter_sha1}.npz"
        )

    calc_mask._mean_stats_cache_path = _mean_stats_cache_path  # type: ignore[assignment]
    try:
        yield
    finally:
        calc_mask._mean_stats_cache_path = old_func  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers for invoking existing scripts in-process
# ---------------------------------------------------------------------------

def call_module_main(module, argv: list[str]) -> None:
    """
    Invoke a project script in-process.

    Most project scripts expose main(), but learn_scoring_weights.py currently
    exposes parse_args() + run_once() instead of main(). This helper supports
    both conventions so cal_noise_mask.py can call all stages uniformly.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [getattr(module, "__file__", "module")] + argv

        if hasattr(module, "main"):
            module.main()
            return

        if hasattr(module, "parse_args") and hasattr(module, "run_once"):
            parsed_args = module.parse_args()
            if hasattr(module, "parse_seed_list"):
                output_seeds = module.parse_seed_list(parsed_args.seed)
            else:
                output_seeds = parse_seed_list(parsed_args.seed)
            if not output_seeds:
                raise ValueError(f"{module.__name__}: --seed must contain at least one seed.")
            module.run_once(parsed_args, output_seeds)
            return

        raise AttributeError(
            f"Module {module.__name__} has neither main() nor parse_args()+run_once()."
        )
    finally:
        sys.argv = old_argv


def proxy_epochs_for_dataset(dataset_name: str) -> int:
    return int(get_default_training_config(dataset_name)["epochs"])


def all_proxy_logs_exist(dataset_name: str, proxy_model: str, epochs: int, k_folds: int = 5) -> bool:
    log_dir = PROXY_LOG_ROOT / dataset_name / proxy_model / str(int(epochs))
    meta_path = log_dir / "meta.json"
    fold_paths = [log_dir / f"fold_{i}.npz" for i in range(k_folds)]
    return meta_path.is_file() and all(path.is_file() for path in fold_paths)


def weights_entry_exists(weights_path: Path, dataset_name: str, seed: int) -> bool:
    if not weights_path.is_file():
        return False
    try:
        data = json.loads(weights_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    entry = data.get(dataset_name)
    return isinstance(entry, dict) and isinstance(entry.get(str(seed)), dict)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_adapter_stage(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[Path, Path]:
    image_path, text_path = noise_adapter_paths(args.dataset, seed)
    if image_path.is_file() and text_path.is_file() and not args.force:
        print(f"[adapter] skip existing adapters: {image_path.parent}")
        return image_path, text_path

    print(f"[adapter] train noisy-label CLIP adapters | dataset={args.dataset} seed={seed}")
    with patched_cifar_noise(args.dataset, seed, verbose_once=True), patched_adapter_output():
        cli = [
            "--dataset", args.dataset,
            "--data-root", str(DATA_ROOT),
            "--clip-model", args.clip_model,
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        if args.debug_prompts:
            cli += ["--debug-prompts"]
        call_module_main(train_adapter_mod, cli)

    if not image_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"Adapter training finished but weights are missing: {image_path}, {text_path}")
    return image_path, text_path


def run_proxy_stage(args: argparse.Namespace, seed: int, device: torch.device) -> None:
    epochs = proxy_epochs_for_dataset(args.dataset)
    if all_proxy_logs_exist(args.dataset, args.proxy_model, epochs) and not args.force:
        print(f"[proxy] skip existing proxy logs: {PROXY_LOG_ROOT / args.dataset / args.proxy_model / str(epochs)}")
        return

    print(f"[proxy] train noisy-label CV proxy | dataset={args.dataset} seed={seed} model={args.proxy_model}")
    with patched_cifar_noise(args.dataset, seed, verbose_once=True), patched_proxy_output():
        cli = [
            "--dataset", args.dataset,
            "--data_root", str(DATA_ROOT),
            "--model", args.proxy_model,
            "--seed", str(seed),
            "--num_workers", str(args.num_workers),
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        call_module_main(train_proxy_mod, cli)


def run_weight_learning_stage(args: argparse.Namespace, seed: int, device: torch.device, image_path: Path, text_path: Path) -> None:
    weights_path = WEIGHTS_ROOT / "scoring_weights.json"
    epochs = proxy_epochs_for_dataset(args.dataset)

    if weights_entry_exists(weights_path, args.dataset, seed) and not args.force:
        print(f"[weights] skip existing learned weights: {weights_path} dataset={args.dataset} seed={seed}")
        return

    print(f"[weights] learn static-score weights | dataset={args.dataset} seed={seed}")
    with patched_cifar_noise(args.dataset, seed, verbose_once=True), patched_weight_learning_paths():
        cli = [
            "--dataset", args.dataset,
            "--data-root", str(DATA_ROOT),
            "--proxy-log", str(PROXY_LOG_ROOT),
            "--proxy-model", args.proxy_model,
            "--proxy-epochs", str(epochs),
            "--adapter-image-path", str(image_path),
            "--adapter-text-path", str(text_path),
            "--clip-model", args.clip_model,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--output", str(weights_path),
            "--seed", str(seed),
            "--proxy-training-seed", str(seed),
        ]
        if args.device is not None:
            cli += ["--device", str(device)]
        if args.debug_prompts:
            cli += ["--debug-prompts"]
        call_module_main(learn_weights_mod, cli)


def build_noisy_score_loader(preprocess, dataset_name: str, seed: int, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    dataset, _, _, _ = load_noisy_reference_dataset(dataset_name, seed, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def load_class_names_for_noisy_dataset(dataset_name: str) -> list[str]:
    if dataset_name == CIFAR10:
        dataset = datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=None)
    elif dataset_name == CIFAR100:
        dataset = datasets.CIFAR100(root=str(DATA_ROOT), train=True, download=True, transform=None)
    else:
        raise ValueError(dataset_name)
    return list(
        resolve_class_names_for_prompts(
            dataset_name=dataset_name,
            data_root=DATA_ROOT,
            class_names=dataset.classes,  # type: ignore[attr-defined]
        )
    )


def ensure_required_weights(weights_path: Path, dataset_name: str, seed: int) -> dict[str, float]:
    all_weights = calc_mask.ensure_scoring_weights(weights_path, dataset_name)
    return calc_mask.load_scoring_weights(all_weights, "learned", seed)


def run_mask_stage(args: argparse.Namespace, seed: int, device: torch.device, image_path: Path, text_path: Path) -> None:
    keep_ratios = parse_ratio_list(args.kr)

    # Skip early if all target masks exist.
    target_paths = [
        resolve_mask_path(
            mode=args.mode_name,
            dataset=args.dataset,
            model=args.model_name,
            seed=seed,
            keep_ratio=kr,
            root=MASK_ROOT,
        )
        for kr in keep_ratios
    ]
    if args.skip_saved and all(path.exists() for path in target_paths):
        print(f"[mask] skip all existing masks | dataset={args.dataset} seed={seed} kr={keep_ratios}")
        return

    print(f"[mask] compute noisy learned_group masks | dataset={args.dataset} seed={seed} kr={keep_ratios}")

    class_names = load_class_names_for_noisy_dataset(args.dataset)
    num_classes = len(class_names)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=args.dataset,
        data_root=str(DATA_ROOT),
        debug_prompts=args.debug_prompts,
    )

    dds_loader = build_noisy_score_loader(
        dds_metric.extractor.preprocess,
        args.dataset,
        seed,
        device,
        args.batch_size,
        args.num_workers,
    )
    div_loader = build_noisy_score_loader(
        div_metric.extractor.preprocess,
        args.dataset,
        seed,
        device,
        args.batch_size,
        args.num_workers,
    )
    sa_loader = build_noisy_score_loader(
        sa_metric.extractor.preprocess,
        args.dataset,
        seed,
        device,
        args.batch_size,
        args.num_workers,
    )
    dataset_for_labels, clean_targets, noisy_targets, is_noisy = load_noisy_reference_dataset(
        args.dataset, seed, transform=None, verbose=True
    )

    image_adapter, text_adapter, _ = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=seed,
        map_location=device,
        adapter_image_path=image_path,
        adapter_text_path=text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    weights_path = WEIGHTS_ROOT / "scoring_weights.json"
    weights = ensure_required_weights(weights_path, args.dataset, seed)
    print(f"[mask] weights seed={seed}: {weights}")

    num_samples = len(dataset_for_labels)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores

        return {
            "sa": np.asarray(sa_scores, dtype=np.float32),
            "div": np.asarray(div_scores, dtype=np.float32),
            "dds": np.asarray(dds_scores, dtype=np.float32),
            "labels": np.asarray(dataset_for_labels.targets, dtype=np.int64),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=STATIC_SCORE_ROOT,
        dataset=args.dataset,
        seed=seed,
        clip_model=args.clip_model,
        adapter_image_path=str(image_path),
        adapter_text_path=str(text_path),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)
    div_scores_np = np.asarray(static_scores["div"], dtype=np.float32)
    dds_scores_np = np.asarray(static_scores["dds"], dtype=np.float32)
    labels_np = np.asarray(static_scores["labels"], dtype=np.int64)

    if not np.array_equal(labels_np, noisy_targets.astype(np.int64)):
        raise RuntimeError("Static-score labels are not identical to noisy targets; abort to avoid invalid mask.")

    with patched_group_mean_cache():
        for kr, mask_path in zip(keep_ratios, target_paths):
            if args.skip_saved and mask_path.exists():
                print(f"[mask] skip existing: {mask_path}")
                continue

            mask, selected_by_class, group_stats = calc_mask.select_group_mask(
                sa_scores_np,
                div_metric=div_metric,
                div_loader=div_loader,
                image_adapter=image_adapter,
                labels=labels_np,
                weights=weights,
                num_classes=num_classes,
                keep_ratio=kr,
                device=device,
                dataset_name=args.dataset,
                seed=seed,
                weight_group="learned",
                clip_model=args.clip_model,
                adapter_image_path=str(image_path),
                div_static_scores=div_scores_np,
                dds_static_scores=dds_scores_np,
                group_candidate_pool_size=args.group_candidate_pool_size,
            )

            selected = mask.astype(bool)
            num_selected = int(mask.sum())
            num_noisy_selected = int(is_noisy[selected].sum())
            noise_ratio_in_mask = float(num_noisy_selected / max(1, num_selected))

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_path,
                mask=mask.astype(np.uint8),
                selected_indices=np.flatnonzero(mask).astype(np.int64),
                clean_targets=clean_targets.astype(np.int64),
                noisy_targets=noisy_targets.astype(np.int64),
                is_noisy=is_noisy.astype(np.uint8),
                weights=json.dumps(weights, ensure_ascii=False),
                selected_by_class=json.dumps(selected_by_class, ensure_ascii=False),
                group_stats=json.dumps(group_stats, ensure_ascii=False),
                dataset=np.array(args.dataset),
                method=np.array(args.mode_name),
                seed=np.array(seed, dtype=np.int64),
                keep_ratio=np.array(kr, dtype=np.int64),
                num_selected=np.array(num_selected, dtype=np.int64),
                num_noisy_total=np.array(int(is_noisy.sum()), dtype=np.int64),
                num_noisy_selected=np.array(num_noisy_selected, dtype=np.int64),
                noise_ratio_total=np.array(float(is_noisy.mean()), dtype=np.float32),
                noise_ratio_in_mask=np.array(noise_ratio_in_mask, dtype=np.float32),
            )

            summary_path = mask_path.with_suffix(".json")
            summary = {
                "dataset": args.dataset,
                "method": args.mode_name,
                "seed": int(seed),
                "keep_ratio": int(kr),
                "num_selected": num_selected,
                "num_noisy_total": int(is_noisy.sum()),
                "num_noisy_selected": num_noisy_selected,
                "noise_ratio_total": float(is_noisy.mean()),
                "noise_ratio_in_mask": noise_ratio_in_mask,
                "weights": weights,
                "mask_path": str(mask_path),
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            print(
                f"[mask] saved {mask_path} | selected={num_selected} "
                f"noisy_selected={num_noisy_selected} "
                f"noise_ratio={noise_ratio_in_mask:.4f}"
            )


def run_one_seed(args: argparse.Namespace, seed: int, device: torch.device) -> None:
    start = time.perf_counter()
    print("=" * 100)
    print(f"[start] dataset={args.dataset} seed={seed} device={device}")
    print(f"[paths] data={DATA_ROOT} noise={NOISE_ROOT} noise_exp={NOISE_EXP_ROOT}")

    set_seed(seed)
    image_path, text_path = run_adapter_stage(args, seed, device)
    run_proxy_stage(args, seed, device)
    run_weight_learning_stage(args, seed, device, image_path, text_path)
    run_mask_stage(args, seed, device, image_path, text_path)

    print(f"[done] dataset={args.dataset} seed={seed} elapsed={time.perf_counter() - start:.2f}s")
    print("=" * 100)


def validate_environment() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"data directory not found: {DATA_ROOT}")
    if not NOISE_ROOT.exists():
        raise FileNotFoundError(f"noise directory not found: {NOISE_ROOT}")
    NOISE_EXP_ROOT.mkdir(parents=True, exist_ok=True)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    PROXY_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DYNAMIC_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_environment()

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("--seed cannot be empty.")

    print(f"[config] dataset={args.dataset} seeds={seeds} kr={parse_ratio_list(args.kr)}")
    print(f"[config] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"[config] torch.cuda.is_available={torch.cuda.is_available()} device={device}")

    for seed in seeds:
        noise_path = NOISE_ROOT / args.dataset / f"noise_list_{seed}.txt"
        if not noise_path.is_file():
            raise FileNotFoundError(f"Noise file missing for seed={seed}: {noise_path}")
        run_one_seed(args, int(seed), device)


if __name__ == "__main__":
    main()
