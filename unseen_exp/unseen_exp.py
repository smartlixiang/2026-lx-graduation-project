#!/usr/bin/env python3
"""Prepare scorers and masks for the two explicit unseen-sample protocols.

Experiment 1 calibrates on 50% known data and selects from the clean full
CIFAR-100/Tiny-ImageNet training set (60/70/80/90%).  
Experiment 3 corrupts the full CIFAR-100 training set first, then makes the
50/50 views, and uses center-repair selection (30/40/50/60%).

This entry point never trains a final classifier; it ends after writing masks.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from corruption_exp import corruption_opt
from corruption_exp.cal_corruption_mask import (FixedCorruptionDataset, CorruptionInfo,
                                                  load_corruption_info)
from dataset.dataset_config import CIFAR100, TINY_IMAGENET
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
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
    3: UnseenExperimentConfig(3, (CIFAR100,), .5, 50, (30, 40, 50, 60),
                              "full", "full_corrupted", True, "center_repair",
                              ("learned_group", "naive_group", "random")),
}
MODE_METHODS = {"learned_group": "unseen_learned_group",
                "naive_group": "unseen_naive_group", "random": "unseen_random"}
COMPONENTS = {"A": AbsorptionGainScore, "C": ConfusionComplementarityScore,
              "T": TransferabilityScore}
K_FOLDS = 5
ALL_STAGES = frozenset(range(1, 7))
# Keep this synchronized with learn_scoring_weights.py --ratio-lambda.
RATIO_LAMBDA_DEFAULT = 1e-3
CORRUPTION_TYPE_NAMES = tuple(
    corruption_opt.CORRUPTION_ID_TO_NAME[type_id]
    for type_id in range(corruption_opt.NUM_CORRUPTION_TYPES)
)

# Proxy schedules used only by the unseen-sample protocols. Tiny-ImageNet
# keeps a longer high-LR phase (25 epochs) before the first decay because the
# 20-epoch decay caused an abrupt early saturation in the observed curves.
# The last value in
# ``phase_boundaries`` is max_epochs, not a MultiStepLR milestone: a decay at
# the final epoch would have no subsequent optimization step on which to act.
UNSEEN_PROXY_SCHEDULES = {
    CIFAR100: {
        "epochs": 100,
        "lr_milestones": (40, 75),
        "phase_boundaries": (40, 75, 100),
    },
    TINY_IMAGENET: {
        "epochs": 45,
        "lr_milestones": (15, 30),
        "phase_boundaries": (15, 30, 45),
    },
}


def parse_skip_saved_stages(text: str | None) -> frozenset[int]:
    """Parse the stages whose valid artifacts may be reused."""
    if text is None:
        return frozenset()
    parts = text.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("--skip-saved must be comma-separated stage numbers 1 through 6")
    try:
        stages = frozenset(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("--skip-saved must be comma-separated stage numbers 1 through 6") from exc
    if not stages.issubset(ALL_STAGES):
        raise ValueError("--skip-saved stages must be between 1 and 6")
    return stages


def stage_reuse_requested(args, stage: int, *, upstream_dirty: bool = False) -> bool:
    return not upstream_dirty and stage in args.skip_saved_stages


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
    if exp not in EXPERIMENT_CONFIGS:
        raise ValueError(f"unsupported experiment {exp}; choose 1 or 3")
    config = EXPERIMENT_CONFIGS[exp]
    if dataset not in config.allowed_datasets:
        raise ValueError(f"Experiment {exp} supports datasets {config.allowed_datasets}, not {dataset!r}")
    if mode not in config.allowed_modes:
        raise ValueError(f"Experiment {exp} does not support mode {mode!r}; allowed: {config.allowed_modes}")
    return config, parse_keep_ratios(kr, config)


def parse_args() -> argparse.Namespace:
    details = """Experiment 1: 50/50, CIFAR-100 or Tiny-ImageNet, full clean references and selection.
Experiment 3: corrupt full CIFAR-100 first, 50/50 views, full corrupted references and center-repair."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=details,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, type=int, choices=(1, 3), help="Protocol number; see details below.")
    parser.add_argument("--dataset", default=CIFAR100, choices=(CIFAR100, TINY_IMAGENET))
    parser.add_argument("--mode", choices=tuple(MODE_METHODS), default="learned_group")
    parser.add_argument("--kr", help="Comma-separated subset of the protocol keep ratios.")
    parser.add_argument("--seed", type=int, default=int(CONFIG.global_seed))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--device")
    # Stage 1: Train and cache the CLIP image and text Adapters on the known subset.
    # Stage 2: Compute and cache the SA, Div, and DDS static scores.
    # Stage 3: Train the proxy model with five-fold cross-validation and cache its logits.
    # Stage 4: Extract A, C, and T, synthesize dynamic pseudo-labels, and cache them.
    # Stage 5: Learn and save the weights combining the SA, Div, and DDS scores.
    # Stage 6: Generate and save the group-based or random data-selection masks.
    parser.add_argument("--skip-saved", nargs="?", const="1,2,3,4,5,6", metavar="STAGES",
                        help="Comma-separated stages whose valid caches may be reused (bare flag: all stages).")
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--dist-weight-factor", type=float, default=1.0)
    parser.add_argument("--debug-prompts", action="store_true")
    args = parser.parse_args()
    try:
        args.skip_saved_stages = parse_skip_saved_stages(args.skip_saved)
        del args.skip_saved
        args.config, args.keep_ratios = validate_experiment(args.exp, args.dataset, args.mode, args.kr)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _atomic_savez(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def corruption_context(info: CorruptionInfo | None) -> dict[str, np.ndarray]:
    if info is None:
        return {"corruption_indices": np.empty(0, dtype=np.int64),
                "corruption_type_ids": np.empty(0, dtype=np.int16)}
    indices = np.flatnonzero(info.is_corrupted).astype(np.int64)
    return {"corruption_indices": indices,
            "corruption_type_ids": info.corruption_types[indices].astype(np.int16)}


def _json_corruption_context(info: CorruptionInfo | None) -> dict[str, list[int]]:
    """Verbose corruption context retained for non-weight cache compatibility."""
    return {key: value.tolist() for key, value in corruption_context(info).items()}


def corruption_fingerprint_from_arrays(indices: np.ndarray, type_ids: np.ndarray) -> str:
    indices_array = np.ascontiguousarray(indices, dtype=np.int64)
    type_array = np.ascontiguousarray(type_ids, dtype=np.int16)
    if indices_array.ndim != 1 or type_array.ndim != 1:
        raise ValueError("corruption indices and type ids must be one-dimensional")
    if indices_array.shape[0] != type_array.shape[0]:
        raise ValueError("corruption indices and type ids must have equal length")
    digest = hashlib.sha256()
    digest.update(indices_array.tobytes())
    digest.update(b"\0CORRUPTION-TYPES\0")
    digest.update(type_array.tobytes())
    digest.update(np.asarray([indices_array.size], dtype=np.int64).tobytes())
    return digest.hexdigest()


def compact_corruption_context(info: CorruptionInfo | None) -> dict[str, Any]:
    context = corruption_context(info)
    indices = context["corruption_indices"]
    type_ids = context["corruption_type_ids"]
    return {
        "corruption_count": int(indices.size),
        "corruption_fingerprint": corruption_fingerprint_from_arrays(indices, type_ids),
    }


def _weight_corruption_matches(entry: dict[str, Any], info: CorruptionInfo | None) -> bool:
    """Validate compact weight metadata while accepting the previous list format."""
    expected_verbose = _json_corruption_context(info)
    has_legacy = "corruption_indices" in entry or "corruption_type_ids" in entry
    if has_legacy:
        return (
            entry.get("corruption_indices") == expected_verbose["corruption_indices"]
            and entry.get("corruption_type_ids") == expected_verbose["corruption_type_ids"]
        )

    expected_compact = compact_corruption_context(info)
    if info is None and "corruption_count" not in entry and "corruption_fingerprint" not in entry:
        # Preserve compatibility with clean historical weight entries that did
        # not record any corruption fields.
        return True
    return (
        int(entry.get("corruption_count", -1)) == expected_compact["corruption_count"]
        and entry.get("corruption_fingerprint") == expected_compact["corruption_fingerprint"]
    )


def _integer_vector(value: np.ndarray) -> bool:
    return value.ndim == 1 and np.issubdtype(value.dtype, np.integer)


def _corruption_json_matches(entry: dict[str, Any], info: CorruptionInfo | None) -> bool:
    expected = _json_corruption_context(info)
    return (entry.get("corruption_indices", []) == expected["corruption_indices"]
            and entry.get("corruption_type_ids", []) == expected["corruption_type_ids"])


def build_raw_dataset(dataset_name: str, data_root: str | Path, transform=None) -> Dataset:
    root = Path(data_root)
    if not root.is_absolute(): root = PROJECT_ROOT / root
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


def print_scoring_weights(weights: dict[str, float]) -> None:
    """Print the stage-5 static-score weights on one line."""
    print(
        "[Stage 5][Weights] "
        f"SA={float(weights['sa']):.5f}, "
        f"Div={float(weights['div']):.5f}, "
        f"DDS={float(weights['dds']):.5f}"
    )


def corruption_type_mask_statistics(
    mask: np.ndarray,
    info: CorruptionInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-type counts and two useful proportions for a selected mask.

    ``ratios_in_mask`` uses all selected samples as the denominator.  The
    additional ``ratios_among_corrupted`` uses only selected corrupted samples
    as the denominator, making it possible to inspect whether the five fixed
    corruption types are represented evenly among the corrupted selections.
    """
    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.shape != (info.num_samples,):
        raise ValueError(
            f"mask shape {mask_array.shape} does not match corruption manifest "
            f"size {(info.num_samples,)}"
        )
    if not set(np.unique(mask_array)).issubset({0, 1}):
        raise ValueError("mask must contain only 0/1 values")

    selected_types = np.asarray(info.corruption_types, dtype=np.int16)[
        mask_array.astype(bool)
    ]
    counts = np.asarray(
        [np.sum(selected_types == type_id) for type_id in range(len(CORRUPTION_TYPE_NAMES))],
        dtype=np.int64,
    )
    selected_total = int(mask_array.sum())
    corrupted_selected = int(counts.sum())
    ratios_in_mask = counts.astype(np.float64) / max(1, selected_total)
    ratios_among_corrupted = counts.astype(np.float64) / max(1, corrupted_selected)
    return counts, ratios_in_mask, ratios_among_corrupted


def print_corruption_type_statistics(
    path: Path,
    mask: np.ndarray,
    info: CorruptionInfo | None,
) -> None:
    """Print experiment-3 corruption counts and proportions in one line."""
    if info is None:
        return
    counts, ratios_in_mask, ratios_among_corrupted = corruption_type_mask_statistics(
        mask, info
    )
    parts = [
        (
            f"{name}={int(counts[index])} "
            f"(mask={100.0 * ratios_in_mask[index]:.2f}%, "
            f"corrupted={100.0 * ratios_among_corrupted[index]:.2f}%)"
        )
        for index, name in enumerate(CORRUPTION_TYPE_NAMES)
    ]
    print(f"[Stage 6][Corruption] {path} | " + ", ".join(parts))


def print_saved_corruption_type_statistics(
    path: Path,
    info: CorruptionInfo | None,
) -> None:
    """Load a reused mask and print the same corruption statistics."""
    if info is None:
        return
    with np.load(path, allow_pickle=False) as data:
        if "mask" not in data.files:
            raise KeyError(f"mask array missing from cached file: {path}")
        mask = np.asarray(data["mask"], dtype=np.uint8)
    print_corruption_type_statistics(path, mask, info)


def mask_metadata(args, mask, known, unseen, info: CorruptionInfo | None, keep_ratio):
    selected = np.flatnonzero(mask).astype(np.int64)
    target_pool = np.arange(len(mask))
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
            **corruption_context(info),
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
                and int(data["keep_ratio"]) == keep_ratio
                and str(data["method"]) == MODE_METHODS[args.mode]
                and str(data["selection_scope"]) == args.config.selection_scope
                and float(data["known_ratio"]) == args.config.known_ratio
                and int(data["unseen_ratio"]) == args.config.unseen_ratio)
            valid = (scalar_checks and mask.shape == (num_samples,) and set(np.unique(mask)).issubset({0, 1})
                and int(mask.sum()) == target and np.array_equal(data["selected_indices"], np.flatnonzero(mask))
                and np.array_equal(data["known_indices"], known) and np.array_equal(data["unseen_indices"], unseen)
                and int(data["known_selected"]) == int(mask[known].sum())
                and int(data["unseen_selected"]) == int(mask[unseen].sum())
                and int(data["target_pool_size"]) == num_samples
                and int(data["target_selected"]) == target)
            if info is not None:
                corrupt_selected = int(mask[info.is_corrupted].sum())
                valid = (valid and np.array_equal(data["corruption_types"], info.corruption_types)
                    and np.array_equal(data["is_corrupted"], info.is_corrupted)
                    and int(data["num_corrupted_total"]) == int(info.is_corrupted.sum())
                    and int(data["num_corrupted_selected"]) == corrupt_selected
                    and np.isclose(float(data["corruption_ratio_total"]), .2)
                    and np.isclose(float(data["corruption_ratio_in_mask"]), corrupt_selected / max(1, target)))
            return bool(valid)
    except Exception:
        return False


def save_mask(args, mask, known, unseen, info, keep_ratio):
    path = mask_path_for(args.exp, args.mode, args.dataset, args.seed, keep_ratio)
    _atomic_savez(path, **mask_metadata(args, mask, known, unseen, info, keep_ratio))
    if args.exp == 3 and info is not None:
        print_corruption_type_statistics(path, mask, info)
    return path


@contextlib.contextmanager
def patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(obj, name); setattr(obj, name, value)
    try: yield
    finally: setattr(obj, name, old)


def adapter_batch_size(dataset: str) -> int:
    """Resolve the effective batch size using train_adapter's defaults."""
    return int(train_adapter_module._default_batch_size(dataset))


def expected_adapter_metadata(args, known_indices, unseen_indices, info, *, batch_size) -> dict[str, Any]:
    """Return precisely the protocol inputs that determine adapter training."""
    return dict(exp=args.exp, dataset=args.dataset, seed=args.seed,
        known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_indices=np.asarray(known_indices).tolist(), unseen_indices=np.asarray(unseen_indices).tolist(),
        clip_model=args.clip_model, adapter_type="linear", training_objective="InfoNCE",
        prompt_template="a photo of a {}", epochs=30, batch_size=batch_size,
        learning_rate=1e-4, weight_decay=0.0, temperature=0.07, step_size=30, gamma=0.1,
        **_json_corruption_context(info))


def validate_adapter_cache(adapter_dir, args, known_indices, unseen_indices, info) -> tuple[bool, str]:
    """Accept every historical Adapter cache that contains both required weight files.

    Adapter metadata is intentionally not part of cache validity.  Older runs may
    have no ``meta.json`` or may use a different metadata schema, but the saved
    image and context Adapter weights remain loadable by the downstream scorer.
    """
    adapter_dir = Path(adapter_dir)
    image_path = adapter_dir / "adapter_image.pt"
    context_path = adapter_dir / "adapter_context.pt"

    if not image_path.is_file():
        return False, "missing adapter_image.pt"
    if not context_path.is_file():
        return False, "missing adapter_context.pt"
    return True, "valid"


def adapter_cache_valid(adapter_dir, args, known_indices, unseen_indices, info) -> bool:
    valid, _ = validate_adapter_cache(adapter_dir, args, known_indices, unseen_indices, info)
    return valid


def train_adapter_on_known(args, known_indices, unseen_indices, full_factory, info=None,
                           *, upstream_dirty=False) -> tuple[Path, bool]:
    out = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model)["adapter"]
    meta_path = out / "meta.json"
    valid, reason = validate_adapter_cache(out, args, known_indices, unseen_indices, info)
    reuse = stage_reuse_requested(args, 1, upstream_dirty=upstream_dirty)
    if reuse and valid:
        print(f"[Stage 1][Reuse] valid adapter cache: {out}")
        return out, False
    if reuse:
        print(f"[Stage 1][Recompute] requested reuse but adapter cache is invalid: {reason}")
    elif upstream_dirty:
        print("[Stage 1][Dependency] recomputing because an upstream artifact changed")
    else:
        print("[Stage 1][Force] recomputing adapter")
    meta_path.unlink(missing_ok=True)
    def patched_build(_name, _root, transform): return IndexedSubsetDataset(full_factory(transform), known_indices)
    def patched_dir(_name, _seed): out.mkdir(parents=True, exist_ok=True); return out
    ns = SimpleNamespace(dataset=args.dataset, data_root=str(args.data_root), clip_model=args.clip_model,
        prompt_template="a photo of a {}", batch_size=None, num_workers=4, epochs=30, lr=1e-4,
        weight_decay=0., hidden_dim=256, temperature=.07, step_size=30, gamma=.1,
        device=args.device, seed=str(args.seed), debug_prompts=args.debug_prompts)
    with patch_attr(train_adapter_module, "_build_dataset", patched_build), patch_attr(train_adapter_module, "resolve_adapter_dir", patched_dir):
        train_adapter_module.train_for_seed(ns, args.seed, multi_seed=False)
    if not ((out / "adapter_image.pt").is_file() and (out / "adapter_context.pt").is_file()):
        raise RuntimeError("adapter training did not produce both adapter files")
    # Metadata is best-effort bookkeeping only.  Cache validity intentionally
    # depends solely on the two Adapter weight files for backward compatibility.
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    except (OSError, json.JSONDecodeError, TypeError):
        meta = {}
    meta.update(expected_adapter_metadata(args, known_indices, unseen_indices, info,
                                          batch_size=adapter_batch_size(args.dataset)))
    _atomic_json(meta_path, meta)
    valid, reason = validate_adapter_cache(out, args, known_indices, unseen_indices, info)
    if not valid:
        raise RuntimeError(f"newly trained adapter cache failed validation: {reason}")
    return out, True


def unseen_proxy_schedule(dataset: str) -> dict[str, Any]:
    try:
        schedule = UNSEEN_PROXY_SCHEDULES[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported unseen proxy dataset: {dataset}") from exc
    return {
        "epochs": int(schedule["epochs"]),
        "lr_milestones": list(schedule["lr_milestones"]),
        "phase_boundaries": tuple(int(value) for value in schedule["phase_boundaries"]),
    }


def resolve_unseen_epoch_windows(dataset: str, num_epochs: int):
    """Return early/middle/late windows aligned with unseen-proxy LR phases."""
    schedule = unseen_proxy_schedule(dataset)
    expected_epochs = int(schedule["epochs"])
    if int(num_epochs) != expected_epochs:
        return dynamic_utils.resolve_epoch_windows(int(num_epochs))
    first, second, final = schedule["phase_boundaries"]
    if final != expected_epochs or not (0 < first < second < final):
        raise RuntimeError(f"invalid unseen proxy phase boundaries: {schedule}")
    return (
        np.arange(0, first, dtype=np.int64),
        np.arange(first, second, dtype=np.int64),
        np.arange(second, final, dtype=np.int64),
    )


@contextlib.contextmanager
def patch_dynamic_epoch_windows(dataset: str) -> Iterator[None]:
    """Temporarily align A/C/T early-middle-late windows with LR milestones."""
    original_default = dynamic_utils.resolve_epoch_windows

    def resolver(num_epochs: int):
        schedule = unseen_proxy_schedule(dataset)
        if int(num_epochs) == int(schedule["epochs"]):
            first, second, final = schedule["phase_boundaries"]
            return (
                np.arange(0, first, dtype=np.int64),
                np.arange(first, second, dtype=np.int64),
                np.arange(second, final, dtype=np.int64),
            )
        return original_default(int(num_epochs))

    modules = [dynamic_utils]
    for component_class in COMPONENTS.values():
        module = sys.modules.get(component_class.__module__)
        if module is not None and module not in modules:
            modules.append(module)

    previous = []
    for module in modules:
        if hasattr(module, "resolve_epoch_windows"):
            previous.append((module, getattr(module, "resolve_epoch_windows")))
            setattr(module, "resolve_epoch_windows", resolver)
    try:
        yield
    finally:
        for module, old_value in reversed(previous):
            setattr(module, "resolve_epoch_windows", old_value)


def make_proxy_args(args):
    schedule = unseen_proxy_schedule(args.dataset)
    proxy_args = SimpleNamespace(dataset=args.dataset, data_root=str(args.data_root), model=args.proxy_model,
        epochs=schedule["epochs"], batch_size=None, num_workers=4, lr=None, momentum=None, weight_decay=None,
        lr_milestones=schedule["lr_milestones"], lr_gamma=None,
        device=args.device or "", k_folds=K_FOLDS, seed=str(args.seed))
    proxy_args = train_proxy_module.apply_dataset_defaults(proxy_args)
    if int(proxy_args.epochs) != int(schedule["epochs"]):
        raise RuntimeError("unseen proxy max_epochs was unexpectedly overwritten")
    if list(proxy_args.lr_milestones) != list(schedule["lr_milestones"]):
        raise RuntimeError("unseen proxy LR milestones were unexpectedly overwritten")
    return proxy_args


def resolve_proxy_epochs(args) -> int:
    return int(make_proxy_args(args).epochs)


def dataset_transform(dataset):
    seen: set[int] = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        transform = getattr(current, "transform", None)
        if transform is not None: return transform
        current = next((getattr(current, name) for name in ("dataset", "base_dataset", "raw_dataset")
                        if hasattr(current, name)), None)
    return None


def build_proxy_known_dataset(original_train_dataset, full_factory, known_indices):
    protocol_full_dataset = full_factory(dataset_transform(original_train_dataset))
    return IndexedSubsetDataset(protocol_full_dataset, known_indices)


def proxy_cache_valid(proxy_dir, args, known_indices, unseen_indices, proxy_epochs, info) -> bool:
    try:
        meta = json.loads((proxy_dir / "meta.json").read_text(encoding="utf-8"))
        model = meta.get("proxy_model", meta.get("model"))
        if not (int(meta["exp"]) == args.exp and meta["dataset"] == args.dataset
                and int(meta["seed"]) == args.seed and int(meta["num_samples"]) == len(known_indices)
                and int(meta["epochs"]) == proxy_epochs and int(meta["k_folds"]) == K_FOLDS
                and float(meta["known_ratio"]) == args.config.known_ratio
                and int(meta["unseen_ratio"]) == args.config.unseen_ratio
                and meta["known_indices"] == known_indices.tolist() and meta["unseen_indices"] == unseen_indices.tolist()
                and model == args.proxy_model and meta["proxy_model"] == args.proxy_model
                and list(meta.get("lr_milestones", [])) == unseen_proxy_schedule(args.dataset)["lr_milestones"]
                and meta["static_reference_scope"] == args.config.static_reference_scope
                and meta["selection_scope"] == args.config.selection_scope and _corruption_json_matches(meta, info)):
            return False
        num_classes = int(meta["num_classes"]); validations = []
        expected_local = np.arange(len(known_indices), dtype=np.int64)
        for fold_no in range(1, K_FOLDS + 1):
            with np.load(proxy_dir / f"fold_{fold_no}.npz", allow_pickle=False) as fold:
                required = {"train_indices", "val_indices", "train_logits", "val_logits"}
                if not required.issubset(fold.files): return False
                train, val = np.asarray(fold["train_indices"]), np.asarray(fold["val_indices"])
                if not (_integer_vector(train) and _integer_vector(val)): return False
                if (np.any(train < 0) or np.any(train >= len(known_indices)) or np.any(val < 0)
                        or np.any(val >= len(known_indices)) or np.intersect1d(train, val).size
                        or not np.array_equal(np.sort(np.r_[train, val]), expected_local)): return False
                train_logits, val_logits = np.asarray(fold["train_logits"]), np.asarray(fold["val_logits"])
                if (train_logits.shape != (proxy_epochs, len(train), num_classes)
                        or val_logits.shape != (proxy_epochs, len(val), num_classes)
                        or not np.isfinite(train_logits).all() or not np.isfinite(val_logits).all()): return False
                validations.append(val.astype(np.int64))
        return np.array_equal(np.sort(np.concatenate(validations)), expected_local)
    except Exception:
        return False


def train_proxy_on_known(args, known_indices, unseen_indices, full_factory, info=None,
                         *, upstream_dirty=False):
    """Run five-fold proxy training with the unseen-protocol schedule."""
    proxy_args = make_proxy_args(args)
    epochs = int(proxy_args.epochs)
    out = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model, epochs)["proxy"]
    meta_path = out / "meta.json"
    reuse = stage_reuse_requested(args, 3, upstream_dirty=upstream_dirty)
    if reuse and proxy_cache_valid(out, args, known_indices, unseen_indices, epochs, info):
        print(f"[Stage 3][Reuse] valid proxy cache: {out}")
        return out, epochs, False
    if reuse:
        print("[Stage 3][Recompute] requested reuse but proxy cache is invalid")
    elif upstream_dirty:
        print("[Stage 3][Dependency] retraining proxy because an upstream artifact changed")
    else:
        print("[Stage 3][Force] retraining proxy")
    meta_path.unlink(missing_ok=True)
    for fold_no in range(1, K_FOLDS + 1): (out / f"fold_{fold_no}.npz").unlink(missing_ok=True)
    original_loader = train_proxy_module.BaseDataLoader
    class KnownLoader:
        def __init__(self, dataset_name, data_path, batch_size, num_workers, val_split, seed):
            self.batch_size, self.num_workers = batch_size, num_workers
            self.inner = original_loader(dataset_name, data_path=data_path, batch_size=batch_size,
                num_workers=num_workers, val_split=val_split, seed=seed)
        def load(self):
            train, val, test = self.inner.load(); self.num_classes = self.inner.num_classes
            known = build_proxy_known_dataset(train.dataset, full_factory, known_indices)
            return DataLoader(known, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers), val, test
    def patched_dir(_dataset, seed=None, *, proxy_model="resnet18", epochs, root=None):
        result = cache_paths(args.exp, args.dataset, args.seed, proxy_model, int(epochs))["proxy"]
        result.mkdir(parents=True, exist_ok=True); return result
    with patch_attr(train_proxy_module, "BaseDataLoader", KnownLoader), patch_attr(train_proxy_module, "resolve_proxy_log_dir", patched_dir):
        train_proxy_module.run_for_seed(proxy_args, args.seed)
    meta = json.loads(meta_path.read_text()); meta.update(exp=args.exp, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, known_indices=known_indices.tolist(), unseen_indices=unseen_indices.tolist(),
        proxy_model=args.proxy_model, static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope, **_json_corruption_context(info))
    _atomic_json(meta_path, meta)
    if not proxy_cache_valid(out, args, known_indices, unseen_indices, epochs, info):
        meta_path.unlink(missing_ok=True)
        raise RuntimeError("proxy training did not produce a complete valid five-fold cache")
    return out, epochs, True


def split_fingerprint(known, unseen, num_samples: int | None = None) -> str:
    known_array = np.ascontiguousarray(known, dtype=np.int64)
    unseen_array = np.ascontiguousarray(unseen, dtype=np.int64)
    total = len(known_array) + len(unseen_array) if num_samples is None else int(num_samples)
    digest = hashlib.sha256()
    digest.update(known_array.tobytes())
    digest.update(b"\0KNOWN/UNSEEN\0")
    digest.update(unseen_array.tobytes())
    digest.update(np.asarray([total], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _compact_weight_payload(payload: Any) -> Any:
    """Remove verbose split-index arrays from every scoring-weight entry.

    Legacy entries that still contain both index arrays are migrated first by
    deriving ``known_count``, ``unseen_count`` and ``split_fingerprint``.  The
    arrays are then removed recursively, so every JSON write is compact even
    when the file contains several datasets or seeds.
    """
    if isinstance(payload, dict):
        has_known = "known_indices" in payload
        has_unseen = "unseen_indices" in payload
        if has_known and has_unseen:
            try:
                known_array = np.asarray(payload["known_indices"], dtype=np.int64)
                unseen_array = np.asarray(payload["unseen_indices"], dtype=np.int64)
                if known_array.ndim == 1 and unseen_array.ndim == 1:
                    payload["known_count"] = int(known_array.size)
                    payload["unseen_count"] = int(unseen_array.size)
                    payload["split_fingerprint"] = split_fingerprint(
                        known_array, unseen_array
                    )
            except (TypeError, ValueError, OverflowError):
                # Malformed legacy entries remain invalid, but verbose arrays
                # are still removed from the JSON written by this script.
                pass
        if "corruption_indices" in payload and "corruption_type_ids" in payload:
            try:
                corruption_indices = np.asarray(payload["corruption_indices"], dtype=np.int64)
                corruption_type_ids = np.asarray(payload["corruption_type_ids"], dtype=np.int16)
                if (
                    corruption_indices.ndim == 1
                    and corruption_type_ids.ndim == 1
                    and corruption_indices.shape == corruption_type_ids.shape
                ):
                    payload["corruption_count"] = int(corruption_indices.size)
                    payload["corruption_fingerprint"] = corruption_fingerprint_from_arrays(
                        corruption_indices, corruption_type_ids
                    )
            except (TypeError, ValueError, OverflowError):
                pass
        payload.pop("known_indices", None)
        payload.pop("unseen_indices", None)
        payload.pop("corruption_indices", None)
        payload.pop("corruption_type_ids", None)
        for value in payload.values():
            _compact_weight_payload(value)
    elif isinstance(payload, list):
        for value in payload:
            _compact_weight_payload(value)
    return payload


def write_weight_payload(path: Path, payload: dict[str, Any]) -> None:
    _atomic_json(path, _compact_weight_payload(payload))


def load_scoring_weights(path, args, known, unseen, proxy_epochs, info):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = payload[args.dataset][str(args.seed)]
        required = {"sa", "div", "dds", "bias", "ratio_lambda", "exp", "dataset", "seed",
            "known_ratio", "unseen_ratio", "proxy_model", "proxy_epochs",
            "clip_model", "static_reference_scope", "selection_scope"}
        if not required.issubset(entry): return None
        values = np.asarray([entry[key] for key in ("sa", "div", "dds", "bias")], dtype=np.float64)
        weights = values[:3]
        has_legacy_arrays = "known_indices" in entry or "unseen_indices" in entry
        compact_split_valid = (
            int(entry.get("known_count", -1)) == len(known)
            and int(entry.get("unseen_count", -1)) == len(unseen)
            and entry.get("split_fingerprint") == split_fingerprint(known, unseen)
        )
        legacy_split_valid = (
            "known_indices" in entry
            and "unseen_indices" in entry
            and entry["known_indices"] == known.tolist()
            and entry["unseen_indices"] == unseen.tolist()
        )
        split_valid = compact_split_valid or legacy_split_valid
        valid = (np.isfinite(values).all() and np.all(weights > 0)
            and np.isclose(weights.sum(), 1.0, atol=1e-4, rtol=0.0)
            and float(entry["ratio_lambda"]) == RATIO_LAMBDA_DEFAULT
            and int(entry["exp"]) == args.exp
            and entry["dataset"] == args.dataset and int(entry["seed"]) == args.seed
            and float(entry["known_ratio"]) == args.config.known_ratio
            and int(entry["unseen_ratio"]) == args.config.unseen_ratio
            and split_valid
            and entry["proxy_model"] == args.proxy_model and int(entry["proxy_epochs"]) == proxy_epochs
            and entry["clip_model"] == args.clip_model
            and entry["static_reference_scope"] == args.config.static_reference_scope
            and entry["selection_scope"] == args.config.selection_scope
            and _weight_corruption_matches(entry, info))
        if not valid:
            return None
        if has_legacy_arrays:
            entry.update(known_count=len(known), unseen_count=len(unseen),
                         split_fingerprint=split_fingerprint(known, unseen))
            write_weight_payload(path, payload)
        return {key: float(entry[key]) for key in ("sa", "div", "dds")}
    except Exception:
        return None


def load_dynamic_component_cache(path, component_name, args, known, unseen, known_labels,
                                 proxy_epochs, info):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"labels", "raw_foldwise", "fold_normalized", "aggregated", "final_normalized",
                "component", "exp", "dataset", "seed", "known_ratio", "unseen_ratio", "known_indices",
                "unseen_indices", "proxy_model", "proxy_epochs", "clip_model", "static_reference_scope",
                "selection_scope", "corruption_types", "is_corrupted"}
            if not required.issubset(data.files): return None
            raw, normalized = np.asarray(data["raw_foldwise"]), np.asarray(data["fold_normalized"])
            aggregated, final = np.asarray(data["aggregated"]), np.asarray(data["final_normalized"])
            n = len(known_labels)
            valid = (str(data["component"]) == component_name and int(data["exp"]) == args.exp
                and str(data["dataset"]) == args.dataset and int(data["seed"]) == args.seed
                and float(data["known_ratio"]) == args.config.known_ratio
                and int(data["unseen_ratio"]) == args.config.unseen_ratio
                and np.array_equal(data["known_indices"], known) and np.array_equal(data["unseen_indices"], unseen)
                and str(data["proxy_model"]) == args.proxy_model and int(data["proxy_epochs"]) == proxy_epochs
                and str(data["clip_model"]) == args.clip_model
                and str(data["static_reference_scope"]) == args.config.static_reference_scope
                and str(data["selection_scope"]) == args.config.selection_scope
                and np.array_equal(data["labels"], known_labels) and aggregated.shape == (n,) and final.shape == (n,)
                and raw.ndim >= 2 and normalized.ndim >= 2 and raw.shape[:2] == (K_FOLDS, n)
                and normalized.shape[:2] == (K_FOLDS, n) and np.isfinite(aggregated).all()
                and np.isfinite(final).all() and not np.isinf(raw).any() and not np.isinf(normalized).any())
            expected_types = info.corruption_types if info else np.empty(0, np.int16)
            expected_corrupted = info.is_corrupted if info else np.empty(0, bool)
            valid = valid and np.array_equal(data["corruption_types"], expected_types) and np.array_equal(data["is_corrupted"], expected_corrupted)
            return SimpleNamespace(raw_foldwise=raw, fold_normalized=normalized,
                aggregated=aggregated, final_normalized=final) if valid else None
    except Exception:
        return None


def _dynamic_metadata(args, known, unseen, labels, proxy_epochs, info):
    return dict(labels=labels, exp=args.exp, dataset=args.dataset, seed=args.seed,
        known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_indices=known, unseen_indices=unseen, proxy_model=args.proxy_model,
        proxy_epochs=proxy_epochs, clip_model=args.clip_model,
        static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope,
        corruption_types=info.corruption_types if info else np.empty(0, np.int16),
        is_corrupted=info.is_corrupted if info else np.empty(0, bool))


def load_pseudo_labels_cache(path, args, known, unseen, labels, proxy_epochs, info):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = set(_dynamic_metadata(args, known, unseen, labels, proxy_epochs, info)) | {"dynamic_target"}
            if not required.issubset(data.files): return None
            target = np.asarray(data["dynamic_target"])
            expected = _dynamic_metadata(args, known, unseen, labels, proxy_epochs, info)
            valid = target.shape == (len(labels),) and np.isfinite(target).all()
            for key, value in expected.items():
                actual = data[key]
                valid = valid and (np.array_equal(actual, value) if isinstance(value, np.ndarray)
                                   else actual.item() == value)
            return target if valid else None
    except Exception:
        return None


def run_dynamic_stage(args, proxy_dir, proxy_epochs, known, unseen, labels, info,
                      *, upstream_dirty=False):
    cache_dir = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model, proxy_epochs)["dynamic"]
    components = {}
    reuse = stage_reuse_requested(args, 4, upstream_dirty=upstream_dirty)
    if reuse:
        for name in COMPONENTS:
            cached = load_dynamic_component_cache(cache_dir / f"{name}.npz", name, args, known,
                                                   unseen, labels, proxy_epochs, info)
            if cached is not None: components[name] = cached
    missing = [name for name in COMPONENTS if name not in components]
    pseudo = (load_pseudo_labels_cache(cache_dir / "pseudo_labels.npz", args, known, unseen,
                                       labels, proxy_epochs, info) if reuse and not missing else None)
    if not missing and pseudo is not None:
        print(f"[Stage 4][Reuse] valid A,C,T and pseudo labels: {cache_dir}")
        return components, pseudo, False
    folds = labels_all = None
    if missing:
        old_labels = dynamic_utils.load_dataset_labels
        dynamic_utils.load_dataset_labels = lambda *_: labels.copy()
        try:
            folds, labels_all = dynamic_utils.load_cv_fold_logs(proxy_dir, args.dataset, str(args.data_root))
        finally:
            dynamic_utils.load_dataset_labels = old_labels
    with patch_dynamic_epoch_windows(args.dataset):
        for name in missing:
            value = COMPONENTS[name]().compute(folds=folds, labels_all=labels_all)
            components[name] = value
            component_payload = _dynamic_metadata(
                args, known, unseen, labels_all, proxy_epochs, info
            )
            component_payload.update(
                raw_foldwise=value.raw_foldwise,
                fold_normalized=value.fold_normalized,
                aggregated=value.aggregated,
                final_normalized=value.final_normalized,
                component=name,
            )
            _atomic_savez(cache_dir / f"{name}.npz", **component_payload)

    pseudo, _ = build_dynamic_target(components)
    pseudo_payload = _dynamic_metadata(
        args, known, unseen, labels, proxy_epochs, info
    )
    pseudo_payload["dynamic_target"] = pseudo
    _atomic_savez(cache_dir / "pseudo_labels.npz", **pseudo_payload)

    if reuse:
        reused_names = [name for name in COMPONENTS if name not in missing]
        recomputed_text = ",".join(missing) if missing else "pseudo labels only"
        print(
            f"[Stage 4][Partial reuse] reused {','.join(reused_names) or 'none'}; "
            f"recomputed {recomputed_text}"
        )
    elif upstream_dirty:
        print("[Stage 4][Dependency] recomputed A,C,T and pseudo labels")
    else:
        print("[Stage 4][Force] recomputed A,C,T and pseudo labels")
    return components, pseudo, True


def get_dynamic_components(args, proxy_dir, proxy_epochs, known, unseen, labels, info):
    """Compatibility wrapper returning the three mathematical components."""
    return run_dynamic_stage(args, proxy_dir, proxy_epochs, known, unseen, labels, info)[0]


def prepare_proxy_and_weights(args, known, unseen, static, full_factory, info=None,
                              *, proxy_upstream_dirty=False, weights_upstream_dirty=False):
    """Execute stages 3--5 independently and propagate explicit dirty state."""
    epochs = resolve_proxy_epochs(args)
    weight_path = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model, epochs)["weights"]
    proxy_dir, actual_epochs, proxy_recomputed = train_proxy_on_known(
        args, known, unseen, full_factory, info, upstream_dirty=proxy_upstream_dirty)
    if actual_epochs != epochs: raise RuntimeError("proxy epoch configuration changed during preparation")
    labels = np.asarray(static["labels"], dtype=np.int64)
    _, dynamic_target, dynamic_recomputed = run_dynamic_stage(
        args, proxy_dir, epochs, known, unseen, labels, info, upstream_dirty=proxy_recomputed)
    dirty = weights_upstream_dirty or proxy_recomputed or dynamic_recomputed
    if stage_reuse_requested(args, 5, upstream_dirty=dirty):
        cached = load_scoring_weights(weight_path, args, known, unseen, epochs, info)
        if cached is not None:
            print(f"[Stage 5][Reuse] valid scoring weights: {weight_path}")
            print_scoring_weights(cached)
            return cached, (proxy_recomputed, dynamic_recomputed, False)
    if dirty:
        print("[Stage 5][Dependency] refitting weights because an upstream stage changed")
    elif 5 in args.skip_saved_stages:
        print("[Stage 5][Recompute] requested reuse but scoring-weight cache is invalid")
    else:
        print("[Stage 5][Force] refitting scoring weights")
    features = np.stack([standard_zscore_by_class(static[key], labels) for key in ("sa", "div", "dds")], axis=1)
    device = torch.device(args.device) if args.device else CONFIG.global_device
    fit = fit_softplus_ratio_regression(
        features, dynamic_target, ratio_lambda=RATIO_LAMBDA_DEFAULT,
        learning_rate=2e-3, max_iter=10000, tol=1e-6, device=device)
    weights = np.asarray(fit["normalized_weights"], dtype=np.float64)
    if (weights.shape != (3,) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6)):
        raise RuntimeError(f"invalid fitted scoring weights: {weights!r}")
    entry = dict(sa=float(weights[0]), div=float(weights[1]), dds=float(weights[2]), bias=float(fit["bias"]),
        ratio_lambda=RATIO_LAMBDA_DEFAULT, exp=args.exp, dataset=args.dataset, seed=args.seed,
        known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_count=len(known), unseen_count=len(unseen), split_fingerprint=split_fingerprint(known, unseen),
        proxy_model=args.proxy_model,
        proxy_epochs=epochs, clip_model=args.clip_model, static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope, **compact_corruption_context(info))
    try: payload = json.loads(weight_path.read_text()) if weight_path.exists() else {}
    except (OSError, json.JSONDecodeError): payload = {}
    payload.setdefault(args.dataset, {})[str(args.seed)] = entry
    write_weight_payload(weight_path, payload)
    learned_weights = {key: entry[key] for key in ("sa", "div", "dds")}
    print_scoring_weights(learned_weights)
    return learned_weights, (proxy_recomputed, dynamic_recomputed, True)


def static_reference_scope(args, cache_kind):
    return "full_corrupted" if args.exp == 3 else "full"


def static_sample_indices(dataset):
    if isinstance(dataset, IndexedSubsetDataset): return dataset.indices.copy()
    return np.arange(len(dataset), dtype=np.int64)


def _score_array(result: Any) -> np.ndarray:
    """Convert a scorer result to a finite one-dimensional NumPy array."""
    scores = getattr(result, "scores", result)
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError(f"score output must be one-dimensional, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("score output contains NaN or infinity")
    return values


def load_static_cache(path, args, cache_kind, labels, sample_indices, known, unseen, info):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"sa", "div", "dds", "labels", "exp", "dataset", "seed", "known_ratio",
                "unseen_ratio", "known_indices", "unseen_indices", "clip_model",
                "num_samples", "static_reference_scope", "reference_scope", "selection_scope",
                "cache_kind", "sample_indices", "corruption_types", "is_corrupted"}
            if not required.issubset(data.files): return None
            scores = {key: np.asarray(data[key]) for key in ("sa", "div", "dds")}
            n = len(labels)
            valid = (all(value.shape == (n,) and np.isfinite(value).all() for value in scores.values())
                and np.asarray(data["labels"]).shape == (n,) and np.array_equal(data["labels"], labels)
                and int(data["exp"]) == args.exp and str(data["dataset"]) == args.dataset
                and int(data["seed"]) == args.seed and float(data["known_ratio"]) == args.config.known_ratio
                and int(data["unseen_ratio"]) == args.config.unseen_ratio
                and np.array_equal(data["known_indices"], known) and np.array_equal(data["unseen_indices"], unseen)
                and str(data["clip_model"]) == args.clip_model
                and int(data["num_samples"]) == n and str(data["static_reference_scope"]) == args.config.static_reference_scope
                and str(data["reference_scope"]) == static_reference_scope(args, cache_kind)
                and str(data["selection_scope"]) == args.config.selection_scope
                and str(data["cache_kind"]) == cache_kind and np.array_equal(data["sample_indices"], sample_indices))
            expected_types = info.corruption_types if info else np.empty(0, np.int16)
            expected_corrupted = info.is_corrupted if info else np.empty(0, bool)
            valid = valid and np.array_equal(data["corruption_types"], expected_types) and np.array_equal(data["is_corrupted"], expected_corrupted)
            return {**scores, "labels": labels.copy()} if valid else None
    except Exception:
        return None


def compute_static_scores(args, dataset, adapter_dir, cache_kind, known, unseen, info,
                          *, upstream_dirty=False, return_status=False):
    path = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model)[f"{cache_kind}_static"]
    labels = get_targets(dataset)
    sample_indices = static_sample_indices(dataset)
    reuse = stage_reuse_requested(args, 2, upstream_dirty=upstream_dirty)
    if reuse:
        cached = load_static_cache(path, args, cache_kind, labels, sample_indices, known, unseen, info)
        if cached is not None:
            print(f"[Stage 2][Reuse] valid static scores: {path}")
            return (cached, False) if return_status else cached
    if reuse:
        print("[Stage 2][Recompute] requested reuse but static-score cache is invalid")
    elif upstream_dirty:
        print("[Stage 2][Dependency] recomputing static scores")
    else:
        print("[Stage 2][Force] recomputing static scores")
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
    sa_scores = _score_array(
        sa.score_dataset(loader(sa.extractor.preprocess), adapter_image=image, adapter_text=text)
    )
    div_scores = _score_array(
        div.score_dataset(loader(div.extractor.preprocess), adapter=image)
    )
    dds_scores = _score_array(
        dds.score_dataset(loader(dds.extractor.preprocess), adapter=image)
    )
    expected_shape = (len(dataset),)
    for name, values in (("SA", sa_scores), ("Div", div_scores), ("DDS", dds_scores)):
        if values.shape != expected_shape:
            raise RuntimeError(
                f"{name} score count mismatch: got {values.shape}, expected {expected_shape}"
            )
    _atomic_savez(path, sa=sa_scores, div=div_scores, dds=dds_scores, labels=labels, exp=args.exp,
        dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_indices=known, unseen_indices=unseen, clip_model=args.clip_model,
        num_samples=len(dataset), static_reference_scope=args.config.static_reference_scope,
        reference_scope=static_reference_scope(args, cache_kind), selection_scope=args.config.selection_scope,
        cache_kind=cache_kind, sample_indices=sample_indices,
        corruption_types=info.corruption_types if info else np.empty(0, np.int16),
        is_corrupted=info.is_corrupted if info else np.empty(0, bool))
    result = {"sa": sa_scores, "div": div_scores, "dds": dds_scores, "labels": labels}
    return (result, True) if return_status else result


def invalidate_adapter_dependents(args) -> None:
    """Invalidate caches derived from a newly trained adapter, without involving mode."""
    paths = cache_paths(args.exp, args.dataset, args.seed, args.proxy_model)
    paths["selection_static"].unlink(missing_ok=True)
    weight_path = paths["weights"]
    if not weight_path.is_file():
        return
    try:
        values = json.loads(weight_path.read_text(encoding="utf-8"))
        dataset_values = values.get(args.dataset, {})
        dataset_values.pop(str(args.seed), None)
        if not dataset_values:
            values.pop(args.dataset, None)
        _atomic_json(weight_path, values)
    except Exception:
        weight_path.unlink(missing_ok=True)


def resolve_group_weights(args, known, unseen, static_selection, known_ds, selection_ds,
                          adapter, full_factory, info, *, upstream_dirty=False,
                          return_status=False):
    """Choose naive constants or run the learned-only calibration pipeline."""
    if args.mode == "naive_group":
        result = {"sa": 1/3, "div": 1/3, "dds": 1/3}
        return (result, False) if return_status else result
    static_calibration = {key: value[known] for key, value in static_selection.items()}
    weights, statuses = prepare_proxy_and_weights(
        args, known, unseen, static_calibration, full_factory, info,
        weights_upstream_dirty=upstream_dirty)
    return (weights, any(statuses)) if return_status else weights


def run_group_solver(args, static, dataset, adapter_dir, weights, target_size):
    if not 0 <= int(target_size) <= len(dataset):
        raise ValueError(
            f"target_size must be between 0 and {len(dataset)}, got {target_size}"
        )
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
    if solver is mask_solvers.select_group_mask:
        kwargs["weight_group"] = "learned"
        kwargs["dist_weight_factor"] = args.dist_weight_factor
    local, _, _ = solver(**kwargs)
    if int(local.sum()) != target_size: raise RuntimeError(f"group solver selected {local.sum()}, expected {target_size}")
    return local


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    print(f"Requested reusable stages: {sorted(args.skip_saved_stages)}")
    print(f"Forced stages: {sorted(ALL_STAGES - args.skip_saved_stages)}")
    raw = build_raw_dataset(args.dataset, args.data_root)
    full, known_ds, unseen_ds, known, unseen, info = build_protocol_datasets(args, raw)
    pool_indices = np.arange(len(full), dtype=np.int64)
    if args.mode == "random":
        for stage in range(1, 6): print(f"[Stage {stage}][Not applicable] mode=random")
        for kr in args.keep_ratios:
            path = mask_path_for(args.exp, args.mode, args.dataset, args.seed, kr)
            if stage_reuse_requested(args, 6) and mask_cache_valid(path, args, len(full), known, unseen, info, kr):
                print(f"[Stage 6][Reuse] valid mask, kr={kr}: {path}")
                if args.exp == 3 and info is not None:
                    print_saved_corruption_type_statistics(path, info)
                continue
            status = "Recompute" if 6 in args.skip_saved_stages else "Force"
            print(f"[Stage 6][{status}] regenerating mask, kr={kr}")
            target = round(len(full) * kr / 100)
            save_mask(args, generate_random_mask(pool_indices, len(full), target, args.seed, args.exp, kr), known, unseen, info, kr)
        return
    # Corruption is applied to the full dataset before this factory creates either view.
    def full_factory(transform):
        base = build_raw_dataset(args.dataset, args.data_root, transform=None)
        if info is not None: return FixedCorruptionDataset(base, transform=transform, corruption_info=info)
        base.transform = transform; return base
    adapter, adapter_recomputed = train_adapter_on_known(args, known, unseen, full_factory, info)
    static_selection, static_recomputed = compute_static_scores(
        args, full, adapter, "selection", known, unseen, info,
        upstream_dirty=adapter_recomputed, return_status=True)
    if args.mode == "naive_group":
        for stage in (3, 4, 5): print(f"[Stage {stage}][Not applicable] mode=naive_group")
        weights, weights_dirty = {"sa": 1/3, "div": 1/3, "dds": 1/3}, False
    else:
        weights, weights_dirty = resolve_group_weights(
            args, known, unseen, static_selection, known_ds, full, adapter, full_factory, info,
            upstream_dirty=static_recomputed, return_status=True)
    stage6_dirty = adapter_recomputed or static_recomputed or weights_dirty
    for kr in args.keep_ratios:
        path = mask_path_for(args.exp, args.mode, args.dataset, args.seed, kr)
        if stage_reuse_requested(args, 6, upstream_dirty=stage6_dirty) and mask_cache_valid(
                path, args, len(full), known, unseen, info, kr):
            print(f"[Stage 6][Reuse] valid mask, kr={kr}: {path}")
            if args.exp == 3 and info is not None:
                print_saved_corruption_type_statistics(path, info)
            continue
        print(f"[Stage 6][{'Dependency' if stage6_dirty else 'Force'}] regenerating mask, kr={kr}")
        target = round(len(full) * kr / 100)
        mask = run_group_solver(args, static_selection, full, adapter, weights, target)
        save_mask(args, mask, known, unseen, info, kr)


if __name__ == "__main__":
    main()