#!/usr/bin/env python
"""Generate learned_group type-1 masks with fixed human metric weights.

Weights in (SA, Div, SV) order:
    high_sa:  (0.8, 0.1, 0.1)
    high_div: (0.1, 0.8, 0.1)
    high_sv:  (0.1, 0.1, 0.8)
SV retains the historical ``dds`` key in the code and static caches.

Only existing adapters and validated raw static caches are used. No proxy
training, dynamic-target loading, weight fitting, or static-score recomputation
is performed. Dynamic Div and distribution correction follow the main solver.

Example:
    CUDA_VISIBLE_DEVICES=4 python ablation_human_weights.py --seed 22
Outputs:
    mask/<high_sa|high_div|high_sv>/cifar100/<seed>/mask_<kr>.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from calculate_my_mask import (
    _build_dataset,
    build_score_loader,
    parse_ratio_list,
    select_group_mask,
)
from dataset.dataset_config import CIFAR100
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.path_rules import resolve_mask_path
from utils.seed import parse_seed_list, set_seed
from utils.static_score_cache import (
    get_or_compute_static_scores,
    resolve_static_score_cache_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent
HUMAN_WEIGHTS = {
    "high_sa": {"sa": 0.8, "div": 0.1, "dds": 0.1},
    "high_div": {"sa": 0.1, "div": 0.8, "dds": 0.1},
    "high_sv": {"sa": 0.1, "div": 0.1, "dds": 0.8},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CIFAR100 learned-group ablation with fixed human weights."
    )
    parser.add_argument("--dataset", default=CIFAR100, choices=(CIFAR100,))
    parser.add_argument("--kr", default="30,50,70", help="Keep ratios in percent.")
    parser.add_argument("--seed", default="22,42,96", help="Comma-separated seeds.")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    # Use the same group defaults as the main and other ablation experiments.
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def load_cached_static_scores(
    *, dataset_name: str, seed: int, clip_model: str,
    adapter_paths: dict[str, Path], div_metric: Div,
    dds_metric: DifficultyDirection, sa_metric: SemanticAlignment,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Use the main cache metadata/hash checks and reject any cache miss."""
    cache_dir = resolve_static_score_cache_dir(
        PROJECT_ROOT / "static_scores", dataset_name, seed, div_metric.k,
        dds_metric.eigval_lower_bound, dds_metric.eigval_upper_bound,
    )
    for name in ("SA", "Div", "DDS"):
        path = cache_dir / f"{name}_cache.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Required static cache is missing: {path}")

    def reject_recomputation() -> dict[str, np.ndarray]:
        raise ValueError(
            f"Static caches failed the main experiment's metadata/hash/index "
            f"validation: {cache_dir}. This ablation requires matching local "
            "caches and will not recompute static scores."
        )

    scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=dataset_name,
        seed=seed,
        clip_model=clip_model,
        adapter_image_path=str(adapter_paths["image_path"]),
        adapter_text_path=str(adapter_paths["text_path"]),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=labels.size,
        compute_fn=reject_recomputation,
    )
    if not np.array_equal(labels, scores["labels"]):
        raise ValueError("Static cache labels do not match the current dataset.")
    for name in ("sa", "div", "dds"):
        scores[name] = np.asarray(scores[name], dtype=np.float32)
        if not np.all(np.isfinite(scores[name])):
            raise ValueError(f"Static {name} cache contains NaN/inf: {cache_dir}")
    return scores


def run(args: argparse.Namespace) -> None:
    keep_ratios = list(dict.fromkeys(parse_ratio_list(args.kr)))
    seeds = list(dict.fromkeys(parse_seed_list(args.seed)))
    if not keep_ratios or any(not 0 < ratio <= 100 for ratio in keep_ratios):
        raise ValueError("--kr must contain percentages in [1, 100].")
    if not seeds:
        raise ValueError("--seed must not be empty.")
    if args.group_init_count < 1 or args.group_candidate_pool_size < 1:
        raise ValueError("Group initialization count and candidate pool must be >= 1.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be >= 1 and worker count must be >= 0.")

    device = torch.device(args.device) if args.device else CONFIG.global_device
    dataset_name = args.dataset
    tasks = {
        seed: [
            (mode, ratio, resolve_mask_path(
                mode=mode, dataset=dataset_name, model=args.model_name,
                seed=seed, keep_ratio=ratio,
            ))
            for mode in HUMAN_WEIGHTS for ratio in keep_ratios
        ]
        for seed in seeds
    }
    if args.skip_saved:
        for seed in seeds:
            for _, _, path in tasks[seed]:
                if path.exists():
                    print(f"[skip] {path}")
            tasks[seed] = [task for task in tasks[seed] if not task[2].exists()]
    if not any(tasks.values()):
        print("All requested masks already exist.")
        return

    print(
        f"[Config] dataset={dataset_name} | device={device} | type=1 | "
        f"candidate_pool_size={args.group_candidate_pool_size} | "
        f"group_init_count={args.group_init_count} | cache_only=True"
    )
    dataset = _build_dataset(dataset_name, transform=None)
    class_names = list(resolve_class_names_for_prompts(
        dataset_name=dataset_name, data_root=PROJECT_ROOT / "data",
        class_names=dataset.classes,
    ))
    labels = np.asarray(dataset.targets, dtype=np.int64)
    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device,
    )
    div_metric = Div(
        class_names=class_names, clip_model=args.clip_model, device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device,
        dataset_name=dataset_name, data_root=str(PROJECT_ROOT / "data"),
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess, dataset_name, device,
        args.batch_size, args.num_workers,
    )

    for seed in seeds:
        if not tasks[seed]:
            continue
        set_seed(seed)
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name, clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim, seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()
        scores = load_cached_static_scores(
            dataset_name=dataset_name, seed=seed, clip_model=args.clip_model,
            adapter_paths=adapter_paths, div_metric=div_metric,
            dds_metric=dds_metric, sa_metric=sa_metric, labels=labels,
        )
        print(f"[seed={seed}] Loaded matching Adapter and static score caches.")
        for mode, keep_ratio, path in tasks[seed]:
            # Keep each variant's random state independent of execution order
            # and --skip-saved, and identical to a standalone main experiment.
            set_seed(seed)
            weights = HUMAN_WEIGHTS[mode].copy()
            print(f"[seed={seed}] mode={mode} | kr={keep_ratio} | weights={weights}")
            start = time.perf_counter()
            mask, _, stats = select_group_mask(
                scores["sa"], div_metric=div_metric, div_loader=div_loader,
                image_adapter=image_adapter, labels=labels, weights=weights,
                num_classes=len(class_names), keep_ratio=keep_ratio,
                device=device, seed=seed, weight_group="learned",
                div_static_scores=scores["div"], dds_static_scores=scores["dds"],
                group_candidate_pool_size=args.group_candidate_pool_size,
                group_init_count=args.group_init_count,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, mask=mask.astype(np.uint8))
            print(
                f"[saved] {path} | selected={int(mask.sum())} | "
                f"elapsed={time.perf_counter() - start:.2f}s | "
                f"subset_score={stats['subset_comprehensive_score']:.6f} | "
                f"distribution_shift={stats['distribution_shift']:.6f}"
            )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
