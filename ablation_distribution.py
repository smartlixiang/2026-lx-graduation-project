"""Ablation of the distribution-correction term in learned_group type-1 selection.

This experiment keeps the full learned SA/Div/DDS weighting and the current
type-1 group selection pipeline, but disables only the distribution-correction
term by calling calculate_my_mask.select_group_mask(..., dist_weight_factor=0.0).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calculate_my_mask import (
    _build_dataset,
    build_score_loader,
    ensure_scoring_weights,
    load_scoring_weights,
    parse_ratio_list,
    select_group_mask,
)
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.path_rules import resolve_mask_path
from utils.seed import parse_seed_list, set_seed
from utils.static_score_cache import get_or_compute_static_scores


PROJECT_ROOT = Path(__file__).resolve().parent
MODE = "ablation_distribution"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learned-group type-1 ablation without distribution correction."
    )
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--kr", type=str, default="20,30,40,50,60,70,80,90")
    parser.add_argument("--seed", type=str, default=",".join(map(str, CONFIG.exp_seeds)))
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else CONFIG.global_device
    dataset_name = args.dataset.strip().lower()

    dataset = _build_dataset(dataset_name, transform=None)
    class_names = list(
        resolve_class_names_for_prompts(
            dataset_name=dataset_name,
            data_root=PROJECT_ROOT / "data",
            class_names=dataset.classes,
        )
    )

    dds_metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=str(PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )

    keep_ratios = parse_ratio_list(args.kr)
    if not keep_ratios:
        raise ValueError("kr 参数不能为空。")
    seeds = parse_seed_list(args.seed)
    if not seeds:
        raise ValueError("seed 参数不能为空。")

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)

    labels = np.asarray(dataset.targets, dtype=np.int64)
    total_tasks = len(seeds) * len(keep_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)

        # Only the group distribution-correction term is ablated.
        # Keep the full learned SA/Div/DDS weights unchanged.
        weights = load_scoring_weights(
            all_weights,
            weight_group="learned",
            seed=seed,
        )

        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()

        def compute_scores() -> dict[str, np.ndarray]:
            return {
                "dds": np.asarray(
                    dds_metric.score_dataset(
                        tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
                        adapter=image_adapter,
                    ).scores
                ),
                "div": np.asarray(
                    div_metric.score_dataset(
                        tqdm(div_loader, desc="Scoring Div", unit="batch"),
                        adapter=image_adapter,
                    ).scores
                ),
                "sa": np.asarray(
                    sa_metric.score_dataset(
                        tqdm(sa_loader, desc="Scoring SA", unit="batch"),
                        adapter_image=image_adapter,
                        adapter_text=text_adapter,
                    ).scores
                ),
                "labels": labels,
            }

        static_start = time.perf_counter()
        static_scores = get_or_compute_static_scores(
            cache_root=PROJECT_ROOT / "static_scores",
            dataset=dataset_name,
            seed=seed,
            clip_model=args.clip_model,
            adapter_image_path=str(adapter_paths["image_path"]),
            adapter_text_path=str(adapter_paths["text_path"]),
            div_k=div_metric.k,
            dds_k=dds_metric.k,
            dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
            dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
            prompt_template=sa_metric.prompt_template,
            num_samples=len(dataset),
            compute_fn=compute_scores,
        )
        static_seconds = time.perf_counter() - static_start

        if not np.array_equal(labels, np.asarray(static_scores["labels"], dtype=np.int64)):
            raise ValueError("Static cache labels do not match the current dataset.")

        sa_raw = np.asarray(static_scores["sa"], dtype=np.float32)
        div_raw = np.asarray(static_scores["div"], dtype=np.float32)
        dds_raw = np.asarray(static_scores["dds"], dtype=np.float32)

        print(
            f"[seed={seed}] learned weights: "
            f"SA={weights['sa']:.6f}, "
            f"Div={weights['div']:.6f}, "
            f"DDS={weights['dds']:.6f}"
        )

        for keep_ratio in keep_ratios:
            task_idx += 1
            mask_path = resolve_mask_path(
                mode=MODE,
                dataset=dataset_name,
                model=args.model_name,
                seed=seed,
                keep_ratio=keep_ratio,
            )

            if args.skip_saved and mask_path.exists():
                print(f"[skip] {mask_path}")
                continue

            print(
                f"[Mask {task_idx}/{total_tasks}] "
                f"seed={seed} | kr={keep_ratio} | "
                f"mode={MODE} | type=1 | dist_weight_factor=0"
            )

            start = time.perf_counter()

            mask, selected_by_class, stats = select_group_mask(
                sa_raw,
                div_metric=div_metric,
                div_loader=div_loader,
                image_adapter=image_adapter,
                labels=labels,
                weights=weights,
                num_classes=len(class_names),
                keep_ratio=keep_ratio,
                device=device,
                seed=seed,
                weight_group="learned",
                div_static_scores=div_raw,
                dds_static_scores=dds_raw,
                group_candidate_pool_size=args.group_candidate_pool_size,
                group_init_count=args.group_init_count,
                dist_weight_factor=0.0,
            )

            if not np.isclose(float(stats["dist_weight_max"]), 0.0):
                raise RuntimeError(
                    "Distribution correction was not fully disabled: "
                    f"dist_weight_max={stats['dist_weight_max']}"
                )
            if not np.isclose(float(stats["dist_weight_min"]), 0.0):
                raise RuntimeError(
                    "Distribution correction was not fully disabled: "
                    f"dist_weight_min={stats['dist_weight_min']}"
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            print(
                f"[saved] {mask_path} "
                f"selected={int(mask.sum())} "
                f"elapsed={time.perf_counter() - start:.2f}s "
                f"static_score_seconds={static_seconds:.2f} "
                f"subset_score={stats['subset_comprehensive_score']:.6f} "
                f"distribution_shift={stats['distribution_shift']:.6f}"
            )


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()