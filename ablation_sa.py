from __future__ import annotations

import argparse
import json
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calculate_my_mask import (  # noqa: E402
    _build_dataset,
    build_score_loader,
    parse_ratio_list,
    select_group_mask,
    select_topk_mask,
)
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR100  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.class_name_utils import resolve_class_names_for_prompts  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


ABLATION_NAME = "ablation_sa"
ABLATION_COMPONENT = "sa"
ACTIVE_COMPONENTS = ("dds", "div")
ALL_COMPONENTS = ("dds", "div", "sa")


class RatioBandDifficultyDirection(DifficultyDirection):
    """DDS variant using eigenvalue-ratio band [lower, upper].

    The repository's current DifficultyDirection class keeps the historical
    2%-20% cache parameters, but its default direction selector is the dominant
    cumulative-ratio version. This subclass restores the ablation setting that
    selects PCA directions whose individual eigenvalue ratio lies in [2%, 20%].
    """

    def _select_difficulty_dirs(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_dim = eigenvalues.shape[0]
        if feat_dim == 0:
            return eigenvectors[:, :0], eigenvalues[:0]

        eigvals = torch.clamp(eigenvalues, min=0.0).flip(0)
        eigvecs = eigenvectors.flip(1)
        total = eigvals.sum()
        if total.item() <= 0:
            return eigvecs[:, :1], eigvals[:1]

        ratios = eigvals / total
        keep = (ratios >= self.eigval_lower_bound) & (ratios <= self.eigval_upper_bound)
        if not bool(keep.any()):
            # Robust fallback: choose the direction whose ratio is closest to the band.
            lower_gap = torch.clamp(self.eigval_lower_bound - ratios, min=0.0)
            upper_gap = torch.clamp(ratios - self.eigval_upper_bound, min=0.0)
            nearest = torch.argmin(lower_gap + upper_gap)
            keep = torch.zeros_like(ratios, dtype=torch.bool)
            keep[nearest] = True
        return eigvecs[:, keep], eigvals[keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate SA and calculate masks.")
    parser.add_argument("--dataset", type=str, default=CIFAR100, choices=AVAILABLE_DATASETS)
    parser.add_argument("--kr", type=str, default="20,30,40,50,60,70,80,90")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=str, default="22,42,96")
    parser.add_argument("--weight-group", type=str, default="learned", choices=("naive", "learned"))
    parser.add_argument("--method", type=str, default="group", choices=("topk", "group"))
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=1)
    parser.add_argument("--debug-prompts", action="store_true")
    return parser.parse_args()


def _renormalize_without_ablation(weights: dict[str, object]) -> dict[str, float]:
    out = {k: 0.0 for k in ALL_COMPONENTS}
    active_sum = 0.0
    for key in ACTIVE_COMPONENTS:
        try:
            value = float(weights.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        value = max(value, 0.0)
        out[key] = value
        active_sum += value
    if active_sum <= 1e-12:
        for key in ACTIVE_COMPONENTS:
            out[key] = 1.0 / len(ACTIVE_COMPONENTS)
    else:
        for key in ACTIVE_COMPONENTS:
            out[key] /= active_sum
    out[ABLATION_COMPONENT] = 0.0
    return out


def _load_original_weights(dataset_name: str) -> dict[str, dict[str, object]]:
    path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(dataset_name, {})
    return entry if isinstance(entry, dict) else {}


def load_ablation_weights(dataset_name: str, weight_group: str, seed: int) -> dict[str, float]:
    path = PROJECT_ROOT / "weights" / "ablation_weights.json"
    data: dict[str, object] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded

    dataset_entry = data.setdefault(dataset_name, {})
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        data[dataset_name] = dataset_entry
    ablation_entry = dataset_entry.setdefault(ABLATION_NAME, {})
    if not isinstance(ablation_entry, dict):
        ablation_entry = {}
        dataset_entry[ABLATION_NAME] = ablation_entry

    key = "naive" if weight_group == "naive" else str(seed)
    selected = ablation_entry.get(key)
    if isinstance(selected, dict):
        weights = _renormalize_without_ablation(selected)
    elif weight_group == "naive":
        weights = {"dds": 0.5, "div": 0.5, "sa": 0.0}
    else:
        original = _load_original_weights(dataset_name)
        fallback = original.get(str(seed)) or original.get("learned") or original.get("naive") or {}
        weights = _renormalize_without_ablation(fallback if isinstance(fallback, dict) else {})
        print(
            f"[Warning] weights/ablation_weights.json missing {dataset_name}/{ABLATION_NAME}/{key}; "
            "use renormalized original weights as fallback."
        )

    ablation_entry[key] = weights
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return weights


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    keep_ratios = parse_ratio_list(args.kr)
    seeds = parse_seed_list(args.seed)
    if not keep_ratios or not seeds:
        raise ValueError("kr 和 seed 均不能为空。")

    dataset_for_names = _build_dataset(dataset_name, transform=None)
    class_names = resolve_class_names_for_prompts(
        dataset_name=dataset_name,
        data_root=PROJECT_ROOT / "data",
        class_names=dataset_for_names.classes,  # type: ignore[attr-defined]
    )
    print(f"[Init] {dataset_name} samples={len(dataset_for_names)} classes={len(class_names)}")

    dds_metric = RatioBandDifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        eigval_lower_bound=0.02,
        eigval_upper_bound=0.20,
    )
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=str(PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )

    batch_size = 128
    num_workers = 4
    dds_loader = build_score_loader(dds_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, dataset_name, device, batch_size, num_workers)

    method_name = f"{ABLATION_NAME}_{args.weight_group}_{args.method}"
    total_tasks = len(seeds) * len(keep_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)
        weights = load_ablation_weights(dataset_name, args.weight_group, seed)
        print(f"[Seed {seed}] weights={weights}")

        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()

        def _compute_scores() -> dict[str, np.ndarray]:
            dds_scores = dds_metric.score_dataset(tqdm(dds_loader, desc="Scoring DDS(2%-20%)", unit="batch"), adapter=image_adapter).scores
            div_scores = div_metric.score_dataset(tqdm(div_loader, desc="Scoring Div", unit="batch"), adapter=image_adapter).scores
            sa_scores = sa_metric.score_dataset(
                tqdm(sa_loader, desc="Scoring SA", unit="batch"),
                adapter_image=image_adapter,
                adapter_text=text_adapter,
            ).scores
            return {
                "dds": np.asarray(dds_scores),
                "div": np.asarray(div_scores),
                "sa": np.asarray(sa_scores),
                "labels": np.asarray(dataset_for_names.targets),
            }

        static_start = time.perf_counter()
        static_scores = get_or_compute_static_scores(
            cache_root=PROJECT_ROOT / "static_scores" / "ablation_2_20",
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
            num_samples=len(dataset_for_names),
            compute_fn=_compute_scores,
        )
        static_seconds = time.perf_counter() - static_start

        dds_scores_np = np.asarray(static_scores["dds"], dtype=np.float32)
        div_scores_np = np.asarray(static_scores["div"], dtype=np.float32)
        sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)
        labels = np.asarray(dataset_for_names.targets)

        total_scores_np = weights["dds"] * dds_scores_np + weights["div"] * div_scores_np

        for keep_ratio in keep_ratios:
            task_idx += 1
            print(f"[Mask {task_idx}/{total_tasks}] {method_name} | seed={seed} | kr={keep_ratio}")
            mask_path = resolve_mask_path(
                mode=method_name,
                dataset=dataset_name,
                model=args.model_name,
                seed=seed,
                keep_ratio=keep_ratio,
            )
            if args.skip_saved and mask_path.exists():
                print(f"[Skip] {mask_path}")
                continue

            group_stats = None
            if args.method == "topk":
                mask, selected_by_class = select_topk_mask(
                    total_scores_np,
                    labels,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                )
            else:
                # SA ablation is straightforward: SA is kept for cache compatibility,
                # but its score and weight are zeroed in the group objective.
                mask, selected_by_class, group_stats = select_group_mask(
                    np.zeros_like(sa_scores_np, dtype=np.float32),
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels,
                    weights=weights,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                    device=device,
                    dataset_name=dataset_name,
                    seed=seed,
                    weight_group=args.weight_group,
                    clip_model=args.clip_model,
                    adapter_image_path=str(adapter_paths["image_path"]),
                    div_static_scores=div_scores_np,
                    dds_static_scores=dds_scores_np,
                    group_candidate_pool_size=args.group_candidate_pool_size,
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            print(
                f"saved: {mask_path} | selected={int(mask.sum())} | "
                f"static_seconds={static_seconds:.2f} | total_seconds={time.perf_counter() - total_start:.2f}"
            )
            if group_stats is not None:
                print(
                    f"group_summary: subset_score={group_stats['subset_comprehensive_score']:.6f} | "
                    f"distribution_shift={group_stats['distribution_shift']:.6f}"
                )


if __name__ == "__main__":
    main()
