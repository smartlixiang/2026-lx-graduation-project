from __future__ import annotations
from utils.static_score_cache import get_or_compute_static_scores
from utils.seed import set_seed
from utils.global_config import CONFIG
from scoring import DifficultyDirection, Div, SemanticAlignment
from model.adapter import load_trained_adapters
from dataset.dataset_config import CIFAR10

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from utils.group_lambda import (
    DEFAULT_EPS,
    DEFAULT_M,
    DEFAULT_R,
    compute_balance_penalty,
    get_or_estimate_lambda,
)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIXED_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 CIFAR10 在 cr=20..90 下 topk 与 group_old 的子集评分（Div/DDS 在子集上动态计算）"
    )
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=[CIFAR10])
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--weight-group", type=str, default="naive", choices=["naive", "learned"])
    parser.add_argument("--group-old-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "subset_mask_scores_cifar10.csv")
    return parser.parse_args()


def _build_dataset(transform) -> datasets.CIFAR10:
    data_root = PROJECT_ROOT / "data"
    return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)


def ensure_scoring_weights(path: Path, dataset_name: str) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    updated = False
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded

    dataset_entry = data.get(dataset_name)
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        updated = True

    naive = dataset_entry.get("naive")
    if not isinstance(naive, dict):
        naive = {}
        updated = True

    default_weight = 1.0 / 3.0
    for key in ("dds", "div", "sa"):
        if key not in naive:
            naive[key] = default_weight
            updated = True

    naive_total = 0.0
    for key in ("dds", "div", "sa"):
        try:
            naive[key] = float(naive[key])
        except (TypeError, ValueError):
            naive[key] = default_weight
            updated = True
        naive_total += naive[key]

    if naive_total <= 0:
        for key in ("dds", "div", "sa"):
            naive[key] = default_weight
        updated = True
    elif abs(naive_total - 1.0) > 1e-12:
        for key in ("dds", "div", "sa"):
            naive[key] /= naive_total
        updated = True

    dataset_entry["naive"] = naive
    data[dataset_name] = dataset_entry
    if updated or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return {group_name: group for group_name, group in dataset_entry.items() if isinstance(group, dict)}


def load_scoring_weights(all_weights: dict[str, dict[str, object]], weight_group: str, seed: int) -> dict[str, float]:
    if weight_group == "naive":
        selected = all_weights.get("naive")
        if not isinstance(selected, dict):
            raise KeyError("未找到 naive 权重组")
    else:
        selected = all_weights.get(str(seed))
        if not isinstance(selected, dict):
            raise KeyError(f"未找到 learned 权重组（seed={seed}）")

    required = {"dds", "div", "sa"}
    if not required.issubset(selected.keys()):
        raise KeyError(f"权重缺少键: {required - set(selected.keys())}")
    return {k: float(selected[k]) for k in required}


def build_score_loader(preprocess, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    dataset = _build_dataset(preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def select_global_topk(scores: np.ndarray, cut_ratio: int) -> np.ndarray:
    n = scores.shape[0]
    k = max(1, min(n, int(round(n * cut_ratio / 100.0))))
    idx = np.argpartition(-scores, k - 1)[:k]
    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    return mask


def select_random_mask(n_samples: int, cut_ratio: int) -> np.ndarray:
    k = max(1, min(n_samples, int(round(n_samples * cut_ratio / 100.0))))
    idx = np.random.choice(n_samples, size=k, replace=False)
    mask = np.zeros(n_samples, dtype=np.uint8)
    mask[idx] = 1
    return mask


def compute_subset_score(
    selected_mask: np.ndarray,
    *,
    sa_scores: np.ndarray,
    labels_t: torch.Tensor,
    weights: dict[str, float],
    div_metric: Div,
    dds_metric: DifficultyDirection,
    div_loader: DataLoader,
    dds_loader: DataLoader,
    image_adapter,
    div_features,
    dds_features,
    lambda_value: float,
    labels_np: np.ndarray,
    num_classes: int,
    target_size: int,
) -> tuple[float, float, float, float, float]:
    div_dyn = np.asarray(
        div_metric.score_dataset_dynamic(
            div_loader,
            adapter=image_adapter,
            selected_mask=selected_mask,
            image_features=div_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    dds_dyn = np.asarray(
        dds_metric.score_dataset_dynamic(
            dds_loader,
            adapter=image_adapter,
            selected_mask=selected_mask,
            image_features=dds_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    chosen = selected_mask.astype(bool)
    div_sum = float(div_dyn[chosen].sum())
    dds_sum = float(dds_dyn[chosen].sum())
    total = float((weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_dyn)[chosen].sum())
    penalty = compute_balance_penalty(selected_mask, labels_np, num_classes, target_size)
    adjusted = float(total - lambda_value * penalty)
    return div_sum, dds_sum, total, penalty, adjusted


def select_group_old_best(
    *,
    n_samples: int,
    cut_ratio: int,
    outer_iters: int,
    sa_scores: np.ndarray,
    labels_t: torch.Tensor,
    weights: dict[str, float],
    div_metric: Div,
    dds_metric: DifficultyDirection,
    div_loader: DataLoader,
    dds_loader: DataLoader,
    image_adapter,
    div_features,
    dds_features,
    lambda_value: float,
    labels_np: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, float, float, float, float, float, int]:
    k = max(1, min(n_samples, int(round(n_samples * cut_ratio / 100.0))))
    cur_idx = np.random.choice(n_samples, size=k, replace=False)
    cur_mask = np.zeros(n_samples, dtype=np.uint8)
    cur_mask[cur_idx] = 1

    best_mask = cur_mask.copy()
    best_div, best_dds, best_total, best_penalty, best_total_adj = compute_subset_score(
        cur_mask,
        sa_scores=sa_scores,
        labels_t=labels_t,
        weights=weights,
        div_metric=div_metric,
        dds_metric=dds_metric,
        div_loader=div_loader,
        dds_loader=dds_loader,
        image_adapter=image_adapter,
        div_features=div_features,
        dds_features=dds_features,
        lambda_value=lambda_value,
        labels_np=labels_np,
        num_classes=num_classes,
        target_size=k,
    )
    best_iter = 0

    for t in range(1, outer_iters + 1):
        div_dyn = np.asarray(
            div_metric.score_dataset_dynamic(
                div_loader,
                adapter=image_adapter,
                selected_mask=cur_mask,
                image_features=div_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        dds_dyn = np.asarray(
            dds_metric.score_dataset_dynamic(
                dds_loader,
                adapter=image_adapter,
                selected_mask=cur_mask,
                image_features=dds_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        total_scores = weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_dyn
        next_mask = select_global_topk(total_scores, cut_ratio=cut_ratio)
        cur_mask = next_mask

        div_sum, dds_sum, total_sum, pen_sum, total_sum_adj = compute_subset_score(
            cur_mask,
            sa_scores=sa_scores,
            labels_t=labels_t,
            weights=weights,
            div_metric=div_metric,
            dds_metric=dds_metric,
            div_loader=div_loader,
            dds_loader=dds_loader,
            image_adapter=image_adapter,
            div_features=div_features,
            dds_features=dds_features,
            lambda_value=lambda_value,
            labels_np=labels_np,
            num_classes=num_classes,
            target_size=k,
        )
        if total_sum_adj > best_total_adj:
            best_total_adj = total_sum_adj
            best_total = total_sum
            best_div = div_sum
            best_dds = dds_sum
            best_penalty = pen_sum
            best_mask = cur_mask.copy()
            best_iter = t

    return best_mask, best_div, best_dds, best_total, best_penalty, best_total_adj, best_iter


def main() -> None:
    args = parse_args()
    set_seed(FIXED_SEED)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    dataset_for_names = _build_dataset(transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    labels = np.asarray(dataset_for_names.targets)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, args.dataset)
    weights = load_scoring_weights(all_weights, args.weight_group, FIXED_SEED)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    dds_loader = build_score_loader(dds_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    div_loader = build_score_loader(div_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    sa_loader = build_score_loader(sa_metric.extractor.preprocess, device, args.batch_size, args.num_workers)

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=FIXED_SEED,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {
            "sa": np.asarray(sa_scores_local),
            "div": np.asarray(div_scores_local),
            "dds": np.asarray(dds_scores_local),
            "labels": labels,
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=args.dataset,
        seed=FIXED_SEED,
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

    sa_scores = np.asarray(static_scores["sa"], dtype=np.float32)
    div_scores = np.asarray(static_scores["div"], dtype=np.float32)
    dds_scores = np.asarray(static_scores["dds"], dtype=np.float32)
    total_static = weights["sa"] * sa_scores + weights["div"] * div_scores + weights["dds"] * dds_scores

    start = time.perf_counter()
    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    dds_features, _ = dds_metric._encode_images(dds_loader, image_adapter)
    print(f"[Init] Encoded image features for dynamic scoring in {time.perf_counter() - start:.2f}s")

    cut_ratios = list(range(20, 91, 10))
    eval_seeds = [1, 2, 3, 4, 5]
    lambda_sample_M = DEFAULT_M
    lambda_ratio_r = DEFAULT_R
    lambda_eps = DEFAULT_EPS
    rows: list[dict[str, object]] = []

    for cr in tqdm(cut_ratios, desc="Evaluating cut ratios", unit="cr"):
        print(f"\n[CR={cr}] evaluating topk/random/group_old ...")
        target_size = max(1, min(len(dataset_for_names), int(round(len(dataset_for_names) * cr / 100.0))))
        lambda_info_topk = get_or_estimate_lambda(
            cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
            dataset=args.dataset,
            seed=FIXED_SEED,
            cr=cr,
            weight_group=args.weight_group,
            n_samples=len(dataset_for_names),
            target_size=target_size,
            eval_score_fn=lambda mask: compute_subset_score(
                mask,
                sa_scores=sa_scores,
                labels_t=labels_t,
                weights=weights,
                div_metric=div_metric,
                dds_metric=dds_metric,
                div_loader=div_loader,
                dds_loader=dds_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                dds_features=dds_features,
                lambda_value=0.0,
                labels_np=labels,
                num_classes=len(class_names),
                target_size=target_size,
            )[2],
            penalty_fn=lambda mask: compute_balance_penalty(mask, labels, len(class_names), target_size),
            M=lambda_sample_M,
            r=lambda_ratio_r,
            eps=lambda_eps,
            tqdm_desc=f"Estimating lambda (seed={FIXED_SEED}, cr={cr})",
        )
        lambda_topk = float(lambda_info_topk["lambda"])
        print(f"[Lambda] dataset={args.dataset} | seed={FIXED_SEED} | cr={cr} | lambda={lambda_topk:.8f}")
        topk_mask = select_global_topk(total_static, cut_ratio=cr)
        _, _, _, _, topk_total = compute_subset_score(
            topk_mask,
            sa_scores=sa_scores,
            labels_t=labels_t,
            weights=weights,
            div_metric=div_metric,
            dds_metric=dds_metric,
            div_loader=div_loader,
            dds_loader=dds_loader,
            image_adapter=image_adapter,
            div_features=div_features,
            dds_features=dds_features,
            lambda_value=lambda_topk,
            labels_np=labels,
            num_classes=len(class_names),
            target_size=target_size,
        )

        random_scores: list[float] = []
        group_old_scores: list[float] = []
        for seed in tqdm(eval_seeds, desc=f"CR={cr} seeds", unit="seed", leave=False):
            set_seed(seed)

            random_mask = select_random_mask(len(dataset_for_names), cut_ratio=cr)
            lambda_info_seed = get_or_estimate_lambda(
                cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
                dataset=args.dataset,
                seed=seed,
                cr=cr,
                weight_group=args.weight_group,
                n_samples=len(dataset_for_names),
                target_size=target_size,
                eval_score_fn=lambda mask: compute_subset_score(
                    mask,
                    sa_scores=sa_scores,
                    labels_t=labels_t,
                    weights=weights,
                    div_metric=div_metric,
                    dds_metric=dds_metric,
                    div_loader=div_loader,
                    dds_loader=dds_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    dds_features=dds_features,
                    lambda_value=0.0,
                    labels_np=labels,
                    num_classes=len(class_names),
                    target_size=target_size,
                )[2],
                penalty_fn=lambda mask: compute_balance_penalty(mask, labels, len(class_names), target_size),
                M=lambda_sample_M,
                r=lambda_ratio_r,
                eps=lambda_eps,
                tqdm_desc=f"Estimating lambda (seed={seed}, cr={cr})",
            )
            lambda_seed = float(lambda_info_seed["lambda"])

            _, _, _, _, random_total = compute_subset_score(
                random_mask,
                sa_scores=sa_scores,
                labels_t=labels_t,
                weights=weights,
                div_metric=div_metric,
                dds_metric=dds_metric,
                div_loader=div_loader,
                dds_loader=dds_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                dds_features=dds_features,
                lambda_value=lambda_seed,
                labels_np=labels,
                num_classes=len(class_names),
                target_size=target_size,
            )
            random_scores.append(random_total)

            _, _, _, _, _, grp_total, _ = select_group_old_best(
                n_samples=len(dataset_for_names),
                cut_ratio=cr,
                outer_iters=args.group_old_iters,
                sa_scores=sa_scores,
                labels_t=labels_t,
                weights=weights,
                div_metric=div_metric,
                dds_metric=dds_metric,
                div_loader=div_loader,
                dds_loader=dds_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                dds_features=dds_features,
                lambda_value=lambda_seed,
                labels_np=labels,
                num_classes=len(class_names),
            )
            group_old_scores.append(grp_total)

            print(
                f"  seed={seed} | random_total={random_total:.4f} | group_old_total={grp_total:.4f}"
            )

        rows.append(
            {
                "cr": cr,
                "topk": topk_total,
                "random_mean": float(np.mean(random_scores)),
                "random_max": float(np.max(random_scores)),
                "group_old_mean": float(np.mean(group_old_scores)),
                "group_old_max": float(np.max(group_old_scores)),
            }
        )

        print(
            f"[CR={cr}] topk={topk_total:.4f}, random_mean={np.mean(random_scores):.4f}, random_max={np.max(random_scores):.4f}, "
            f"group_old_mean={np.mean(group_old_scores):.4f}, group_old_max={np.max(group_old_scores):.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cr",
                "topk",
                "random_mean",
                "random_max",
                "group_old_mean",
                "group_old_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
