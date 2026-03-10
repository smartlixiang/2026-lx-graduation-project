from __future__ import annotations
from utils.static_score_cache import get_or_compute_static_scores
from utils.seed import set_seed
from utils.global_config import CONFIG
from scoring import DifficultyDirection, Div, SemanticAlignment
from model.adapter import load_trained_adapters
from dataset.dataset_config import CIFAR10, CIFAR100
from utils.path_rules import resolve_mask_path

import argparse
import hashlib
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
        description="计算 CIFAR10 在 kr=20..90 下 topk 与 group_old 的子集评分（Div 动态，DDS 静态）"
    )
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=[CIFAR10, CIFAR100])
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--weight-group", type=str, default="naive", choices=["naive", "learned"])
    parser.add_argument("--group-old-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="可选输出路径；默认保存到 group_baseline/<dataset>/<weight-group>.csv")
    return parser.parse_args()


def _build_dataset(dataset_name: str, transform):
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


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


def build_score_loader(dataset_name: str, preprocess, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    dataset = _build_dataset(dataset_name, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mean_stats_cache_path(dataset_name: str, clip_model: str, adapter_image_path: str) -> Path:
    adapter_sha1 = _hash_file(Path(adapter_image_path))
    clip_tag = clip_model.replace("/", "-").replace(" ", "_")
    return PROJECT_ROOT / "static_scores" / "group_mean_stats" / dataset_name / clip_tag / f"img_adapter_{adapter_sha1}.npz"


def _get_or_compute_group_mean_stats(
    *,
    cache_path: Path,
    image_features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(image_features.shape[0])
    feat_dim = int(image_features.shape[1]) if image_features.ndim == 2 else 0
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            means = np.asarray(cached["full_class_mean"], dtype=np.float32)
            vars_ = np.asarray(cached["full_class_var"], dtype=np.float32)
            if means.shape == (num_classes, feat_dim) and vars_.shape == (num_classes,):
                return means, vars_
        except Exception:
            pass

    full_class_mean = np.zeros((num_classes, feat_dim), dtype=np.float32)
    full_class_var = np.zeros((num_classes,), dtype=np.float32)
    for class_id in range(num_classes):
        class_feats = image_features[labels == class_id]
        if class_feats.shape[0] == 0:
            continue
        class_mean = np.mean(class_feats, axis=0, dtype=np.float32)
        diff = class_feats - class_mean
        sigma2 = float(np.mean(np.sum(diff * diff, axis=1)))
        full_class_mean[class_id] = class_mean
        full_class_var[class_id] = np.float32(max(sigma2, 0.0))

    np.savez_compressed(cache_path, full_class_mean=full_class_mean, full_class_var=full_class_var)
    return full_class_mean, full_class_var


def compute_mean_penalty(
    selected_mask: np.ndarray,
    *,
    labels_np: np.ndarray,
    num_classes: int,
    image_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
    eps: float,
) -> float:
    selected = selected_mask.astype(bool)
    penalty_sum = 0.0
    for class_id in range(num_classes):
        class_selected = selected & (labels_np == class_id)
        if int(np.sum(class_selected)) <= 0:
            continue
        subset_mean = np.mean(image_features_np[class_selected], axis=0, dtype=np.float32)
        diff = subset_mean - full_class_mean[class_id]
        dist2 = float(np.dot(diff, diff))
        penalty_sum += dist2 / (float(full_class_var[class_id]) + eps)
    return float(penalty_sum)


def select_global_topk(scores: np.ndarray, keep_ratio: int) -> np.ndarray:
    n = scores.shape[0]
    k = max(1, min(n, int(round(n * keep_ratio / 100.0))))
    idx = np.argpartition(-scores, k - 1)[:k]
    mask = np.zeros(n, dtype=np.uint8)
    mask[idx] = 1
    return mask


def select_random_mask(n_samples: int, keep_ratio: int) -> np.ndarray:
    k = max(1, min(n_samples, int(round(n_samples * keep_ratio / 100.0))))
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
    dds_scores: np.ndarray,
    div_loader: DataLoader,
    image_adapter,
    div_features,
    lambda_cls: float,
    lambda_mean: float,
    labels_np: np.ndarray,
    num_classes: int,
    target_size: int,
    image_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
    eps: float,
) -> tuple[float, float, float, float, float, float]:
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
    chosen = selected_mask.astype(bool)
    div_sum = float(div_dyn[chosen].sum())
    dds_sum = float(dds_scores[chosen].sum())
    total = float((weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_scores)[chosen].sum())
    penalty = compute_balance_penalty(selected_mask, labels_np, num_classes, target_size)
    mean_penalty = compute_mean_penalty(
        selected_mask,
        labels_np=labels_np,
        num_classes=num_classes,
        image_features_np=image_features_np,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        eps=eps,
    )
    adjusted = float(total - lambda_cls * penalty - lambda_mean * mean_penalty)
    return div_sum, dds_sum, total, penalty, mean_penalty, adjusted


def select_group_old_best(
    *,
    n_samples: int,
    keep_ratio: int,
    outer_iters: int,
    sa_scores: np.ndarray,
    labels_t: torch.Tensor,
    weights: dict[str, float],
    div_metric: Div,
    dds_scores: np.ndarray,
    div_loader: DataLoader,
    image_adapter,
    div_features,
    lambda_cls: float,
    lambda_mean: float,
    labels_np: np.ndarray,
    num_classes: int,
    image_features_np: np.ndarray,
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float, float, float, int]:
    k = max(1, min(n_samples, int(round(n_samples * keep_ratio / 100.0))))
    cur_idx = np.random.choice(n_samples, size=k, replace=False)
    cur_mask = np.zeros(n_samples, dtype=np.uint8)
    cur_mask[cur_idx] = 1

    best_mask = cur_mask.copy()
    best_div, best_dds, best_total, best_penalty, best_mean_penalty, best_total_adj = compute_subset_score(
        cur_mask,
        sa_scores=sa_scores,
        labels_t=labels_t,
        weights=weights,
        div_metric=div_metric,
        dds_scores=dds_scores,
        div_loader=div_loader,
        image_adapter=image_adapter,
        div_features=div_features,
        lambda_cls=lambda_cls,
        lambda_mean=lambda_mean,
        labels_np=labels_np,
        num_classes=num_classes,
        target_size=k,
        image_features_np=image_features_np,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        eps=DEFAULT_EPS,
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
        total_scores = weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_scores
        next_mask = select_global_topk(total_scores, keep_ratio=keep_ratio)
        cur_mask = next_mask

        div_sum, dds_sum, total_sum, pen_sum, mean_pen_sum, total_sum_adj = compute_subset_score(
            cur_mask,
            sa_scores=sa_scores,
            labels_t=labels_t,
            weights=weights,
            div_metric=div_metric,
            dds_scores=dds_scores,
            div_loader=div_loader,
            image_adapter=image_adapter,
            div_features=div_features,
            lambda_cls=lambda_cls,
            lambda_mean=lambda_mean,
            labels_np=labels_np,
            num_classes=num_classes,
            target_size=k,
            image_features_np=image_features_np,
            full_class_mean=full_class_mean,
            full_class_var=full_class_var,
            eps=DEFAULT_EPS,
        )
        if total_sum_adj > best_total_adj:
            best_total_adj = total_sum_adj
            best_total = total_sum
            best_div = div_sum
            best_dds = dds_sum
            best_penalty = pen_sum
            best_mean_penalty = mean_pen_sum
            best_mask = cur_mask.copy()
            best_iter = t

    return best_mask, best_div, best_dds, best_total, best_penalty, best_mean_penalty, best_total_adj, best_iter


def main() -> None:
    args = parse_args()
    set_seed(FIXED_SEED)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    dataset_for_names = _build_dataset(args.dataset, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    labels = np.asarray(dataset_for_names.targets)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, args.dataset)
    weights = load_scoring_weights(all_weights, args.weight_group, FIXED_SEED)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    dds_loader = build_score_loader(args.dataset, dds_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    div_loader = build_score_loader(args.dataset, div_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    sa_loader = build_score_loader(args.dataset, sa_metric.extractor.preprocess, device, args.batch_size, args.num_workers)

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
    div_features_np = (
        div_features.detach().cpu().numpy() if isinstance(div_features, torch.Tensor) else np.asarray(div_features)
    ).astype(np.float32)
    mean_stats_cache = _mean_stats_cache_path(args.dataset, args.clip_model, str(adapter_paths["image_path"]))
    full_class_mean, full_class_var = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache,
        image_features=div_features_np,
        labels=labels,
        num_classes=len(class_names),
    )
    print(f"[Init] Encoded image features for dynamic scoring in {time.perf_counter() - start:.2f}s")

    keep_ratios = list(range(20, 91, 10))
    eval_seeds = [22, 42, 96]
    lambda_sample_M = DEFAULT_M
    lambda_ratio_r = DEFAULT_R
    lambda_eps = DEFAULT_EPS
    rows: list[dict[str, object]] = []

    for kr in tqdm(keep_ratios, desc="Evaluating keep ratios", unit="kr"):
        print(f"\n[KR={kr}] evaluating topk/random/group_old ...")
        target_size = max(1, min(len(dataset_for_names), int(round(len(dataset_for_names) * kr / 100.0))))
        lambda_info_topk = get_or_estimate_lambda(
            cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
            dataset=args.dataset,
            seed=FIXED_SEED,
            kr=kr,
            weight_group=args.weight_group,
            n_samples=len(dataset_for_names),
            target_size=target_size,
            eval_score_fn=lambda mask: compute_subset_score(
                mask,
                sa_scores=sa_scores,
                labels_t=labels_t,
                weights=weights,
                div_metric=div_metric,
                dds_scores=dds_scores,
                div_loader=div_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                lambda_cls=0.0,
                lambda_mean=0.0,
                labels_np=labels,
                num_classes=len(class_names),
                target_size=target_size,
                image_features_np=div_features_np,
                full_class_mean=full_class_mean,
                full_class_var=full_class_var,
                eps=lambda_eps,
            )[2],
            penalty_fn=lambda mask: compute_balance_penalty(mask, labels, len(class_names), target_size),
            mean_penalty_fn=lambda mask: compute_mean_penalty(
                mask,
                labels_np=labels,
                num_classes=len(class_names),
                image_features_np=div_features_np,
                full_class_mean=full_class_mean,
                full_class_var=full_class_var,
                eps=lambda_eps,
            ),
            M=lambda_sample_M,
            r=lambda_ratio_r,
            eps=lambda_eps,
            tqdm_desc=f"Estimating lambda (seed={FIXED_SEED}, kr={kr})",
        )
        lambda_topk_cls = float(lambda_info_topk.get("lambda_cls", lambda_info_topk["lambda"]))
        lambda_topk_mean = float(lambda_info_topk["lambda_mean"])
        print(
            f"[Lambda] dataset={args.dataset} | seed={FIXED_SEED} | kr={kr} "
            f"| lambda_cls={lambda_topk_cls:.8f} | lambda_mean={lambda_topk_mean:.8f}"
        )
        topk_mask = select_global_topk(total_static, keep_ratio=kr)
        _, _, _, _, _, topk_total = compute_subset_score(
            topk_mask,
            sa_scores=sa_scores,
            labels_t=labels_t,
            weights=weights,
            div_metric=div_metric,
            dds_scores=dds_scores,
            div_loader=div_loader,
            image_adapter=image_adapter,
            div_features=div_features,
            lambda_cls=lambda_topk_cls,
            lambda_mean=lambda_topk_mean,
            labels_np=labels,
            num_classes=len(class_names),
            target_size=target_size,
            image_features_np=div_features_np,
            full_class_mean=full_class_mean,
            full_class_var=full_class_var,
            eps=lambda_eps,
        )

        random_scores: list[float] = []
        group_old_scores: list[float] = []
        herding_scores: list[float] = []
        for seed in tqdm(eval_seeds, desc=f"KR={kr} seeds", unit="seed", leave=False):
            set_seed(seed)

            lambda_info_seed = get_or_estimate_lambda(
                cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
                dataset=args.dataset,
                seed=seed,
                kr=kr,
                weight_group=args.weight_group,
                n_samples=len(dataset_for_names),
                target_size=target_size,
                eval_score_fn=lambda mask: compute_subset_score(
                    mask,
                    sa_scores=sa_scores,
                    labels_t=labels_t,
                    weights=weights,
                    div_metric=div_metric,
                    dds_scores=dds_scores,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    lambda_cls=0.0,
                    lambda_mean=0.0,
                    labels_np=labels,
                    num_classes=len(class_names),
                    target_size=target_size,
                    image_features_np=div_features_np,
                    full_class_mean=full_class_mean,
                    full_class_var=full_class_var,
                    eps=lambda_eps,
                )[2],
                penalty_fn=lambda mask: compute_balance_penalty(mask, labels, len(class_names), target_size),
                mean_penalty_fn=lambda mask: compute_mean_penalty(
                    mask,
                    labels_np=labels,
                    num_classes=len(class_names),
                    image_features_np=div_features_np,
                    full_class_mean=full_class_mean,
                    full_class_var=full_class_var,
                    eps=lambda_eps,
                ),
                M=lambda_sample_M,
                r=lambda_ratio_r,
                eps=lambda_eps,
                tqdm_desc=f"Estimating lambda (seed={seed}, kr={kr})",
            )
            lambda_seed_cls = float(lambda_info_seed.get("lambda_cls", lambda_info_seed["lambda"]))
            lambda_seed_mean = float(lambda_info_seed["lambda_mean"])

            random_repeat = 2
            for rep_idx in range(random_repeat):
                random_mask = select_random_mask(len(dataset_for_names), keep_ratio=kr)
                _, _, _, _, _, random_total = compute_subset_score(
                    random_mask,
                    sa_scores=sa_scores,
                    labels_t=labels_t,
                    weights=weights,
                    div_metric=div_metric,
                    dds_scores=dds_scores,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    lambda_cls=lambda_seed_cls,
                    lambda_mean=lambda_seed_mean,
                    labels_np=labels,
                    num_classes=len(class_names),
                    target_size=target_size,
                    image_features_np=div_features_np,
                    full_class_mean=full_class_mean,
                    full_class_var=full_class_var,
                    eps=lambda_eps,
                )
                random_scores.append(random_total)
                print(f"  seed={seed} rep={rep_idx + 1}/{random_repeat} | random_total={random_total:.4f}")

            _, _, _, _, _, _, grp_total, _ = select_group_old_best(
                n_samples=len(dataset_for_names),
                keep_ratio=kr,
                outer_iters=args.group_old_iters,
                sa_scores=sa_scores,
                labels_t=labels_t,
                weights=weights,
                div_metric=div_metric,
                dds_scores=dds_scores,
                div_loader=div_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                lambda_cls=lambda_seed_cls,
                lambda_mean=lambda_seed_mean,
                labels_np=labels,
                num_classes=len(class_names),
                image_features_np=div_features_np,
                full_class_mean=full_class_mean,
                full_class_var=full_class_var,
            )
            group_old_scores.append(grp_total)

            herding_mask_path = resolve_mask_path(
                "herding",
                args.dataset,
                args.model_name,
                seed,
                kr,
            )
            if herding_mask_path.exists():
                try:
                    with np.load(herding_mask_path, allow_pickle=False) as loaded_mask:
                        if "mask" in loaded_mask:
                            herding_mask = np.asarray(loaded_mask["mask"], dtype=np.uint8)
                        else:
                            first_key = next(iter(loaded_mask.files), None)
                            herding_mask = np.asarray(loaded_mask[first_key], dtype=np.uint8) if first_key else np.zeros(len(dataset_for_names), dtype=np.uint8)
                    if herding_mask.shape[0] == len(dataset_for_names):
                        _, _, _, _, _, herding_total = compute_subset_score(
                            herding_mask,
                            sa_scores=sa_scores,
                            labels_t=labels_t,
                            weights=weights,
                            div_metric=div_metric,
                            dds_scores=dds_scores,
                            div_loader=div_loader,
                            image_adapter=image_adapter,
                            div_features=div_features,
                            lambda_cls=lambda_seed_cls,
                            lambda_mean=lambda_seed_mean,
                            labels_np=labels,
                            num_classes=len(class_names),
                            target_size=target_size,
                            image_features_np=div_features_np,
                            full_class_mean=full_class_mean,
                            full_class_var=full_class_var,
                            eps=lambda_eps,
                        )
                        herding_scores.append(herding_total)
                    else:
                        print(f"[HerdingMask] invalid length for {herding_mask_path}, expect {len(dataset_for_names)}")
                except Exception as exc:
                    print(f"[HerdingMask] failed to load {herding_mask_path}: {exc}")
            else:
                print(f"[HerdingMask] missing local mask: {herding_mask_path}")

            print(f"  seed={seed} | group_old_total={grp_total:.4f}")

        rows.append(
            {
                "kr": kr,
                "topk": topk_total,
                "random_mean": float(np.mean(random_scores)),
                "random_max": float(np.max(random_scores)),
                "group_old_mean": float(np.mean(group_old_scores)),
                "group_old_max": float(np.max(group_old_scores)),
                "herding_mean": float(np.mean(herding_scores)) if herding_scores else float("nan"),
                "herding_max": float(np.max(herding_scores)) if herding_scores else float("nan"),
            }
        )

        print(
            f"[KR={kr}] topk={topk_total:.4f}, random_mean={np.mean(random_scores):.4f}, random_max={np.max(random_scores):.4f}, "
            f"group_old_mean={np.mean(group_old_scores):.4f}, group_old_max={np.max(group_old_scores):.4f}, "
            f"herding_mean={(np.mean(herding_scores) if herding_scores else float('nan')):.4f}, "
            f"herding_max={(np.max(herding_scores) if herding_scores else float('nan')):.4f}"
        )

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "group_baseline" / args.dataset / f"{args.weight_group}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kr",
                "topk",
                "random_mean",
                "random_max",
                "group_old_mean",
                "group_old_max",
                "herding_mean",
                "herding_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
