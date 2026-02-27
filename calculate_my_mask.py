from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import load_trained_adapters  # noqa: E402
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402
from utils.group_lambda import (  # noqa: E402
    DEFAULT_EPS,
    DEFAULT_M,
    DEFAULT_R,
    compute_balance_penalty,
    get_or_estimate_lambda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate selection masks")
    parser.add_argument(
        "--dataset",
        type=str,
        default=CIFAR10,
        choices=AVAILABLE_DATASETS,
        help="目标数据集名称",
    )
    parser.add_argument(
        "--cr",
        type=str,
        default="80",
        help="cut_ratio 列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(CONFIG.global_seed),
        help="随机种子列表，逗号分隔",
    )
    parser.add_argument(
        "--weight-group",
        type=str,
        default="naive",
        help="权重组，仅支持 {naive, learned}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="topk",
        help="数据选择方式，可选 {topk, group}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="mask 保存路径中的模型名称",
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="group 模式下是否额外对比 topk",
    )
    return parser.parse_args()


def _build_dataset(dataset_name: str, transform) -> datasets.VisionDataset:
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_ratio_list(ratio_text: str) -> list[int]:
    cleaned = ratio_text.strip()
    if not cleaned:
        return []
    if "," in cleaned:
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [cleaned]
    return [int(item) for item in items]


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
    return {
        group_name: group
        for group_name, group in dataset_entry.items()
        if isinstance(group, dict)
    }


def _to_weight_triplet(selected: dict[str, object], group_name: str) -> dict[str, float]:
    required = {"dds", "div", "sa"}
    missing = required - selected.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"权重组 {group_name} 缺少必要键: {missing_str}")
    weights: dict[str, float] = {}
    for key in sorted(required):
        value = selected[key]
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group_name} 的 {key} 无法转换为 float。") from exc
    return weights


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    weight_group: str,
    seed: int,
) -> dict[str, float]:
    mode = weight_group.strip().lower()
    if mode not in {"naive", "learned"}:
        raise ValueError("weight-group 仅支持 {'naive', 'learned'}")

    if mode == "naive":
        selected = all_weights.get("naive")
        if selected is None or not isinstance(selected, dict):
            raise KeyError("未找到 naive 权重组。")
        return _to_weight_triplet(selected, "naive")

    selected = all_weights.get(str(seed))
    if selected is None or not isinstance(selected, dict):
        raise KeyError(f"未找到 learned 权重组（seed={seed}）。")
    return _to_weight_triplet(selected, str(seed))


def build_score_loader(
    preprocess,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = _build_dataset(dataset_name, preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def select_topk_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cut_ratio: int,
) -> tuple[np.ndarray, dict[int, int]]:
    if cut_ratio <= 0 or cut_ratio > 100:
        raise ValueError("cr 必须在 1-100 之间。")
    mask = np.zeros(scores.shape[0], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    ratio = cut_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        if cut_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        class_scores = scores[class_indices]
        topk_indices = class_indices[
            np.argpartition(-class_scores, num_select - 1)[:num_select]
        ]
        mask[topk_indices] = 1
        selected_by_class[class_id] = int(num_select)
    return mask, selected_by_class


def select_group_mask(
    sa_scores: np.ndarray,
    dds_metric: DifficultyDirection,
    div_metric: Div,
    dds_loader: DataLoader,
    div_loader: DataLoader,
    image_adapter,
    labels: np.ndarray,
    weights: dict[str, float],
    num_classes: int,
    cut_ratio: int,
    device: torch.device,
    dataset_name: str,
    seed: int,
    weight_group: str,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if cut_ratio <= 0 or cut_ratio > 100:
        raise ValueError("cr 必须在 1-100 之间。")

    num_samples = sa_scores.shape[0]
    if labels.shape[0] != num_samples:
        raise ValueError("sa_scores 与 labels 的样本数不一致。")

    sr = float(cut_ratio) / 100.0
    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    dds_features, _ = dds_metric._encode_images(dds_loader, image_adapter)

    target_size = int(round(sr * num_samples))
    if num_samples > 0:
        target_size = min(num_samples, max(1, target_size))
    else:
        target_size = 0

    ga_population_size = 8
    ga_generations = 100
    ga_offspring = ga_population_size
    lambda_sample_M = DEFAULT_M
    lambda_ratio_r = DEFAULT_R
    lambda_eps = DEFAULT_EPS
    # ====== adaptive schedule block ======
    level = 1
    WARMUP = 5
    STALL_TRIGGER = 3
    EPS = 1e-8

    cr_ratio = cut_ratio / 100.0

    ls_max = 0.06 - 0.03 * cr_ratio
    ls_min = 0.3 * ls_max
    ls_step = (ls_max - ls_min) / 3.0
    ls_levels = [ls_min + level_idx * ls_step for level_idx in range(4)]

    m_max = 2.0 + 3.0 * (1.0 - cr_ratio)
    mut_max = 0.01 * m_max
    mut_min = mut_max / 4.0
    mut_step = (mut_max - mut_min) / 3.0
    mut_levels = [mut_min + level_idx * mut_step for level_idx in range(4)]

    C_MAX = 0.8
    c_min = 0.7 - 0.2 * (1 - cr_ratio)
    c_step = (C_MAX - c_min) / 3.0
    c_levels = [c_min + level_idx * c_step for level_idx in range(4)]

    stall_counter = 0

    cached_real_stats: dict[tuple[int, ...], tuple[float, np.ndarray, np.ndarray]] = {}

    def _real_stats_cached(cur_mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        cur_key = tuple(np.flatnonzero(cur_mask).tolist())
        cached = cached_real_stats.get(cur_key)
        if cached is not None:
            return cached
        div_scores = np.asarray(
            div_metric.score_dataset_dynamic(
                div_loader,
                adapter=image_adapter,
                selected_mask=cur_mask,
                image_features=div_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        dds_scores = np.asarray(
            dds_metric.score_dataset_dynamic(
                dds_loader,
                adapter=image_adapter,
                selected_mask=cur_mask,
                image_features=dds_features,
                labels=labels_t,
            ).scores,
            dtype=np.float32,
        )
        s_ref = (
            weights["sa"] * sa_scores_np
            + weights["div"] * div_scores
            + weights["dds"] * dds_scores
        )
        counts = np.bincount(labels_np[cur_mask.astype(bool)], minlength=num_classes).astype(np.int64)
        s_val = float(np.sum(s_ref[cur_mask.astype(bool)]))
        result = (s_val, s_ref, counts)
        cached_real_stats[cur_key] = result
        return result

    def _indices_to_mask(indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(num_samples, dtype=np.uint8)
        if indices.size > 0:
            mask[indices] = 1
        return mask

    def _pick_top_static(candidate_idx: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or candidate_idx.size == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(-sa_scores_np[candidate_idx])
        return candidate_idx[order[: min(k, candidate_idx.size)]].astype(np.int64)

    def _repair_size(indices: np.ndarray) -> np.ndarray:
        unique_idx = np.unique(indices.astype(np.int64))
        if target_size == 0:
            return np.empty(0, dtype=np.int64)
        if unique_idx.size > target_size:
            return _pick_top_static(unique_idx, target_size)
        if unique_idx.size < target_size:
            comp = np.setdiff1d(np.arange(num_samples, dtype=np.int64), unique_idx, assume_unique=False)
            fill = _pick_top_static(comp, target_size - unique_idx.size)
            unique_idx = np.concatenate([unique_idx, fill])
        if unique_idx.size > target_size:
            unique_idx = _pick_top_static(unique_idx, target_size)
        return unique_idx.astype(np.int64)

    def _evaluate(indices: np.ndarray) -> dict[str, object]:
        mask = _indices_to_mask(indices)
        raw_score, s_ref, counts = _real_stats_cached(mask)
        penalty = compute_balance_penalty(mask, labels_np, num_classes, target_size)
        adjusted_score = raw_score - lambda_val * penalty
        return {
            "indices": np.sort(indices.astype(np.int64)),
            "mask": mask,
            "fitness": float(adjusted_score),
            "raw_score": float(raw_score),
            "penalty": float(penalty),
            "adjusted_score": float(adjusted_score),
            "s_ref": s_ref,
            "counts": counts,
        }

    def _local_search_step(child: np.ndarray, proxy_scores: np.ndarray, k_ls_value: int) -> np.ndarray:
        k_ls_value = min(k_ls_value, num_samples - child.size, child.size)
        if k_ls_value <= 0:
            return child
        child_order = np.argsort(proxy_scores[child])
        ls_drop = child[child_order[:k_ls_value]]
        child_comp = np.setdiff1d(np.arange(num_samples, dtype=np.int64), child, assume_unique=False)
        if child_comp.size == 0:
            return child
        ls_add_order = np.argsort(-proxy_scores[child_comp])
        ls_add = child_comp[ls_add_order[:k_ls_value]]
        child = np.setdiff1d(child, ls_drop, assume_unique=False)
        child = np.concatenate([child, ls_add])
        return child

    def _tournament_select(population: list[dict[str, object]]) -> dict[str, object]:
        if len(population) == 1:
            return population[0]
        chosen = np.random.choice(len(population), size=2, replace=False)
        a = population[int(chosen[0])]
        b = population[int(chosen[1])]
        return a if float(a["fitness"]) >= float(b["fitness"]) else b

    static_topk = _pick_top_static(np.arange(num_samples, dtype=np.int64), target_size)

    lambda_info = get_or_estimate_lambda(
        cache_path=PROJECT_ROOT / "utils" / "group_lambda.json",
        dataset=dataset_name,
        seed=seed,
        cr=cut_ratio,
        weight_group=weight_group,
        n_samples=num_samples,
        target_size=target_size,
        eval_score_fn=lambda mask: _real_stats_cached(mask)[0],
        penalty_fn=lambda mask: compute_balance_penalty(mask, labels_np, num_classes, target_size),
        M=lambda_sample_M,
        r=lambda_ratio_r,
        eps=lambda_eps,
        tqdm_desc=f"Estimating lambda (seed={seed}, cr={cut_ratio})",
    )
    lambda_val = float(lambda_info["lambda"])
    print(
        "[Lambda] "
        f"dataset={dataset_name} | seed={seed} | cr={cut_ratio} | "
        f"lambda={lambda_val:.8f} | "
        f"mu_S={lambda_info.get('mu_S', float('nan')):.6f}, sigma_S={lambda_info.get('sigma_S', float('nan')):.6f}, "
        f"mu_pen={lambda_info.get('mu_pen', float('nan')):.6f}, sigma_pen={lambda_info.get('sigma_pen', float('nan')):.6f}, "
        f"M={int(lambda_info.get('M', lambda_sample_M))}, r={lambda_info.get('r', lambda_ratio_r):.4f}"
    )

    initial_population_idx: list[np.ndarray] = [static_topk]
    while len(initial_population_idx) < ga_population_size:
        random_idx = np.random.choice(num_samples, size=target_size, replace=False).astype(np.int64)
        initial_population_idx.append(np.sort(random_idx))

    population = [_evaluate(_repair_size(idx)) for idx in initial_population_idx]
    best_initial = max(population, key=lambda item: float(item["fitness"]))
    score_history = [float(best_initial["raw_score"])]
    correction_history = [float(lambda_val * float(best_initial["penalty"]))]

    generation_iter = tqdm(
        range(ga_generations),
        desc="Group optimization [Memetic-GA]",
        unit="gen",
        leave=True,
    )

    best_fitness = max(float(item["fitness"]) for item in population)

    for gen_idx in generation_iter:
        cur_best = float(population[0]["fitness"])
        if cur_best > best_fitness + EPS:
            best_fitness = cur_best
            stall_counter = 0
        else:
            stall_counter += 1

        if gen_idx < WARMUP:
            level = min(gen_idx + 1, 3)
        elif stall_counter > 0 and stall_counter % STALL_TRIGGER == 0:
            if level > 0:
                level -= 1
            else:
                level = 3

        mutation_ratio_eff = mut_levels[level]
        local_search_ratio_eff = ls_levels[level]
        crossover_sym_ratio_eff = c_levels[3 - level]

        k_mut = max(1, int(mutation_ratio_eff * target_size))
        k_ls = max(1, int(local_search_ratio_eff * target_size))

        offspring: list[dict[str, object]] = []
        for _offspring_idx in range(ga_offspring):
            parent_a = _tournament_select(population)
            parent_b = _tournament_select(population)
            parent_a_idx = np.asarray(parent_a["indices"], dtype=np.int64)
            parent_b_idx = np.asarray(parent_b["indices"], dtype=np.int64)

            inter = np.intersect1d(parent_a_idx, parent_b_idx, assume_unique=False)
            sym = np.setdiff1d(np.union1d(parent_a_idx, parent_b_idx), inter, assume_unique=False)
            child = inter.copy()

            max_from_sym = min(target_size, inter.size + int(np.floor(crossover_sym_ratio_eff * target_size)))
            if child.size < target_size and sym.size > 0:
                need_sym = max(0, max_from_sym - child.size)
                if need_sym > 0:
                    add_sym = _pick_top_static(sym, need_sym)
                    child = np.concatenate([child, add_sym])

            if child.size < target_size:
                complement_union = np.setdiff1d(
                    np.arange(num_samples, dtype=np.int64),
                    np.union1d(parent_a_idx, parent_b_idx),
                    assume_unique=False,
                )
                add_comp = _pick_top_static(complement_union, target_size - child.size)
                child = np.concatenate([child, add_comp])

            child = _repair_size(child)

            child_k_mut = min(k_mut, child.size, num_samples - child.size)
            if child_k_mut > 0:
                drop_idx = np.random.choice(child, size=child_k_mut, replace=False).astype(np.int64)
                comp_child = np.setdiff1d(np.arange(num_samples, dtype=np.int64), child, assume_unique=False)
                add_idx = np.random.choice(comp_child, size=child_k_mut, replace=False).astype(np.int64)
                child = np.setdiff1d(child, drop_idx, assume_unique=False)
                child = np.concatenate([child, add_idx])
                child = _repair_size(child)

            proxy_scores = np.asarray(
                parent_a["s_ref"] if float(parent_a["fitness"]) >= float(parent_b["fitness"]) else parent_b["s_ref"],
                dtype=np.float32,
            )
            child_k_ls = min(k_ls, child.size, num_samples - child.size)
            child = _local_search_step(child, proxy_scores, child_k_ls)
            child = _repair_size(child)

            offspring.append(_evaluate(child))

        merged = population + offspring
        merged_sorted = sorted(merged, key=lambda item: float(item["fitness"]), reverse=True)
        dedup_population: list[dict[str, object]] = []
        seen: set[tuple[int, ...]] = set()
        for item in merged_sorted:
            key = tuple(np.asarray(item["indices"], dtype=np.int64).tolist())
            if key in seen:
                continue
            seen.add(key)
            dedup_population.append(item)
            if len(dedup_population) >= ga_population_size:
                break
        population = dedup_population
        population.sort(key=lambda item: float(item["fitness"]), reverse=True)

        generation_best_item = max(population, key=lambda item: float(item["fitness"]))
        generation_best = float(generation_best_item["fitness"])
        score_history.append(float(generation_best_item["raw_score"]))
        correction_history.append(float(lambda_val * float(generation_best_item["penalty"])))
        generation_iter.set_postfix({"best_S_prime": f"{generation_best:.4f}"})

    # ====== GA 结束后的局部搜索收尾（基于真实动态得分） ======
    # 先取出最优个体的 indices 形式
    best_individual = max(population, key=lambda item: float(item["fitness"]))
    best_indices = np.asarray(best_individual["indices"], dtype=np.int64)
    best_mask = _indices_to_mask(best_indices)

    # 基于当前最优子集，计算真实的动态得分 s_ref，用作后续局部搜索的 "proxy_scores"
    best_S, best_s_ref, _ = _real_stats_cached(best_mask)
    best_penalty = compute_balance_penalty(best_mask, labels_np, num_classes, target_size)
    best_S_prime = best_S - lambda_val * best_penalty

    # 局部搜索的超参数：最多迭代步数、最小提升阈值
    REFINE_MAX_STEPS = 10
    REFINE_EPS = 1e-8

    # 参考 GA 最后一个 level 的 local search 档位，设置收尾用步长（避免改动过猛）
    k_ls_refine = min(
        max(1, int(ls_levels[level] * target_size)),
        best_indices.size,
        max(0, num_samples - best_indices.size),
    )

    refine_steps = REFINE_MAX_STEPS

    for _ in range(refine_steps):
        if k_ls_refine <= 0:
            break

        # 在当前最优解的真实动态得分 best_s_ref 上做一次局部搜索
        refined_indices = _local_search_step(best_indices, best_s_ref, k_ls_refine)
        refined_indices = _repair_size(refined_indices)
        refined_mask = _indices_to_mask(refined_indices)

        refined_S, refined_s_ref, _ = _real_stats_cached(refined_mask)
        refined_penalty = compute_balance_penalty(refined_mask, labels_np, num_classes, target_size)
        refined_S_prime = refined_S - lambda_val * refined_penalty

        # 只接受真正提高 S(D) 的 move；否则视为已经到达局部最优，提前停止
        if refined_S_prime > best_S_prime + REFINE_EPS:
            best_indices = refined_indices
            best_mask = refined_mask
            best_S = refined_S
            best_penalty = refined_penalty
            best_S_prime = refined_S_prime
            best_s_ref = refined_s_ref
        else:
            break

    # 用局部搜索后的 best_mask 作为最终解
    final_mask = best_mask.astype(np.uint8)
    _, final_s_all, _ = _real_stats_cached(final_mask)

    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        selected_by_class[class_id] = int(final_mask[class_indices].sum())

    final_rate = float(final_mask.mean())
    stats: dict[str, object] = {
        "sr": float(sr),
        "final_rate": final_rate,
        "selected_by_class": selected_by_class,
        "S": float(np.sum(final_s_all[final_mask.astype(bool)])),
        "penalty": float(compute_balance_penalty(final_mask, labels_np, num_classes, target_size)),
        "S_prime": float(best_S_prime),
        "best_iter": int(np.argmax(np.asarray(score_history, dtype=np.float32))),
        "score_history": score_history,
        "correction_history": correction_history,
        "ga_population_size": int(ga_population_size),
        "ga_generations": int(ga_generations),
        "ga_offspring": int(ga_offspring),
        "lambda": lambda_val,
        "lambda_info": lambda_info,
    }

    return final_mask, selected_by_class, stats


def _sanitize_for_filename(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def save_score_curve_plot(
    score_history: Sequence[float],
    correction_history: Sequence[float],
    *,
    dataset: str,
    method: str,
    weight_group: str,
    model_name: str,
    seed: int,
    cut_ratio: int,
    clip_model: str,
) -> Path:
    out_dir = PROJECT_ROOT / "mask_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_tag = _sanitize_for_filename(clip_model)
    out_path = (
        out_dir
        / (
            f"dataset_{dataset}_method_{method}_weight_{weight_group}_"
            f"model_{model_name}_seed_{seed}_cr_{cut_ratio}_clip_{clip_tag}_score_curve.png"
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    x = np.arange(len(score_history), dtype=np.int32)
    score_arr = np.asarray(score_history, dtype=np.float64)
    corr_arr = np.asarray(correction_history, dtype=np.float64)
    line1 = ax.plot(x, score_arr, linewidth=1.6, color="#1f77b4", label="S(D)")
    line2 = ax2.plot(x, corr_arr, linewidth=1.6, color="#ff7f0e", label="λ·penalty(D)")
    ax.set_title("Score & correction trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("S(D)", color="#1f77b4")
    ax2.set_ylabel("λ·penalty(D)", color="#ff7f0e")
    ax.grid(alpha=0.25)
    handles = line1 + line2
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    dataset_name = args.dataset.strip().lower()

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    method = args.method.strip().lower()
    if method not in {"topk", "group"}:
        raise ValueError(f"未知 method={method}，应为 {{'topk','group'}}")

    weight_group = args.weight_group.strip().lower()
    if weight_group not in {"naive", "learned"}:
        raise ValueError("weight-group 仅支持 {'naive', 'learned'}")

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)

    data_load_start = time.perf_counter()
    dataset_for_names = _build_dataset(dataset_name, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    print(
        f"[Init] {dataset_name} data ready | samples={len(dataset_for_names)} | elapsed={time.perf_counter() - data_load_start:.2f}s"
    )

    metric_init_start = time.perf_counter()
    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    print(
        f"[Init] Metrics ready (DDS/Div/SA) | elapsed={time.perf_counter() - metric_init_start:.2f}s"
    )

    batch_size = 128
    num_workers = 4

    loader_build_start = time.perf_counter()
    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        dataset_name,
        device,
        batch_size,
        num_workers,
    )
    print(
        f"[Init] DataLoaders ready (DDS/Div/SA) | elapsed={time.perf_counter() - loader_build_start:.2f}s"
    )

    method_name = f"{weight_group}_{method}"
    cut_ratios = parse_ratio_list(args.cr)
    if not cut_ratios:
        raise ValueError("cr 参数不能为空。")
    seeds = parse_seed_list(args.seeds)
    if not seeds:
        raise ValueError("seeds 参数不能为空。")

    total_tasks = len(seeds) * len(cut_ratios)
    task_idx = 0

    for seed in seeds:
        set_seed(seed)
        weights = load_scoring_weights(all_weights, weight_group, seed)
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()

        num_samples = len(dataset_for_names)

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
                "labels": np.asarray(dataset_for_names.targets),
            }

        static_compute_start = time.perf_counter()
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
            num_samples=num_samples,
            compute_fn=_compute_scores,
        )
        static_score_seconds = time.perf_counter() - static_compute_start
        print(
            f"[Seed {seed}] Static scores ready (cache/compute) | elapsed={static_score_seconds:.2f}s"
        )

        dds_scores_np = np.asarray(static_scores["dds"])
        div_scores_np = np.asarray(static_scores["div"])
        sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)

        if not (len(dds_scores_np) == len(div_scores_np) == len(sa_scores_np)):
            raise RuntimeError("三个指标的样本数不一致，无法合并。")

        total_scores_np = (
            weights["dds"] * dds_scores_np
            + weights["div"] * div_scores_np
            + weights["sa"] * sa_scores_np
        )
        labels = np.asarray(dataset_for_names.targets)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
        div_features_for_compare = None
        dds_features_for_compare = None

        def compute_subset_dynamic_sum(selected_mask: np.ndarray) -> float:
            nonlocal div_features_for_compare, dds_features_for_compare
            if div_features_for_compare is None:
                div_features_for_compare, _ = div_metric._encode_images(div_loader, image_adapter)
            if dds_features_for_compare is None:
                dds_features_for_compare, _ = dds_metric._encode_images(dds_loader, image_adapter)

            div_scores_dyn = np.asarray(
                div_metric.score_dataset_dynamic(
                    div_loader,
                    adapter=image_adapter,
                    selected_mask=selected_mask,
                    image_features=div_features_for_compare,
                    labels=labels_t,
                ).scores,
                dtype=np.float32,
            )
            dds_scores_dyn = np.asarray(
                dds_metric.score_dataset_dynamic(
                    dds_loader,
                    adapter=image_adapter,
                    selected_mask=selected_mask,
                    image_features=dds_features_for_compare,
                    labels=labels_t,
                ).scores,
                dtype=np.float32,
            )
            subset_scores = (
                weights["sa"] * sa_scores_np
                + weights["div"] * div_scores_dyn
                + weights["dds"] * dds_scores_dyn
            )
            return float(subset_scores[selected_mask.astype(bool)].sum())

        for cut_ratio in cut_ratios:
            task_idx += 1
            print(
                f"[Mask {task_idx}/{total_tasks}] seed={seed} | cr={cut_ratio} | method={method} | weight_group={weight_group}"
            )
            group_stats: dict[str, object] | None = None
            if method == "topk":
                mask, selected_by_class = select_topk_mask(
                    total_scores_np,
                    labels,
                    num_classes=len(class_names),
                    cut_ratio=cut_ratio,
                )
            else:
                mask, selected_by_class, group_stats = select_group_mask(
                    sa_scores_np,
                    dds_metric=dds_metric,
                    div_metric=div_metric,
                    dds_loader=dds_loader,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels,
                    weights=weights,
                    num_classes=len(class_names),
                    cut_ratio=cut_ratio,
                    device=device,
                    dataset_name=dataset_name,
                    seed=seed,
                    weight_group=weight_group,
                )
                debug_curve = save_score_curve_plot(
                    group_stats.get("score_history", []),
                    group_stats.get("correction_history", []),
                    dataset=dataset_name,
                    method=method,
                    weight_group=weight_group,
                    model_name=args.model_name,
                    seed=seed,
                    cut_ratio=cut_ratio,
                    clip_model=args.clip_model,
                )
                print(f"[Debug] score curve saved to: {debug_curve}")

                if args.compare:
                    topk_mask, _ = select_topk_mask(
                        total_scores_np,
                        labels,
                        num_classes=len(class_names),
                        cut_ratio=cut_ratio,
                    )
                    inter = int(np.logical_and(mask == 1, topk_mask == 1).sum())
                    sel = int(mask.sum())
                    overlap = inter / max(1, sel)
                    sum_group = compute_subset_dynamic_sum(mask)
                    sum_topk = compute_subset_dynamic_sum(topk_mask)
                    better = "group" if sum_group >= sum_topk else "topk"
                    diff = abs(sum_group - sum_topk)
                    print(
                        "[Compare] "
                        f"overlap={overlap:.4f} ({inter}/{sel}) | "
                        f"sum_group={sum_group:.4f} | sum_topk={sum_topk:.4f} | "
                        f"better={better} (Δ={diff:.4f})"
                    )

            total_time = time.perf_counter() - total_start
            mask_path = resolve_mask_path(
                mode=method_name,
                dataset=dataset_name,
                model=args.model_name,
                seed=seed,
                cut_ratio=cut_ratio,
            )
            mask_dir = mask_path.parent
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            print(
                f"seed={seed} | cr={cut_ratio} | selected={int(mask.sum())} "
                f"| static_score_seconds={static_score_seconds:.2f} | total_seconds={total_time:.2f}"
            )
            if group_stats is not None:
                print(
                    "group_stats: "
                    f"sr={group_stats['sr']:.6f} | rate={group_stats['final_rate']:.6f} | "
                    f"m_c={group_stats['selected_by_class']} | "
                    f"best_iter={group_stats['best_iter']} | "
                    f"S(D)={group_stats['S']:.6f} | pen(D)={group_stats['penalty']:.6f} | S'(D)={group_stats['S_prime']:.6f}"
                )
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
