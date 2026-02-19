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
    parser.add_argument("--debug-outer", type=int, default=40)
    parser.add_argument(
        "--branch",
        type=int,
        default=3,
        help="group 模式随机初始化分支数，最终取最优分支",
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    progress_desc: str | None = None,
    debug_outer: int = 3,
    branch_count: int = 1,
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

    rho_min = 0.01
    rho_max = 0.5

    def _real_stats(cur_mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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
        return s_val, s_ref, counts

    branch_total = max(1, int(branch_count))
    outer_iters = max(1, int(debug_outer))
    branch_histories: list[list[float]] = []
    branch_best_scores: list[float] = []
    branch_best_iters: list[int] = []

    global_best_s = float("-inf")
    global_best_mask = np.zeros(num_samples, dtype=np.uint8)
    global_best_branch = 0
    global_best_iter = 0

    for branch_idx in range(branch_total):
        selected_mask = np.zeros(num_samples, dtype=np.uint8)
        init_count = target_size
        init_idx = np.random.choice(num_samples, size=init_count, replace=False)
        selected_mask[init_idx] = 1
        rho = float(rho_max)

        branch_best_mask = selected_mask.copy()
        branch_best_s, _, _ = _real_stats(branch_best_mask)
        branch_best_iter = 0
        branch_scores = [float(branch_best_s)]

        outer_iterator = tqdm(
            range(outer_iters),
            desc=(progress_desc or "Group optimization") + f" [b{branch_idx + 1}/{branch_total}]",
            unit="outer",
            leave=True,
        )

        for outer_idx in outer_iterator:
            div_scores = np.asarray(
                div_metric.score_dataset_dynamic(
                    div_loader,
                    adapter=image_adapter,
                    selected_mask=selected_mask,
                    image_features=div_features,
                    labels=labels_t,
                ).scores,
                dtype=np.float32,
            )
            dds_scores = np.asarray(
                dds_metric.score_dataset_dynamic(
                    dds_loader,
                    adapter=image_adapter,
                    selected_mask=selected_mask,
                    image_features=dds_features,
                    labels=labels_t,
                ).scores,
                dtype=np.float32,
            )
            s_all = (
                weights["sa"] * sa_scores_np
                + weights["div"] * div_scores
                + weights["dds"] * dds_scores
            )

            selected_idx = np.flatnonzero(selected_mask == 1)
            unselected_idx = np.flatnonzero(selected_mask == 0)
            k_replace = max(1, int(round(rho * target_size)))
            k_replace = min(k_replace, selected_idx.size, unselected_idx.size)

            if k_replace > 0:
                drop_order = np.argsort(s_all[selected_idx])
                idx_drop = selected_idx[drop_order[:k_replace]]

                add_order = np.argsort(-s_all[unselected_idx])
                idx_add = unselected_idx[add_order[:k_replace]]

                candidate_mask = selected_mask.copy()
                candidate_mask[idx_drop] = 0
                candidate_mask[idx_add] = 1
            else:
                candidate_mask = selected_mask.copy()

            candidate_idx = np.flatnonzero(candidate_mask == 1)
            if candidate_idx.size < target_size:
                candidate_unselected = np.flatnonzero(candidate_mask == 0)
                if candidate_unselected.size > 0:
                    need = min(target_size - candidate_idx.size, candidate_unselected.size)
                    fill_order = np.argsort(-s_all[candidate_unselected])
                    fill_idx = candidate_unselected[fill_order[:need]]
                    candidate_mask[fill_idx] = 1
                    candidate_idx = np.flatnonzero(candidate_mask == 1)
            if candidate_idx.size > target_size:
                keep_order = np.argsort(-s_all[candidate_idx])
                keep_idx = candidate_idx[keep_order[:target_size]]
                candidate_mask = np.zeros(num_samples, dtype=np.uint8)
                candidate_mask[keep_idx] = 1

            candidate_s, _, _ = _real_stats(candidate_mask)
            current_s = branch_scores[-1]
            if outer_idx < 2:
                accept = candidate_s >= current_s
            else:
                base = min(branch_scores[-1], branch_scores[-2])
                accept = candidate_s >= base

            if accept:
                selected_mask = candidate_mask
                accepted_s = float(candidate_s)
                rho = min(rho_max, rho * 1.1)
            else:
                accepted_s = float(current_s)
                rho = max(rho_min, rho * 0.5)

            branch_scores.append(accepted_s)
            if accepted_s > branch_best_s:
                branch_best_s = float(accepted_s)
                branch_best_mask = selected_mask.copy()
                branch_best_iter = outer_idx + 1

            if branch_best_s > global_best_s:
                global_best_s = float(branch_best_s)
                global_best_mask = branch_best_mask.copy()
                global_best_branch = branch_idx
                global_best_iter = branch_best_iter

            outer_iterator.set_postfix(
                {
                    "branch_best_S": f"{branch_best_s:.4f}",
                    "branch_best_iter": branch_best_iter,
                    "global_best_S": f"{global_best_s:.4f}",
                    "global_best_branch": global_best_branch + 1,
                }
            )

        branch_histories.append(branch_scores)
        branch_best_scores.append(float(branch_best_s))
        branch_best_iters.append(int(branch_best_iter))

    final_mask = global_best_mask.astype(np.uint8)
    _, final_s_all, _ = _real_stats(final_mask)

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
        "outer_iters": int(outer_iters),
        "branch_count": int(branch_total),
        "best_branch": int(global_best_branch + 1),
        "best_iter": int(global_best_iter),
        "branch_best_scores": branch_best_scores,
        "branch_best_iters": branch_best_iters,
        "branch_score_histories": branch_histories,
    }

    return final_mask, selected_by_class, stats


def _sanitize_for_filename(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def save_branch_score_plot(
    branch_score_histories: Sequence[Sequence[float]],
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
            f"model_{model_name}_seed_{seed}_cr_{cut_ratio}_clip_{clip_tag}_branch_curve.png"
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for history in branch_score_histories:
        if len(history) == 0:
            continue
        x = np.arange(len(history), dtype=np.int32)
        ax.plot(x, np.asarray(history, dtype=np.float64), linewidth=1.6)
    ax.set_title("Score trajectory by branch")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("S(D)")
    ax.grid(alpha=0.25)
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

        dds_start = time.perf_counter()
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
        print(
            f"[Seed {seed}] Static scores ready (cache/compute) | elapsed={time.perf_counter() - static_compute_start:.2f}s"
        )

        dds_scores = torch.from_numpy(static_scores["dds"])
        div_scores = torch.from_numpy(static_scores["div"])
        sa_scores = torch.from_numpy(static_scores["sa"])
        dds_time = time.perf_counter() - dds_start
        div_time = dds_time
        sa_time = dds_time

        if not (len(dds_scores) == len(div_scores) == len(sa_scores)):
            raise RuntimeError("三个指标的样本数不一致，无法合并。")

        total_scores = (
            weights["dds"] * dds_scores
            + weights["div"] * div_scores
            + weights["sa"] * sa_scores
        )
        labels = np.asarray(dataset_for_names.targets)
        total_scores_np = np.asarray(total_scores)
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
        sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
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
            debug_curve_path: str | None = None
            if method == "topk":
                mask, selected_by_class = select_topk_mask(
                    total_scores_np,
                    labels,
                    num_classes=len(class_names),
                    cut_ratio=cut_ratio,
                )
                selection_strategy = "topk_per_class"
            else:
                mask, selected_by_class, group_stats = select_group_mask(
                    np.asarray(sa_scores),
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
                    progress_desc=(
                        f"Group mask optimization (seed={seed}, cr={cut_ratio})"
                    ),
                    debug_outer=args.debug_outer,
                    branch_count=args.branch,
                )
                debug_curve = save_branch_score_plot(
                    group_stats.get("branch_score_histories", []),
                    dataset=dataset_name,
                    method=method,
                    weight_group=weight_group,
                    model_name=args.model_name,
                    seed=seed,
                    cut_ratio=cut_ratio,
                    clip_model=args.clip_model,
                )
                debug_curve_path = str(debug_curve)
                print(f"[Debug] branch score curves saved to: {debug_curve}")

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
                selection_strategy = "group_selection"

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

            meta_info = {
                "dataset": dataset_name,
                "model_name": args.model_name,
                "method": method_name,
                "weight_group": weight_group,
                "weight_seed": seed if weight_group == "learned" else None,
                "clip_model": args.clip_model,
                "adapter_seed": seed,
                "adapter_image_path": str(adapter_paths["image_path"]),
                "adapter_text_path": str(adapter_paths["text_path"]),
                "cr": cut_ratio,
                "num_samples": int(mask.shape[0]),
                "selected_count": int(mask.sum()),
                "selected_by_class": selected_by_class,
                "selection_strategy": selection_strategy,
                "seed": seed,
                "group_stats": group_stats,
                "mask_debug_curve": debug_curve_path,
                "timing": {
                    "dds_seconds": dds_time,
                    "div_seconds": div_time,
                    "sa_seconds": sa_time,
                    "total_seconds": total_time,
                },
            }
            with open(mask_dir / "meta_info.json", "w", encoding="utf-8") as f:
                json.dump(meta_info, f, ensure_ascii=False, indent=2)

            print(f"seed={seed} | cr={cut_ratio} | selected={int(mask.sum())}")
            if group_stats is not None:
                print(
                    "group_stats: "
                    f"sr={group_stats['sr']:.6f} | rate={group_stats['final_rate']:.6f} | "
                    f"m_c={group_stats['selected_by_class']} | "
                    f"best_branch={group_stats['best_branch']} | "
                    f"best_iter={group_stats['best_iter']} | "
                    f"S(D)={group_stats['S']:.6f}"
                )
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
