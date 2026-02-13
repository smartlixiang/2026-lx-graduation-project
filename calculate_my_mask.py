from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import load_trained_adapters  # noqa: E402
from dataset.dataset_config import CIFAR10  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate selection masks (CIFAR-10)")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument(
        "--cr",
        type=str,
        default="80",
        help="cut_ratio 列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-image-path",
        type=str,
        default=None,
        help="图像 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument(
        "--adapter-text-path",
        type=str,
        default=None,
        help="文本 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument(
        "--adapter-seed",
        type=int,
        default=None,
        help="adapter 的随机种子（默认使用当前实验 seed）",
    )
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
        help="scoring_weights.json 中的权重组名",
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
    parser.add_argument("--max-outer-iters", type=int, default=150)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--topm", type=int, default=5000)
    parser.add_argument("--use-prob-select", action="store_true", default=True)
    parser.add_argument("--no-use-prob-select", dest="use_prob_select", action="store_false")
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=8.0)
    parser.add_argument("--alpha-mul", type=float, default=1.10)
    parser.add_argument("--lambda-cls", type=float, default=None)
    parser.add_argument("--gamma-cls", type=float, default=0.2)
    parser.add_argument("--candidate-pool-topm", type=int, default=5000, help="兼容旧参数，优先使用 --topm")
    return parser.parse_args()


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


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    group: str,
) -> dict[str, float]:
    selected = all_weights.get(group)
    if selected is None:
        raise KeyError(f"未找到权重组: {group}")
    if not isinstance(selected, dict):
        raise ValueError(f"权重组 {group} 格式不正确。")
    required = {"dds", "div", "sa"}
    missing = required - selected.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"权重组 {group} 缺少必要键: {missing_str}")
    weights: dict[str, float] = {}
    for key in sorted(required):
        value = selected[key]
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group} 的 {key} 无法转换为 float。") from exc
    return weights


def build_score_loader(
    preprocess, data_root: str, device: torch.device, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=preprocess)
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


def save_j_curve(
    j_history: list[float],
    dataset: str,
    cut_ratio: int,
    seed: int,
    weight_group: str,
    method: str,
    model_name: str,
    clip_model: str,
) -> Path | None:
    if not j_history:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] 无法绘制 J 曲线（matplotlib 不可用）: {exc}")
        return None

    debug_dir = PROJECT_ROOT / "mask_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    clip_tag = clip_model.replace("/", "-")
    file_name = (
        f"dataset_{dataset}_method_{method}_weight_{weight_group}_"
        f"model_{model_name}_seed_{seed}_cr_{cut_ratio}_clip_{clip_tag}_J_curve.png"
    )
    save_path = debug_dir / file_name

    xs = np.arange(1, len(j_history) + 1)
    ys = np.asarray(j_history, dtype=np.float64)
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.title(f"J vs Outer Iteration ({dataset}, cr={cut_ratio}, seed={seed})")
    plt.xlabel("Outer iteration")
    plt.ylabel("J(D)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path


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
    max_outer_iters: int = 80,
    rho: float = 0.10,
    topM: int = 5000,
    use_prob_select: bool = True,
    alpha0: float = 1.0,
    alpha_max: float = 8.0,
    alpha_mul: float = 1.10,
    lambda_cls: float | None = None,
    gamma_cls: float = 0.2,
) -> tuple[np.ndarray, dict[int, int], dict[str, object]]:
    if cut_ratio <= 0 or cut_ratio > 100:
        raise ValueError("cr 必须在 1-100 之间。")

    num_samples = sa_scores.shape[0]
    if labels.shape[0] != num_samples:
        raise ValueError("sa_scores 与 labels 的样本数不一致。")

    sa_scores_np = np.asarray(sa_scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    m = int(np.round((cut_ratio / 100.0) * num_samples))
    m = min(max(m, 1), num_samples)

    class_counts_total = np.bincount(labels_np, minlength=num_classes)
    if class_counts_total.sum() == num_samples and num_samples > 0:
        p_c = class_counts_total.astype(np.float64) / float(num_samples)
    else:
        p_c = np.full(num_classes, 1.0 / float(num_classes), dtype=np.float64)
    lambda_cls_val = float(lambda_cls) if lambda_cls is not None else float(gamma_cls * (m / float(num_classes)))

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    dds_features, _ = dds_metric._encode_images(dds_loader, image_adapter)
    all_indices = np.arange(num_samples)

    def _omega(cur_counts: np.ndarray) -> float:
        q = cur_counts.astype(np.float64) / float(m)
        return float(np.sum((q - p_c) ** 2))

    def _real_j(cur_mask: np.ndarray) -> tuple[float, float, np.ndarray]:
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
        s_ref = weights["sa"] * sa_scores_np + weights["div"] * div_scores + weights["dds"] * dds_scores
        counts = np.bincount(labels_np[cur_mask.astype(bool)], minlength=num_classes).astype(np.int64)
        omega_val = _omega(counts)
        j_val = float(np.sum(s_ref[cur_mask.astype(bool)])) - lambda_cls_val * omega_val
        return j_val, omega_val, counts

    def _delta_remove_by_class(cur_counts: np.ndarray) -> np.ndarray:
        before = (cur_counts.astype(np.float64) / float(m) - p_c) ** 2
        after_counts = np.maximum(cur_counts - 1, 0)
        after = (after_counts.astype(np.float64) / float(m) - p_c) ** 2
        return after - before

    def _delta_add_by_class(cur_counts: np.ndarray) -> np.ndarray:
        before = (cur_counts.astype(np.float64) / float(m) - p_c) ** 2
        after = ((cur_counts + 1).astype(np.float64) / float(m) - p_c) ** 2
        return after - before

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    init_idx = np.random.choice(num_samples, size=m, replace=False)
    selected_mask[init_idx] = 1

    j_cur, omega_cur, m_c = _real_j(selected_mask)
    best_mask = selected_mask.copy()
    best_j = float(j_cur)
    best_omega = float(omega_cur)
    outer_j_history = [float(j_cur)]

    outer_iterator = tqdm(
        range(max_outer_iters),
        desc=progress_desc or "IRGR optimization",
        unit="outer",
        leave=True,
    )

    for t in outer_iterator:
        alpha_t = min(alpha_max, alpha0 * (alpha_mul ** t))

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
        s_all = weights["sa"] * sa_scores_np + weights["div"] * div_scores + weights["dds"] * dds_scores

        k = int(rho * m)
        if rho > 0 and k == 0:
            k = 1
        k = min(max(0, k), m)

        if k > 0:
            selected_idx = all_indices[selected_mask == 1]
            delta_remove = _delta_remove_by_class(m_c)
            remove_gain = -s_all[selected_idx] - lambda_cls_val * delta_remove[labels_np[selected_idx]]
            rem_pos = np.argpartition(-remove_gain, k - 1)[:k]
            remove_idx = selected_idx[rem_pos]

            tmp_mask = selected_mask.copy()
            tmp_mask[remove_idx] = 0
            m_c_tmp = m_c.copy()
            rem_counts = np.bincount(labels_np[remove_idx], minlength=num_classes)
            m_c_tmp = m_c_tmp - rem_counts

            candidate_idx = all_indices[tmp_mask == 0]
            delta_add = _delta_add_by_class(m_c_tmp)
            add_gain = s_all[candidate_idx] - lambda_cls_val * delta_add[labels_np[candidate_idx]]

            pool_k = min(max(1, topM), candidate_idx.size)
            if pool_k < candidate_idx.size:
                top_pos = np.argpartition(-add_gain, pool_k - 1)[:pool_k]
            else:
                top_pos = np.arange(candidate_idx.size)
            pool_idx = candidate_idx[top_pos]
            pool_gain = add_gain[top_pos]

            add_cnt = min(k, pool_idx.size)
            if add_cnt > 0:
                if use_prob_select and pool_idx.size > 1:
                    logits = np.clip(alpha_t * (pool_gain - np.max(pool_gain)), -50.0, 50.0)
                    probs = np.exp(logits)
                    probs = probs / probs.sum()
                    chosen_add = np.random.choice(pool_idx, size=add_cnt, replace=False, p=probs)
                else:
                    choose_pos = np.argpartition(-pool_gain, add_cnt - 1)[:add_cnt]
                    chosen_add = pool_idx[choose_pos]
                tmp_mask[chosen_add] = 1

            selected_mask_new = tmp_mask
        else:
            selected_mask_new = selected_mask.copy()

        j_new, omega_new, m_c_new = _real_j(selected_mask_new)

        j_old = j_cur
        best_old = best_j
        j_cur = float(j_new)
        omega_cur = float(omega_new)
        selected_mask = selected_mask_new
        m_c = m_c_new
        outer_j_history.append(j_cur)

        if j_cur > best_j:
            best_j = float(j_cur)
            best_mask = selected_mask.copy()
            best_omega = float(omega_cur)

        max_dev = float(np.max(np.abs(m_c / float(m) - p_c)))
        tqdm.write(
            f"[Group/IRGR] t={t + 1}/{max_outer_iters} ΔJ_cur={j_cur - j_old:.6f} Δbest={best_j - best_old:.6f} "
            f"Omega={omega_cur:.6f} max_dev={max_dev:.6f}"
        )
        outer_iterator.set_postfix(
            t=f"{t + 1}/{max_outer_iters}",
            J_cur=f"{j_cur:.3f}",
            best_J=f"{best_j:.3f}",
            Omega=f"{omega_cur:.6f}",
            alpha=f"{alpha_t:.3f}",
            rho=f"{rho:.3f}",
        )

    final_mask = best_mask.astype(np.uint8)
    final_j, final_omega, final_counts = _real_j(final_mask)

    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        selected_by_class[class_id] = int(final_mask[class_indices].sum())

    # recompute S via final real scores
    final_div = np.asarray(
        div_metric.score_dataset_dynamic(
            div_loader,
            adapter=image_adapter,
            selected_mask=final_mask,
            image_features=div_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    final_dds = np.asarray(
        dds_metric.score_dataset_dynamic(
            dds_loader,
            adapter=image_adapter,
            selected_mask=final_mask,
            image_features=dds_features,
            labels=labels_t,
        ).scores,
        dtype=np.float32,
    )
    final_s_ref = weights["sa"] * sa_scores_np + weights["div"] * final_div + weights["dds"] * final_dds
    final_s = float(np.sum(final_s_ref[final_mask.astype(bool)]))

    stats: dict[str, object] = {
        "m": int(m),
        "lambda_cls": float(lambda_cls_val),
        "gamma_cls": float(gamma_cls),
        "rho": float(rho),
        "m_c": {int(c): int(v) for c, v in enumerate(final_counts.tolist())},
        "Omega": float(final_omega),
        "S": float(final_s),
        "J": float(final_j),
        "best_Omega": float(best_omega),
        "best_J": float(best_j),
        "outer_j_history": outer_j_history,
        "max_outer_iters": int(max_outer_iters),
        "use_prob_select": bool(use_prob_select),
        "topM": int(topM),
    }

    return final_mask, selected_by_class, stats


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, CIFAR10)
    weights = load_scoring_weights(all_weights, args.weight_group)

    data_load_start = time.perf_counter()
    dataset_for_names = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=None
    )
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    print(
        f"[Init] CIFAR-10 data ready | samples={len(dataset_for_names)} | elapsed={time.perf_counter() - data_load_start:.2f}s"
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
        dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    print(
        f"[Init] DataLoaders ready (DDS/Div/SA) | elapsed={time.perf_counter() - loader_build_start:.2f}s"
    )

    method = args.method.strip().lower()
    if method not in {"topk", "group"}:
        raise ValueError(f"未知 method={method}，应为 {{'topk','group'}}")
    method_name = f"{args.weight_group}_{method}"
    cut_ratios = parse_ratio_list(args.cr)
    if not cut_ratios:
        raise ValueError("cr 参数不能为空。")
    seeds = parse_seed_list(args.seeds)
    if args.weight_group == "naive":
        save_seeds = [CONFIG.global_seed]
    else:
        save_seeds = seeds
    total_tasks = len(save_seeds) * len(cut_ratios)
    task_idx = 0
    for seed in save_seeds:
        set_seed(seed)
        adapter_seed = args.adapter_seed if args.adapter_seed is not None else seed
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=CIFAR10,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=adapter_seed,
            map_location=device,
            adapter_image_path=args.adapter_image_path,
            adapter_text_path=args.adapter_text_path,
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
            dataset=CIFAR10,
            seed=adapter_seed,
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

        for cut_ratio in cut_ratios:
            task_idx += 1
            print(
                f"[Mask {task_idx}/{total_tasks}] seed={seed} | cr={cut_ratio} | method={method}"
            )
            group_stats: dict[str, object] | None = None
            if method == "topk":
                mask, selected_by_class = select_topk_mask(
                    total_scores_np,
                    labels,
                    num_classes=len(class_names),
                    cut_ratio=cut_ratio,
                )
                selection_strategy = "topk_per_class"
            elif method == "group":
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
                    max_outer_iters=args.max_outer_iters,
                    rho=args.rho,
                    topM=args.topm if args.topm is not None else args.candidate_pool_topm,
                    use_prob_select=args.use_prob_select,
                    alpha0=args.alpha0,
                    alpha_max=args.alpha_max,
                    alpha_mul=args.alpha_mul,
                    lambda_cls=args.lambda_cls,
                    gamma_cls=args.gamma_cls,
                )
                selection_strategy = "group_selection"
            total_time = time.perf_counter() - total_start
            mask_path = resolve_mask_path(
                mode=method_name,
                dataset="cifar10",
                model=args.model_name,
                seed=seed,
                cut_ratio=cut_ratio,
            )
            mask_dir = mask_path.parent
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            meta_info = {
                "dataset": "cifar10",
                "model_name": args.model_name,
                "method": method_name,
                "weight_group": args.weight_group,
                "clip_model": args.clip_model,
                "adapter_seed": adapter_seed,
                "adapter_image_path": str(adapter_paths["image_path"]),
                "adapter_text_path": str(adapter_paths["text_path"]),
                "cr": cut_ratio,
                "num_samples": int(mask.shape[0]),
                "selected_count": int(mask.sum()),
                "selected_by_class": selected_by_class,
                "selection_strategy": selection_strategy,
                "seeds": save_seeds,
                "group_stats": group_stats,
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
                    f"m={group_stats['m']} | lambda_cls={group_stats['lambda_cls']:.6f} | "
                    f"m_c={group_stats['m_c']} | Omega(D)={group_stats['Omega']:.6f} | "
                    f"S(D)={group_stats['S']:.6f} | J(D)={group_stats['J']:.6f}"
                )
                j_curve_path = save_j_curve(
                    j_history=list(group_stats.get("outer_j_history", [])),
                    dataset="cifar10",
                    cut_ratio=cut_ratio,
                    seed=seed,
                    weight_group=args.weight_group,
                    method=method,
                    model_name=args.model_name,
                    clip_model=args.clip_model,
                )
                if j_curve_path is not None:
                    print(f"J-curve saved to: {j_curve_path}")
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
