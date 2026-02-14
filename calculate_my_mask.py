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
    parser.add_argument("--debug-outer", type=int, default=40)
    parser.add_argument("--yangclip-steps", type=int, default=20000)
    parser.add_argument("--yangclip-lr", type=float, default=0.1)
    parser.add_argument("--yangclip-beta", type=float, default=1.0)
    parser.add_argument("--yangclip-theta", type=float, default=1e-3)
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
    debug_outer: int = 3,
    yangclip_steps: int = 300,
    yangclip_lr: float = 0.1,
    yangclip_beta: float = 50.0,
    yangclip_theta: float = 5e-4,
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

    def _real_stats(cur_mask: np.ndarray) -> tuple[float, float, np.ndarray]:
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
        return s_val, s_val, counts

    def _yangclip_sgd_select(s_all: np.ndarray, outer_idx: int) -> tuple[np.ndarray, dict[str, float]]:
        s_all_t = torch.as_tensor(s_all, dtype=torch.float32, device=device)
        d = torch.nn.Parameter(torch.ones(num_samples, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([d], lr=yangclip_lr)

        final_loss = 0.0
        final_lsa = 0.0
        final_ls = 0.0
        final_ls_eff = 0.0
        final_rate_mean = 0.0
        final_gap_mean = 0.0
        final_gap_max = 0.0

        for _ in range(yangclip_steps):
            optimizer.zero_grad(set_to_none=True)
            z = torch.sigmoid(d)
            b = (z > 0.5).float()
            b_ste = b + z - z.detach()
            class_sums = torch.zeros(num_classes, dtype=torch.float32, device=device)
            class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
            class_sums.index_add_(0, labels_t, b_ste)
            class_counts.index_add_(0, labels_t, torch.ones_like(b_ste))
            rate_c = class_sums / class_counts.clamp_min(1.0)

            lsa = -torch.mean(z * s_all_t)
            ls_c = torch.sqrt((rate_c - sr) ** 2 + 1e-12)
            ls = torch.mean(ls_c)
            ls_eff = torch.relu(ls - yangclip_theta)
            loss = lsa + yangclip_beta * ls_eff
            loss.backward()
            optimizer.step()

            final_loss = float(loss.item())
            final_lsa = float(lsa.item())
            final_ls = float(ls.item())
            final_ls_eff = float(ls_eff.item())
            gap_c = torch.abs(rate_c - sr)
            final_rate_mean = float(rate_c.mean().item())
            final_gap_mean = float(gap_c.mean().item())
            final_gap_max = float(gap_c.max().item())

        with torch.no_grad():
            z_final = torch.sigmoid(d)
            b_final = (z_final > 0.5).to(torch.uint8)
            b_final_f = b_final.float()
            hard_class_sums = torch.zeros(num_classes, dtype=torch.float32, device=device)
            hard_class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
            hard_class_sums.index_add_(0, labels_t, b_final_f)
            hard_class_counts.index_add_(0, labels_t, torch.ones_like(b_final_f))
            hard_rate_c = hard_class_sums / hard_class_counts.clamp_min(1.0)
            hard_gap_c = torch.abs(hard_rate_c - sr)
            rate_hard = float(hard_rate_c.mean().item())
            hard_gap = float(hard_gap_c.max().item())
            d_mean = float(d.mean().item())
            d_std = float(d.std().item())
            d_max = float(torch.max(torch.abs(d)).item())
            z_mean = float(z_final.mean().item())
            z_std = float(z_final.std().item())

        tqdm.write(
            f"[SGD-final][outer {outer_idx + 1}] loss={final_loss:.6f} "
            f"Lsa={final_lsa:.6f} Ls={final_ls:.6f} Ls_eff={final_ls_eff:.6f} "
            f"beta={yangclip_beta:.3f} sr={sr:.6f} "
            f"rate_mean={final_rate_mean:.6f} gap_mean={final_gap_mean:.6f} gap_max={final_gap_max:.6f} "
            f"d_mean/d_std/max|d|={d_mean:.6f}/{d_std:.6f}/{d_max:.6f} "
            f"sigmoid(d)_mean/std={z_mean:.6f}/{z_std:.6f}"
        )

        if hard_gap > 5e-4:
            tqdm.write(
                f"[Warn] hard ratio gap={hard_gap:.6f} > 0.0005 after binarization; "
                "consider increasing --yangclip-beta or --yangclip-steps."
            )

        return b_final.detach().cpu().numpy().astype(np.uint8), {
            "loss": final_loss,
            "Lsa": final_lsa,
            "Ls": final_ls,
            "Ls_eff": final_ls_eff,
            "rate": final_rate_mean,
            "gap": final_gap_mean,
            "hard_rate": rate_hard,
            "hard_gap": hard_gap,
        }

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    init_count = max(1, int(round(sr * num_samples)))
    init_idx = np.random.choice(num_samples, size=init_count, replace=False)
    selected_mask[init_idx] = 1

    j_cur, s_cur, _ = _real_stats(selected_mask)
    outer_j_history = [float(j_cur)]
    outer_iters = max(1, int(debug_outer))
    last_sgd_info: dict[str, float] = {}

    outer_iterator = tqdm(
        range(outer_iters),
        desc=progress_desc or "Group optimization",
        unit="outer",
        leave=True,
    )

    for t in outer_iterator:
        j_old = float(j_cur)

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

        selected_mask, last_sgd_info = _yangclip_sgd_select(s_all, outer_idx=t)
        j_cur, s_cur, _ = _real_stats(selected_mask)
        outer_j_history.append(float(j_cur))

        rate = float(selected_mask.mean())
        gap = abs(rate - sr)
        delta_j = j_cur - j_old
        tqdm.write(
            f"[outer {t + 1}] J_old={j_old:.6f} J_new={j_cur:.6f} ΔJ={delta_j:.6f} "
            f"rate={rate:.6f} gap={gap:.6f}"
        )
        outer_iterator.set_postfix(J_cur=f"{j_cur:.3f}", gap=f"{gap:.6f}")

    final_mask = selected_mask.astype(np.uint8)
    final_j, final_s, final_counts = _real_stats(final_mask)

    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        selected_by_class[class_id] = int(final_mask[class_indices].sum())

    stats: dict[str, object] = {
        "target_sr": float(sr),
        "m_c": {int(c): int(v) for c, v in enumerate(final_counts.tolist())},
        "S": float(final_s),
        "J": float(final_j),
        "final_rate": float(final_mask.mean()),
        "final_gap": float(abs(float(final_mask.mean()) - sr)),
        "outer_j_history": outer_j_history,
        "outer_iters": int(outer_iters),
        "debug_outer": int(debug_outer),
        "yangclip_steps": int(yangclip_steps),
        "yangclip_lr": float(yangclip_lr),
        "yangclip_beta": float(yangclip_beta),
        "yangclip_theta": float(yangclip_theta),
        "last_sgd": last_sgd_info,
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
                    debug_outer=args.debug_outer,
                    yangclip_steps=args.yangclip_steps,
                    yangclip_lr=args.yangclip_lr,
                    yangclip_beta=args.yangclip_beta,
                    yangclip_theta=args.yangclip_theta,
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
                    f"sr={group_stats['target_sr']:.6f} | rate={group_stats['final_rate']:.6f} | "
                    f"gap={group_stats['final_gap']:.6f} | m_c={group_stats['m_c']} | "
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
