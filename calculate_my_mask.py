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
    parser.add_argument("--max-outer-iters", type=int, default=30)
    parser.add_argument("--sgd-steps", type=int, default=200)
    parser.add_argument("--sgd-lr", type=float, default=0.1)
    parser.add_argument("--sgd-opt", type=str, default="adam", choices=["adam"])
    parser.add_argument("--eps-stop-abs", type=float, default=1.0)
    parser.add_argument("--eps-stop-rel", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--debug-outer", type=int, default=3)
    parser.add_argument("--sgd-log-every", type=int, default=10)
    parser.add_argument("--sgd-grad-clip", type=float, default=0.0)
    parser.add_argument("--sgd-early-stop", action="store_true", default=True)
    parser.add_argument("--no-sgd-early-stop", dest="sgd_early_stop", action="store_false")
    parser.add_argument("--sgd-early-patience", type=int, default=10)
    parser.add_argument("--sgd-min-delta", type=float, default=1e-3)
    parser.add_argument("--lambda-cls", type=float, default=None)
    parser.add_argument("--gamma-cls", type=float, default=1)
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
    max_outer_iters: int = 30,
    sgd_steps: int = 200,
    sgd_lr: float = 0.1,
    sgd_opt: str = "adam",
    eps_stop_abs: float = 1.0,
    eps_stop_rel: float = 1e-4,
    patience: int = 3,
    debug_outer: int = 3,
    sgd_log_every: int = 10,
    sgd_grad_clip: float = 0.0,
    sgd_early_stop: bool = True,
    sgd_early_patience: int = 10,
    sgd_min_delta: float = 1e-3,
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
    target_counts = m * p_c
    lambda_cls_val = (
        float(lambda_cls)
        if lambda_cls is not None
        else float(gamma_cls * (m / float(num_classes)))
    )

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    dds_features, _ = dds_metric._encode_images(dds_loader, image_adapter)

    def _omega(cur_counts: np.ndarray) -> float:
        q = cur_counts.astype(np.float64) / float(m)
        return float(np.sum((q - p_c) ** 2))

    def _real_stats(cur_mask: np.ndarray) -> tuple[float, float, float, np.ndarray]:
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
        counts = np.bincount(labels_np[cur_mask.astype(bool)], minlength=num_classes).astype(
            np.int64
        )
        s_val = float(np.sum(s_ref[cur_mask.astype(bool)]))
        omega_val = _omega(counts)
        j_val = s_val - lambda_cls_val * omega_val
        return j_val, s_val, omega_val, counts

    class_index_tensors = {
        class_id: torch.as_tensor(
            np.flatnonzero(labels_np == class_id),
            dtype=torch.long,
            device=device,
        )
        for class_id in range(num_classes)
    }
    p_c_t = torch.as_tensor(p_c, dtype=torch.float32, device=device)
    eps_logit = 1e-6
    log_every = max(1, int(sgd_log_every))

    if sgd_opt != "adam":
        raise ValueError(f"不支持的 sgd_opt={sgd_opt}，当前仅支持 adam。")

    def _mask_stats(mask: np.ndarray) -> tuple[float, int, int, float]:
        counts = np.bincount(labels_np[mask.astype(bool)], minlength=num_classes).astype(np.float64)
        max_dev = float(np.max(np.abs(counts / float(m) - p_c)))
        return max_dev, int(counts.min()), int(counts.max()), float(counts.std())

    def _proxy_from_z(z: np.ndarray, s_all: np.ndarray) -> tuple[float, float, float, np.ndarray, float]:
        top_idx = np.argpartition(-z, m - 1)[:m]
        mask_tmp = np.zeros(num_samples, dtype=np.uint8)
        mask_tmp[top_idx] = 1
        counts = np.bincount(labels_np[mask_tmp.astype(bool)], minlength=num_classes).astype(np.int64)
        proxy_s = float(np.sum(s_all[mask_tmp.astype(bool)]))
        proxy_omega = _omega(counts)
        proxy_j = proxy_s - lambda_cls_val * proxy_omega
        z_threshold = float(np.partition(z, -m)[-m])
        return proxy_j, proxy_s, proxy_omega, mask_tmp, z_threshold

    def _save_outer_z_hist(outer_idx: int, z_values: np.ndarray) -> str:
        debug_dir = PROJECT_ROOT / "mask_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        tag = (progress_desc or "group").replace(" ", "_").replace("/", "-")
        save_path = debug_dir / f"{tag}_outer_{outer_idx + 1:03d}_z_hist.npz"
        hist, bin_edges = np.histogram(z_values, bins=20, range=(0.0, 1.0))
        np.savez_compressed(save_path, z=z_values.astype(np.float32), hist=hist, bin_edges=bin_edges)
        return str(save_path)

    def _optimize_z_from_scores(s_all: np.ndarray, outer_idx: int) -> tuple[np.ndarray, dict[str, float]]:
        s_all_t = torch.as_tensor(s_all, dtype=torch.float32, device=device)
        s_init = np.clip(s_all.astype(np.float64), eps_logit, 1.0 - eps_logit)
        d = torch.tensor(
            np.log(s_init / (1.0 - s_init)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([d], lr=sgd_lr)

        best_obj = -float("inf")
        best_proxy_j = -float("inf")
        no_improve = 0
        init_obj: float | None = None
        init_data_term: float | None = None
        init_cls_term: float | None = None
        init_proxy_j: float | None = None

        last_z = torch.sigmoid(d).detach().cpu().numpy().astype(np.float32)
        last_proxy_j, _, _, _, _ = _proxy_from_z(last_z, s_all)

        for step in range(1, sgd_steps + 1):
            optimizer.zero_grad()
            z = torch.sigmoid(d)
            z_sums = []
            for class_id in range(num_classes):
                cls_idx = class_index_tensors[class_id]
                if cls_idx.numel() == 0:
                    z_sums.append(torch.zeros((), dtype=torch.float32, device=device))
                else:
                    z_sums.append(z.index_select(0, cls_idx).sum())
            z_sum_c = torch.stack(z_sums)
            cls_term_t = torch.sum((z_sum_c / float(m) - p_c_t) ** 2)
            data_term_t = torch.dot(z, s_all_t)
            obj_t = data_term_t - float(lambda_cls_val) * cls_term_t
            loss = -obj_t
            loss.backward()
            grad_norm = float(torch.linalg.vector_norm(d.grad).item()) if d.grad is not None else 0.0
            if sgd_grad_clip > 0 and d.grad is not None:
                torch.nn.utils.clip_grad_norm_([d], sgd_grad_clip)
            optimizer.step()

            need_log = (step == 1) or (step % log_every == 0) or (step == sgd_steps)
            if not need_log:
                continue

            with torch.no_grad():
                z_now = torch.sigmoid(d)
                z_sum_c_now = []
                for class_id in range(num_classes):
                    cls_idx = class_index_tensors[class_id]
                    if cls_idx.numel() == 0:
                        z_sum_c_now.append(torch.zeros((), dtype=torch.float32, device=device))
                    else:
                        z_sum_c_now.append(z_now.index_select(0, cls_idx).sum())
                z_sum_c_now_t = torch.stack(z_sum_c_now)
                cls_term = float(torch.sum((z_sum_c_now_t / float(m) - p_c_t) ** 2).item())
                data_term = float(torch.dot(z_now, s_all_t).item())
                obj = data_term - float(lambda_cls_val) * cls_term

                z_np = z_now.detach().cpu().numpy().astype(np.float32)
                proxy_j, _, proxy_omega, _, _ = _proxy_from_z(z_np, s_all)
                z_sum = float(z_now.sum().item())
                z_sum_c_np = z_sum_c_now_t.detach().cpu().numpy().astype(np.float64)
                z_gap = z_sum_c_np - target_counts
                frac_hi = float((z_np > 0.9).mean())
                frac_lo = float((z_np < 0.1).mean())
                mean_z = float(z_np.mean())
                std_z = float(z_np.std())
                d_norm = float(torch.linalg.vector_norm(d).item())
                d_max = float(torch.max(torch.abs(d)).item())

            if init_obj is None:
                init_obj = obj
                init_data_term = data_term
                init_cls_term = cls_term
                init_proxy_j = proxy_j

            delta_obj = obj - float(init_obj)
            delta_data = data_term - float(init_data_term)
            delta_cls = cls_term - float(init_cls_term)
            delta_proxy_j = proxy_j - float(init_proxy_j)

            rel_improve = (obj - best_obj) / max(abs(best_obj), 1.0) if np.isfinite(best_obj) else float("inf")
            if obj > best_obj and rel_improve >= sgd_min_delta:
                best_obj = obj
                no_improve = 0
            else:
                no_improve += 1

            if proxy_j > best_proxy_j:
                best_proxy_j = proxy_j
            if rel_improve >= sgd_min_delta and proxy_j < best_proxy_j:
                tqdm.write(
                    "[Warn] obj improves but proxy_J stagnates; consider adding (Σz-m)^2 or binarize regularizer later"
                )

            tqdm.write(
                f"[SGD][outer {outer_idx + 1}] step={step}/{sgd_steps} "
                f"obj={obj:.4f}(Δ{delta_obj:.4f}) data={data_term:.4f}(Δ{delta_data:.4f}) cls={cls_term:.6f}(Δ{delta_cls:.6f}) "
                f"z_sum={z_sum:.2f}/m={m} zc[min/mean/max]={z_sum_c_np.min():.2f}/{z_sum_c_np.mean():.2f}/{z_sum_c_np.max():.2f} "
                f"gap[min/mean/max]={z_gap.min():.2f}/{z_gap.mean():.2f}/{z_gap.max():.2f} "
                f"grad={grad_norm:.4f} ||d||={d_norm:.2f} max|d|={d_max:.2f} "
                f"proxy_J={proxy_j:.4f}(Δ{delta_proxy_j:.4f}) Ω_proxy={proxy_omega:.6f} "
                f"frac_hi/lo={frac_hi:.3f}/{frac_lo:.3f} mean/std={mean_z:.3f}/{std_z:.3f}"
            )

            last_z = z_np
            last_proxy_j = proxy_j

            if sgd_early_stop and no_improve >= sgd_early_patience:
                tqdm.write(
                    f"[SGD][outer {outer_idx + 1}] early-stop at step={step} no_improve={no_improve}/{sgd_early_patience}"
                )
                break

        return last_z, {"final_proxy_j": float(last_proxy_j)}

    selected_mask = np.zeros(num_samples, dtype=np.uint8)
    init_idx = np.random.choice(num_samples, size=m, replace=False)
    selected_mask[init_idx] = 1

    j_cur, s_cur, omega_cur, m_c = _real_stats(selected_mask)
    outer_j_history = [float(j_cur)]
    stable_count = 0
    outer_iters = min(debug_outer, max_outer_iters) if debug_outer > 0 else max_outer_iters
    z_hist_paths: list[str] = []

    outer_iterator = tqdm(
        range(outer_iters),
        desc=progress_desc or "Group optimization",
        unit="outer",
        leave=True,
    )

    for t in outer_iterator:
        j_old = float(j_cur)
        selected_mask_old = selected_mask.copy()
        max_dev_old, mc_min_old, mc_max_old, mc_std_old = _mask_stats(selected_mask_old)
        tqdm.write(
            f"[Outer-begin] t={t + 1}/{outer_iters} J={j_cur:.6f} S={s_cur:.6f} Omega={omega_cur:.6f} "
            f"max_dev={max_dev_old:.6f} m_c[min/max/std]={mc_min_old}/{mc_max_old}/{mc_std_old:.3f}"
        )

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

        z_all, _ = _optimize_z_from_scores(s_all, outer_idx=t)
        proxy_j_new, _, _, _, z_threshold = _proxy_from_z(z_all, s_all)
        top_idx = np.argpartition(-z_all, m - 1)[:m]
        selected_mask = np.zeros(num_samples, dtype=np.uint8)
        selected_mask[top_idx] = 1

        j_cur, s_cur, omega_cur, m_c = _real_stats(selected_mask)
        outer_j_history.append(float(j_cur))

        max_dev_new, mc_min_new, mc_max_new, mc_std_new = _mask_stats(selected_mask)
        changed_count = int(np.logical_xor(selected_mask_old.astype(bool), selected_mask.astype(bool)).sum())
        flip_rate = float(changed_count / float(max(m, 1)))
        z_mean = float(z_all.mean())
        z_std = float(z_all.std())
        frac_hi = float((z_all > 0.9).mean())
        frac_lo = float((z_all < 0.1).mean())
        z_hist_path = _save_outer_z_hist(t, z_all)
        z_hist_paths.append(z_hist_path)

        delta = abs(j_cur - j_old)
        stop_threshold = max(eps_stop_abs, eps_stop_rel * max(abs(j_cur), 1.0))
        if delta < stop_threshold:
            stable_count += 1
        else:
            stable_count = 0

        tqdm.write(
            f"[Outer-end] t={t + 1}/{outer_iters} J_new={j_cur:.6f} S_new={s_cur:.6f} Omega_new={omega_cur:.6f} "
            f"max_dev_new={max_dev_new:.6f} m_c[min/max/std]={mc_min_new}/{mc_max_new}/{mc_std_new:.3f} "
            f"flip_rate={flip_rate:.4f} z_th={z_threshold:.6f} z_mean/std={z_mean:.4f}/{z_std:.4f} "
            f"frac_hi/lo={frac_hi:.4f}/{frac_lo:.4f} proxy_J={proxy_j_new:.6f} z_hist={z_hist_path}"
        )
        outer_iterator.set_postfix(
            t=f"{t + 1}/{outer_iters}",
            J_cur=f"{j_cur:.3f}",
            Omega=f"{omega_cur:.6f}",
            flip=f"{flip_rate:.3f}",
            stable=f"{stable_count}/{patience}",
        )
        if stable_count >= patience:
            break

    final_mask = selected_mask.astype(np.uint8)
    final_j, final_s, final_omega, final_counts = _real_stats(final_mask)

    selected_by_class: dict[int, int] = {}
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        selected_by_class[class_id] = int(final_mask[class_indices].sum())

    stats: dict[str, object] = {
        "m": int(m),
        "lambda_cls": float(lambda_cls_val),
        "gamma_cls": float(gamma_cls),
        "m_c": {int(c): int(v) for c, v in enumerate(final_counts.tolist())},
        "Omega": float(final_omega),
        "S": float(final_s),
        "J": float(final_j),
        "outer_j_history": outer_j_history,
        "max_outer_iters": int(max_outer_iters),
        "outer_iters": int(outer_iters),
        "debug_outer": int(debug_outer),
        "sgd_steps": int(sgd_steps),
        "sgd_lr": float(sgd_lr),
        "sgd_opt": sgd_opt,
        "sgd_log_every": int(sgd_log_every),
        "sgd_grad_clip": float(sgd_grad_clip),
        "sgd_early_stop": bool(sgd_early_stop),
        "sgd_early_patience": int(sgd_early_patience),
        "sgd_min_delta": float(sgd_min_delta),
        "eps_stop_abs": float(eps_stop_abs),
        "eps_stop_rel": float(eps_stop_rel),
        "patience": int(patience),
        "z_hist_paths": z_hist_paths,
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
                    sgd_steps=args.sgd_steps,
                    sgd_lr=args.sgd_lr,
                    sgd_opt=args.sgd_opt,
                    eps_stop_abs=args.eps_stop_abs,
                    eps_stop_rel=args.eps_stop_rel,
                    patience=args.patience,
                    debug_outer=args.debug_outer,
                    sgd_log_every=args.sgd_log_every,
                    sgd_grad_clip=args.sgd_grad_clip,
                    sgd_early_stop=args.sgd_early_stop,
                    sgd_early_patience=args.sgd_early_patience,
                    sgd_min_delta=args.sgd_min_delta,
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
