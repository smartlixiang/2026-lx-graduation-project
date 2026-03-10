from __future__ import annotations

from utils.static_score_cache import get_or_compute_static_scores
from utils.seed import set_seed
from utils.path_rules import resolve_mask_path
from utils.group_lambda import DEFAULT_EPS, compute_balance_penalty
from utils.global_config import CONFIG
from scoring import DifficultyDirection, Div, SemanticAlignment
from model.adapter import load_trained_adapters
from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100

import argparse
import csv
import hashlib
import json
import math
import sys
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

METHODS = ["random", "herding", "learned_topk", "learned_group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare group score components under different keep ratios")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--krs", type=str, default="20,30,40,50,60,70,80,90")
    parser.add_argument("--seeds", type=str, default="22,42,96")
    parser.add_argument("--output-dir", type=str, default="test")
    parser.add_argument("--mean-linear-a", type=float, default=1.0)
    parser.add_argument("--class-scale", type=float, default=5.0)
    parser.add_argument("--mean-linear-b", type=float, default=0.0)
    parser.add_argument("--scan-a", type=str, default="")
    parser.add_argument("--scan-b", type=str, default="")
    parser.add_argument("--acc-source", type=str, default="")
    parser.add_argument("--save-diagnostics", action="store_true")
    return parser.parse_args()


def _build_dataset(dataset_name: str, transform):
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_loader(dataset_name: str, preprocess, device: torch.device, batch_size: int, num_workers: int) -> DataLoader:
    ds = _build_dataset(dataset_name, preprocess)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")


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
            cached_n = int(np.asarray(cached["n_samples"]).item())
            cached_dim = int(np.asarray(cached["feat_dim"]).item())
            cached_cls = int(np.asarray(cached["num_classes"]).item())
            means = np.asarray(cached["full_class_mean"], dtype=np.float32)
            vars_ = np.asarray(cached["full_class_var"], dtype=np.float32)
            if (
                cached_n == n_samples
                and cached_dim == feat_dim
                and cached_cls == int(num_classes)
                and means.shape == (num_classes, feat_dim)
                and vars_.shape == (num_classes,)
            ):
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

    np.savez_compressed(
        cache_path,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        n_samples=np.asarray(n_samples, dtype=np.int64),
        feat_dim=np.asarray(feat_dim, dtype=np.int64),
        num_classes=np.asarray(num_classes, dtype=np.int64),
    )
    return full_class_mean, full_class_var


def ensure_scoring_weights(path: Path, dataset_name: str) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    dataset_entry = data.get(dataset_name, {})
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
    return {group_name: group for group_name, group in dataset_entry.items() if isinstance(group, dict)}


def load_scoring_weights(all_weights: dict[str, dict[str, object]], seed: int) -> dict[str, float]:
    learned_weights = all_weights.get(str(seed))
    if not isinstance(learned_weights, dict):
        raise KeyError(f"missing learned weights for seed={seed}")
    return {k: float(learned_weights[k]) for k in ("sa", "div", "dds")}


def load_lambda_std(dataset: str, seed: int, kr: int) -> tuple[float, float] | None:
    cache_path = PROJECT_ROOT / "utils" / "group_lambda.json"
    if not cache_path.exists():
        print(f"[Warning] lambda cache file missing: {cache_path}")
        return None
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    ds_entry = cache.get(str(dataset), {})
    seed_entry = ds_entry.get(str(seed), {}) if isinstance(ds_entry, dict) else {}
    kr_entry = seed_entry.get(str(kr), {}) if isinstance(seed_entry, dict) else {}
    record = kr_entry.get("learned", {}) if isinstance(kr_entry, dict) else {}

    if not isinstance(record, dict) or not record:
        print(f"[Warning] lambda cache missing learned group: dataset={dataset}, seed={seed}, kr={kr}")
        return None
    if "lambda_std_cls" not in record or "lambda_std_mean" not in record:
        print(f"[Warning] lambda_std fields missing in learned group for dataset={dataset}, seed={seed}, kr={kr}")
        return None
    return float(record["lambda_std_cls"]), float(record["lambda_std_mean"])


def load_mask_for_method(method: str, dataset: str, model_name: str, seed: int, kr: int, n_samples: int) -> np.ndarray | None:
    mode_candidates: dict[str, list[str]] = {
        "herding": ["herding"],
        "learned_topk": ["learned_topk", "topk"],
        "learned_group": ["learned_group", "group"],
    }
    candidates = mode_candidates[method]
    for mode in candidates:
        mask_path = resolve_mask_path(mode=mode, dataset=dataset, model=model_name, seed=seed, keep_ratio=kr)
        if not mask_path.exists() and method == "herding":
            fallback = resolve_mask_path(mode=mode, dataset=dataset, model=model_name, seed=22, keep_ratio=kr)
            if fallback.exists():
                mask_path = fallback
        if not mask_path.exists():
            continue
        try:
            with np.load(mask_path) as data:
                if "mask" in data:
                    mask = np.asarray(data["mask"]).astype(np.uint8)
                elif len(data.files) == 1:
                    mask = np.asarray(data[data.files[0]]).astype(np.uint8)
                else:
                    print(f"[Warning] bad mask format: {mask_path}")
                    continue
        except Exception as exc:
            print(f"[Warning] failed reading mask: {mask_path} ({exc})")
            continue
        if mask.shape != (n_samples,):
            print(f"[Warning] mask shape mismatch: {mask_path}, got={mask.shape}, expected=({n_samples},)")
            continue
        mask = (mask > 0).astype(np.uint8)
        return mask
    print(f"[Warning] missing mask for method={method}, seed={seed}, kr={kr}, tried modes={candidates}")
    return None


def select_random_mask(n_samples: int, keep_ratio: int) -> np.ndarray:
    k = max(1, min(n_samples, int(round(n_samples * keep_ratio / 100.0))))
    idx = np.random.choice(n_samples, size=k, replace=False)
    mask = np.zeros(n_samples, dtype=np.uint8)
    mask[idx] = 1
    return mask


def compute_mean_penalty(
    selected_mask: np.ndarray,
    *,
    num_classes: int,
    class_indices_list: list[np.ndarray],
    class_features_list: list[np.ndarray],
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
    eps: float,
) -> float:
    selected = selected_mask.astype(bool)
    penalty_sum = 0.0
    for class_id in range(num_classes):
        class_indices = class_indices_list[class_id]
        if class_indices.size == 0:
            continue
        class_selected = selected[class_indices]
        count = int(np.sum(class_selected))
        if count <= 0:
            continue
        subset_sum = np.sum(class_features_list[class_id][class_selected], axis=0, dtype=np.float32)
        subset_mean = subset_sum / float(count)
        diff = subset_mean - full_class_mean[class_id]
        dist2 = float(np.dot(diff, diff))
        penalty_sum += dist2 / (float(full_class_var[class_id]) + eps)
    return float(penalty_sum)


def compute_subset_base(
    selected_mask: np.ndarray,
    *,
    sa_scores: np.ndarray,
    weights: dict[str, float],
    dds_scores: np.ndarray,
    div_metric: Div,
    div_loader: DataLoader,
    image_adapter,
    div_features,
    labels_t: torch.Tensor,
    labels_np: np.ndarray,
    num_classes: int,
    target_size: int,
    class_indices_list: list[np.ndarray],
    class_features_list: list[np.ndarray],
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
    lambda_std_cls: float,
    lambda_std_mean: float,
) -> dict[str, float]:
    # Div remains subset-dynamic; DDS is fixed as static full-dataset cached scores.
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
    score_ref = weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_scores
    raw_score = float(np.sum(score_ref[chosen]))
    class_penalty_raw = compute_balance_penalty(selected_mask, labels_np, num_classes, target_size)
    mean_penalty_raw = compute_mean_penalty(
        selected_mask,
        num_classes=num_classes,
        class_indices_list=class_indices_list,
        class_features_list=class_features_list,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        eps=DEFAULT_EPS,
    )
    class_base = float(lambda_std_cls * class_penalty_raw)
    mean_base = float(lambda_std_mean * mean_penalty_raw)
    return {
        "raw_score": raw_score,
        "class_penalty_raw": float(class_penalty_raw),
        "mean_penalty_raw": float(mean_penalty_raw),
        "lambda_std_cls": float(lambda_std_cls),
        "lambda_std_mean": float(lambda_std_mean),
        "class_base": class_base,
        "mean_base": mean_base,
    }


def mean_scale_from_linear(a: float, b: float, kr: int) -> float:
    return float(max(0.0, a - b * float(kr)))


def finalize_scores(base_row: dict[str, float], mean_scale: float, class_scale: float) -> dict[str, float]:
    class_correction = float(class_scale * base_row["class_base"])
    mean_correction = float(mean_scale * base_row["mean_base"])
    adjusted_score = float(base_row["raw_score"] - class_correction - mean_correction)
    out = dict(base_row)
    out.update(
        {
            "mean_scale": float(mean_scale),
            "class_scale": float(class_scale),
            "class_correction": class_correction,
            "mean_correction": mean_correction,
            "adjusted_score": adjusted_score,
        }
    )
    return out


def print_table(kr: int, rows: list[tuple[str, dict[str, float]]]) -> None:
    print(f"\n=== KR={kr} ===")
    header = (
        f"{'method':<16}{'raw_score':>14}{'class_base':>14}{'mean_base':>14}"
        f"{'mean_scale':>13}{'class_corr':>14}{'mean_corr':>14}{'adjusted':>14}"
    )
    print(header)
    print("-" * len(header))
    for method, vals in rows:
        print(
            f"{method:<16}"
            f"{vals['raw_score']:>14.4f}"
            f"{vals['class_base']:>14.4f}"
            f"{vals['mean_base']:>14.4f}"
            f"{vals['mean_scale']:>13.4f}"
            f"{vals['class_correction']:>14.4f}"
            f"{vals['mean_correction']:>14.4f}"
            f"{vals['adjusted_score']:>14.4f}"
        )


def save_plot(dataset: str, kr: int, rows: list[tuple[str, dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [name for name, _ in rows]
    metric_labels = ["total score", "class correction", "herding correction"]
    metric_keys = ["adjusted_score", "class_correction", "mean_correction"]
    x = np.arange(len(metric_labels), dtype=np.float64)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(methods))
    for idx, method in enumerate(methods):
        vals = rows[idx][1]
        heights = [vals[k] for k in metric_keys]
        ax.bar(x + offsets[idx], heights, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("score")
    ax.set_xlabel("metric")
    ax.set_title(f"{dataset} | kr={kr}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{kr}.png", dpi=150)
    plt.close(fig)


def save_mean_scale_curve(dataset: str, keep_ratios: list[int], a: float, b: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = np.asarray(keep_ratios, dtype=np.float32)
    ys = np.asarray([mean_scale_from_linear(a, b, int(kr)) for kr in keep_ratios], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("keep_ratio")
    ax.set_ylabel("mean_scale")
    ax.set_title(f"mean_scale(kr) = max(0, {a:.4f} - {b:.4f} * kr) | {dataset}")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"mean_scale_curve_{dataset}.png", dpi=150)
    plt.close(fig)


def save_scan_heatmap(dataset: str, scan_rows: list[dict[str, float]], a_values: list[float], b_values: list[float], output_dir: Path) -> None:
    if not scan_rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = np.full((len(a_values), len(b_values)), np.nan, dtype=np.float32)
    a_index = {v: i for i, v in enumerate(a_values)}
    b_index = {v: i for i, v in enumerate(b_values)}
    for row in scan_rows:
        ai = a_index[float(row["a"])]
        bi = b_index[float(row["b"])]
        grid[ai, bi] = float(row["mean_spearman"])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(b_values)))
    ax.set_xticklabels([f"{v:.4f}" for v in b_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(a_values)))
    ax.set_yticklabels([f"{v:.4f}" for v in a_values])
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"mean_spearman heatmap | {dataset}")
    fig.colorbar(im, ax=ax, label="mean_spearman")
    fig.tight_layout()
    fig.savefig(output_dir / f"scan_heatmap_{dataset}.png", dpi=150)
    plt.close(fig)


def load_acc_source(path_text: str) -> dict[str, dict[str, float]]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[Warning] acc source missing: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"[Warning] bad acc source format: {path}")
        return {}
    parsed: dict[str, dict[str, float]] = {}
    for kr, row in data.items():
        if not isinstance(row, dict):
            continue
        parsed[str(kr)] = {}
        for method in METHODS:
            if method not in row:
                print(f"[Warning] acc source missing method={method} for kr={kr}")
                continue
            parsed[str(kr)][method] = float(row[method])
    return parsed


def rank_from_scores(items: dict[str, float], descending: bool = True) -> list[str]:
    return [k for k, _ in sorted(items.items(), key=lambda kv: kv[1], reverse=descending)]


def _rank_values(order: list[str]) -> dict[str, float]:
    return {name: float(i + 1) for i, name in enumerate(order)}


def spearman_corr(acc_vals: dict[str, float], score_vals: dict[str, float]) -> float:
    common = [m for m in METHODS if m in acc_vals and m in score_vals]
    if len(common) < 2:
        return float("nan")
    acc_order = rank_from_scores({m: acc_vals[m] for m in common}, descending=True)
    score_order = rank_from_scores({m: score_vals[m] for m in common}, descending=True)
    acc_rank = _rank_values(acc_order)
    score_rank = _rank_values(score_order)
    x = np.asarray([acc_rank[m] for m in common], dtype=np.float64)
    y = np.asarray([score_rank[m] for m in common], dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def kendall_tau(acc_vals: dict[str, float], score_vals: dict[str, float]) -> float:
    common = [m for m in METHODS if m in acc_vals and m in score_vals]
    if len(common) < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            mi, mj = common[i], common[j]
            da = acc_vals[mi] - acc_vals[mj]
            ds = score_vals[mi] - score_vals[mj]
            if da == 0 or ds == 0:
                continue
            if da * ds > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return float("nan")
    return float((concordant - discordant) / total)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def pairwise_threshold(row1: dict[str, float], row2: dict[str, float]) -> tuple[float, str]:
    den = float(row1["mean_base"] - row2["mean_base"])
    if abs(den) <= 1e-12:
        return float("nan"), "unsolved(mean_base equal)"
    num = float((row1["raw_score"] - row1["class_base"]) - (row2["raw_score"] - row2["class_base"]))
    return float(num / den), "ok"


def main() -> None:
    args = parse_args()
    keep_ratios = parse_csv_ints(args.krs)
    seeds = parse_csv_ints(args.seeds)
    if not keep_ratios or not seeds:
        raise ValueError("krs and seeds must be non-empty")

    scan_a_values = parse_csv_floats(args.scan_a) if args.scan_a.strip() else []
    scan_b_values = parse_csv_floats(args.scan_b) if args.scan_b.strip() else []
    do_scan = bool(scan_a_values and scan_b_values)

    acc_data = load_acc_source(args.acc_source)

    device = torch.device(args.device) if args.device else CONFIG.global_device
    set_seed(int(seeds[0]))

    dataset_plain = _build_dataset(args.dataset, transform=None)
    class_names = dataset_plain.classes  # type: ignore[attr-defined]
    labels_np = np.asarray(dataset_plain.targets, dtype=np.int64)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    n_samples = int(labels_np.shape[0])
    num_classes = int(len(class_names))

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, args.dataset)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    div_loader = build_loader(args.dataset, div_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    dds_loader = build_loader(args.dataset, dds_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    sa_loader = build_loader(args.dataset, sa_metric.extractor.preprocess, device, args.batch_size, args.num_workers)

    adapter_seed = int(seeds[0])
    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=adapter_seed,
        map_location=device,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(tqdm(dds_loader, desc="Scoring DDS", unit="batch"), adapter=image_adapter).scores
        div_scores_local = div_metric.score_dataset(tqdm(div_loader, desc="Scoring Div", unit="batch"), adapter=image_adapter).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {
            "sa": np.asarray(sa_scores_local),
            "div": np.asarray(div_scores_local),
            "dds": np.asarray(dds_scores_local),
            "labels": labels_np,
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=args.dataset,
        seed=adapter_seed,
        clip_model=args.clip_model,
        adapter_image_path=str(adapter_paths["image_path"]),
        adapter_text_path=str(adapter_paths["text_path"]),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=n_samples,
        compute_fn=_compute_scores,
    )
    sa_scores = np.asarray(static_scores["sa"], dtype=np.float32)
    dds_scores = np.asarray(static_scores["dds"], dtype=np.float32)

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    div_features_np = (
        div_features.detach().cpu().numpy() if isinstance(div_features, torch.Tensor) else np.asarray(div_features)
    ).astype(np.float32)
    mean_stats_cache = _mean_stats_cache_path(args.dataset, args.clip_model, str(adapter_paths["image_path"]))
    full_class_mean, full_class_var = _get_or_compute_group_mean_stats(
        cache_path=mean_stats_cache,
        image_features=div_features_np,
        labels=labels_np,
        num_classes=num_classes,
    )
    class_indices_list = [np.flatnonzero(labels_np == class_id).astype(np.int64) for class_id in range(num_classes)]
    class_features_list = [div_features_np[idx] for idx in class_indices_list]

    output_dir = PROJECT_ROOT / args.output_dir
    diagnostics_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for kr in tqdm(keep_ratios, desc="KR loop", unit="kr"):
        method_results: dict[str, list[dict[str, float]]] = {m: [] for m in METHODS}
        seed_iter = tqdm(seeds, desc=f"Seed loop (kr={kr})", unit="seed", leave=False)
        for seed in seed_iter:
            target_size = max(1, min(n_samples, int(round(n_samples * kr / 100.0))))
            try:
                weights = load_scoring_weights(all_weights, seed)
            except Exception as exc:
                print(f"[Warning] skip seed={seed}, kr={kr}, missing learned weights: {exc}")
                continue
            lambda_pair = load_lambda_std(args.dataset, seed, kr)
            if lambda_pair is None:
                print(f"[Warning] skip seed={seed}, kr={kr}, missing lambda std cache")
                continue
            lambda_std_cls, lambda_std_mean = lambda_pair

            rng = np.random.default_rng(int(seed))
            random_idx = rng.choice(n_samples, size=target_size, replace=False)
            random_mask = np.zeros(n_samples, dtype=np.uint8)
            random_mask[random_idx] = 1
            random_base = compute_subset_base(
                random_mask,
                sa_scores=sa_scores,
                weights=weights,
                dds_scores=dds_scores,
                div_metric=div_metric,
                div_loader=div_loader,
                image_adapter=image_adapter,
                div_features=div_features,
                labels_t=labels_t,
                labels_np=labels_np,
                num_classes=num_classes,
                target_size=target_size,
                class_indices_list=class_indices_list,
                class_features_list=class_features_list,
                full_class_mean=full_class_mean,
                full_class_var=full_class_var,
                lambda_std_cls=lambda_std_cls,
                lambda_std_mean=lambda_std_mean,
            )
            random_base["seed"] = float(seed)
            method_results["random"].append(random_base)

            for method in ("herding", "learned_topk", "learned_group"):
                mask = load_mask_for_method(method, args.dataset, args.model_name, seed, kr, n_samples)
                if mask is None:
                    continue
                method_base = compute_subset_base(
                    mask,
                    sa_scores=sa_scores,
                    weights=weights,
                    dds_scores=dds_scores,
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    labels_t=labels_t,
                    labels_np=labels_np,
                    num_classes=num_classes,
                    target_size=int(mask.sum()),
                    class_indices_list=class_indices_list,
                    class_features_list=class_features_list,
                    full_class_mean=full_class_mean,
                    full_class_var=full_class_var,
                    lambda_std_cls=lambda_std_cls,
                    lambda_std_mean=lambda_std_mean,
                )
                method_base["seed"] = float(seed)
                method_results[method].append(method_base)

        mean_scale = mean_scale_from_linear(args.mean_linear_a, args.mean_linear_b, kr)
        table_rows: list[tuple[str, dict[str, float]]] = []
        per_method_agg: dict[str, dict[str, float]] = {}
        for method in METHODS:
            vals = method_results[method]
            if not vals:
                print(f"[Warning] empty result for method={method}, kr={kr}")
                base_avg = {
                    "raw_score": float("nan"),
                    "class_penalty_raw": float("nan"),
                    "mean_penalty_raw": float("nan"),
                    "lambda_std_cls": float("nan"),
                    "lambda_std_mean": float("nan"),
                    "class_base": float("nan"),
                    "mean_base": float("nan"),
                }
            else:
                base_avg = {
                    k: safe_mean([float(v[k]) for v in vals])
                    for k in vals[0].keys()
                    if k != "seed"
                }

            finalized = finalize_scores(base_avg, mean_scale, args.class_scale)
            table_rows.append((method, finalized))
            per_method_agg[method] = finalized
            summary_rows.append(
                {
                    "dataset": args.dataset,
                    "kr": kr,
                    "method": method,
                    **{k: finalized[k] for k in finalized},
                }
            )

            for per_seed_base in vals:
                seed_value = int(per_seed_base.get("seed", -1))
                per_seed_core = {k: v for k, v in per_seed_base.items() if k != "seed"}
                final_seed = finalize_scores(per_seed_core, mean_scale, args.class_scale)
                diagnostics_rows.append(
                    {
                        "dataset": args.dataset,
                        "kr": kr,
                        "seed": seed_value,
                        "method": method,
                        **{k: final_seed[k] for k in final_seed},
                    }
                )

        print_table(kr, table_rows)
        save_plot(args.dataset, kr, table_rows, output_dir)

        pairs = [
            ("random", "learned_group"),
            ("herding", "learned_group"),
            ("random", "learned_topk"),
            ("herding", "learned_topk"),
        ]
        print(f"\n[Pairwise thresholds] kr={kr}")
        for m1, m2 in pairs:
            t, status = pairwise_threshold(per_method_agg[m1], per_method_agg[m2])
            print(f"  {m1} vs {m2}: A*={t:.4f} ({status})")
            threshold_rows.append(
                {
                    "dataset": args.dataset,
                    "kr": kr,
                    "pair": f"{m1}_vs_{m2}",
                    "method_1": m1,
                    "method_2": m2,
                    "A_star": t,
                    "status": status,
                }
            )

    if args.save_diagnostics:
        diag_json_path = output_dir / f"diagnostics_{args.dataset}.json"
        diag_csv_path = output_dir / f"diagnostics_{args.dataset}.csv"
        summary_csv_path = output_dir / f"summary_{args.dataset}.csv"
        thresholds_csv_path = output_dir / f"thresholds_{args.dataset}.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        diag_json_path.write_text(json.dumps(diagnostics_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(
            diag_csv_path,
            diagnostics_rows,
            [
                "dataset",
                "kr",
                "seed",
                "method",
                "raw_score",
                "class_penalty_raw",
                "mean_penalty_raw",
                "lambda_std_cls",
                "lambda_std_mean",
                "class_base",
                "mean_base",
                "mean_scale",
                "class_scale",
                "class_correction",
                "mean_correction",
                "adjusted_score",
            ],
        )
        write_csv(
            summary_csv_path,
            summary_rows,
            [
                "dataset",
                "kr",
                "method",
                "raw_score",
                "class_penalty_raw",
                "mean_penalty_raw",
                "lambda_std_cls",
                "lambda_std_mean",
                "class_base",
                "mean_base",
                "mean_scale",
                "class_scale",
                "class_correction",
                "mean_correction",
                "adjusted_score",
            ],
        )
        write_csv(
            thresholds_csv_path,
            threshold_rows,
            ["dataset", "kr", "pair", "method_1", "method_2", "A_star", "status"],
        )

    save_mean_scale_curve(args.dataset, keep_ratios, args.mean_linear_a, args.mean_linear_b, output_dir)

    summary_by_kr_method: dict[int, dict[str, dict[str, float]]] = {}
    for row in summary_rows:
        kr = int(row["kr"])
        summary_by_kr_method.setdefault(kr, {})[str(row["method"])] = {k: float(row[k]) for k in row if k not in {"dataset", "kr", "method"}}

    if acc_data:
        print("\n=== Rank consistency analysis (current a,b) ===")
        spearmans: list[float] = []
        top1_matches = 0
        for kr in keep_ratios:
            kr_key = str(kr)
            if kr_key not in acc_data:
                print(f"[Warning] acc-source missing kr={kr}")
                continue
            if kr not in summary_by_kr_method:
                continue
            acc_vals = acc_data[kr_key]
            score_vals = {m: summary_by_kr_method[kr][m]["adjusted_score"] for m in METHODS if m in summary_by_kr_method[kr]}
            acc_rank = rank_from_scores(acc_vals, descending=True)
            score_rank = rank_from_scores(score_vals, descending=True)
            sp = spearman_corr(acc_vals, score_vals)
            kd = kendall_tau(acc_vals, score_vals)
            top1_ok = bool(acc_rank and score_rank and acc_rank[0] == score_rank[0])
            spearmans.append(sp)
            top1_matches += int(top1_ok)
            print(f"kr={kr} | acc_rank={acc_rank} | score_rank={score_rank} | spearman={sp:.4f} | kendall={kd:.4f} | top1={top1_ok}")

        mean_s = safe_mean(spearmans)
        print(f"[Global] mean_spearman={mean_s:.4f}, top1_match_count={top1_matches}/{len(keep_ratios)}")
    else:
        print("[Info] --acc-source empty, skip rank consistency analysis.")

    if do_scan:
        print("\n=== Linear scan over (a,b) ===")
        scan_rows: list[dict[str, float]] = []
        scan_iter = tqdm([(a, b) for a in scan_a_values for b in scan_b_values], desc="Scan (a,b)", unit="pair")
        for a, b in scan_iter:
            spearmans: list[float] = []
            top1_matches = 0
            for kr in keep_ratios:
                if kr not in summary_by_kr_method:
                    continue
                score_vals: dict[str, float] = {}
                mean_scale_scan = mean_scale_from_linear(a, b, kr)
                for method in METHODS:
                    if method not in summary_by_kr_method[kr]:
                        continue
                    base = summary_by_kr_method[kr][method]
                    score_vals[method] = float(base["raw_score"] - args.class_scale * base["class_base"] - mean_scale_scan * base["mean_base"])
                if acc_data and str(kr) in acc_data:
                    acc_vals = acc_data[str(kr)]
                    sp = spearman_corr(acc_vals, score_vals)
                    spearmans.append(sp)
                    acc_rank = rank_from_scores(acc_vals, descending=True)
                    score_rank = rank_from_scores(score_vals, descending=True)
                    if acc_rank and score_rank and acc_rank[0] == score_rank[0]:
                        top1_matches += 1
            mean_s = safe_mean(spearmans) if acc_data else float("nan")
            scan_rows.append(
                {
                    "a": float(a),
                    "b": float(b),
                    "mean_spearman": float(mean_s),
                    "top1_match_count": float(top1_matches),
                }
            )

        scan_rows_sorted = sorted(
            scan_rows,
            key=lambda r: (
                -1e9 if math.isnan(float(r["mean_spearman"])) else float(r["mean_spearman"]),
                float(r["top1_match_count"]),
            ),
            reverse=True,
        )
        print("a,b,mean_spearman,top1_match_count")
        for row in scan_rows_sorted[: min(20, len(scan_rows_sorted))]:
            print(f"{row['a']:.4f},{row['b']:.4f},{row['mean_spearman']:.4f},{int(row['top1_match_count'])}")

        if scan_rows_sorted:
            best = scan_rows_sorted[0]
            print(
                f"best linear formula: scale_mean(kr) = {best['a']:.4f} - {best['b']:.4f} * kr "
                f"(mean_spearman={best['mean_spearman']:.4f}, top1_match_count={int(best['top1_match_count'])})"
            )

        scan_csv_path = output_dir / f"scan_{args.dataset}.csv"
        write_csv(scan_csv_path, scan_rows_sorted, ["a", "b", "mean_spearman", "top1_match_count"])
        save_scan_heatmap(args.dataset, scan_rows, scan_a_values, scan_b_values, output_dir)

    print("\nExample commands:")
    print(
        "python test.py --dataset cifar100 --class-scale 5 --mean-linear-a 7.0 --mean-linear-b 0.06 "
        "--acc-source test/cifar100_acc.json --save-diagnostics"
    )
    print(
        "python test.py --dataset cifar100 --class-scale 5 --scan-a 4,5,6,7,8,9,10 --scan-b 0.03,0.04,0.05,0.06,0.07,0.08,0.09 "
        "--acc-source test/cifar100_acc.json --save-diagnostics"
    )


if __name__ == "__main__":
    main()
