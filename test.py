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
import hashlib
import json
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
    # Read learned scoring weights for the given seed from weights/scoring_weights.json.
    learned_weights = all_weights.get(str(seed))
    if not isinstance(learned_weights, dict):
        raise KeyError(f"missing learned weights for seed={seed}")
    return {k: float(learned_weights[k]) for k in ("sa", "div", "dds")}


# Note:
# `group_lambda.json` may contain cached `lambda_cls` / `lambda_mean` as final values
# under older scaling settings. This test script must NOT directly reuse them.
# We only reuse scaling-independent statistics `lambda_std_cls` and
# `lambda_std_mean`, then apply fixed factors: class-balance uses 5, and herding uses dataset defaults (cifar10=2, cifar100=6).
def load_lambda_scaled(dataset: str, seed: int, kr: int) -> tuple[float, float] | None:
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

    lambda_cls = 5.0 * float(record["lambda_std_cls"])
    mean_scale_by_dataset = {"cifar10": 2.0, "cifar100": 6.0}
    mean_scale = float(mean_scale_by_dataset.get(str(dataset).lower(), 2.0))
    lambda_mean = mean_scale * float(record["lambda_std_mean"])
    return lambda_cls, lambda_mean


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


def compute_mean_penalty(
    selected_mask: np.ndarray,
    *,
    labels_np: np.ndarray,
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


def compute_subset_scores(
    selected_mask: np.ndarray,
    *,
    sa_scores: np.ndarray,
    weights: dict[str, float],
    div_metric: Div,
    dds_metric: DifficultyDirection,
    div_loader: DataLoader,
    dds_loader: DataLoader,
    image_adapter,
    div_features,
    dds_features,
    labels_t: torch.Tensor,
    labels_np: np.ndarray,
    num_classes: int,
    target_size: int,
    class_indices_list: list[np.ndarray],
    class_features_list: list[np.ndarray],
    full_class_mean: np.ndarray,
    full_class_var: np.ndarray,
    lambda_cls: float,
    lambda_mean: float,
) -> dict[str, float]:
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
    score_ref = weights["sa"] * sa_scores + weights["div"] * div_dyn + weights["dds"] * dds_dyn
    raw_score = float(np.sum(score_ref[chosen]))
    class_penalty = compute_balance_penalty(selected_mask, labels_np, num_classes, target_size)
    mean_penalty = compute_mean_penalty(
        selected_mask,
        labels_np=labels_np,
        num_classes=num_classes,
        class_indices_list=class_indices_list,
        class_features_list=class_features_list,
        full_class_mean=full_class_mean,
        full_class_var=full_class_var,
        eps=DEFAULT_EPS,
    )
    class_correction = float(lambda_cls * class_penalty)
    mean_correction = float(lambda_mean * mean_penalty)
    adjusted_score = float(raw_score - class_correction - mean_correction)
    return {
        "raw_score": raw_score,
        "class_correction": class_correction,
        "mean_correction": mean_correction,
        "adjusted_score": adjusted_score,
    }


def print_table(kr: int, rows: list[tuple[str, dict[str, float]]]) -> None:
    print(f"\n=== KR={kr} ===")
    header = f"{'method':<16}{'raw_score':>14}{'class_correction':>20}{'mean_correction':>20}{'adjusted_score':>18}"
    print(header)
    print("-" * len(header))
    for method, vals in rows:
        print(
            f"{method:<16}"
            f"{vals['raw_score']:>14.4f}"
            f"{vals['class_correction']:>20.4f}"
            f"{vals['mean_correction']:>20.4f}"
            f"{vals['adjusted_score']:>18.4f}"
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


def main() -> None:
    args = parse_args()
    keep_ratios = parse_csv_ints(args.krs)
    seeds = parse_csv_ints(args.seeds)
    if not keep_ratios or not seeds:
        raise ValueError("krs and seeds must be non-empty")

    device = torch.device(args.device) if args.device else CONFIG.global_device
    set_seed(int(seeds[0]))

    dataset_plain = _build_dataset(args.dataset, transform=None)
    class_names = dataset_plain.classes  # type: ignore[attr-defined]
    labels_np = np.asarray(dataset_plain.targets, dtype=np.int64)
    n_samples = int(labels_np.shape[0])
    num_classes = int(len(class_names))
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, args.dataset)

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    dds_loader = build_loader(args.dataset, dds_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
    div_loader = build_loader(args.dataset, div_metric.extractor.preprocess, device, args.batch_size, args.num_workers)
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

    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    dds_features, _ = dds_metric._encode_images(dds_loader, image_adapter)
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

    methods = ["random", "herding", "learned_topk", "learned_group"]
    output_dir = PROJECT_ROOT / args.output_dir

    for kr in tqdm(keep_ratios, desc="KR loop", unit="kr"):
        method_results: dict[str, list[dict[str, float]]] = {m: [] for m in methods}
        seed_iter = tqdm(seeds, desc=f"Seed loop (kr={kr})", unit="seed", leave=False)
        for seed in seed_iter:
            target_size = max(1, min(n_samples, int(round(n_samples * kr / 100.0))))
            try:
                weights = load_scoring_weights(all_weights, seed)
            except Exception as exc:
                print(f"[Warning] skip seed={seed}, kr={kr}, missing learned weights: {exc}")
                continue
            lambda_pair = load_lambda_scaled(args.dataset, seed, kr)
            if lambda_pair is None:
                print(f"[Warning] skip seed={seed}, kr={kr}, missing lambda std cache")
                continue
            lambda_cls, lambda_mean = lambda_pair

            rng = np.random.default_rng(int(seed))
            random_idx = rng.choice(n_samples, size=target_size, replace=False)
            random_mask = np.zeros(n_samples, dtype=np.uint8)
            random_mask[random_idx] = 1
            method_results["random"].append(
                compute_subset_scores(
                    random_mask,
                    sa_scores=sa_scores,
                    weights=weights,
                    div_metric=div_metric,
                    dds_metric=dds_metric,
                    div_loader=div_loader,
                    dds_loader=dds_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    dds_features=dds_features,
                    labels_t=labels_t,
                    labels_np=labels_np,
                    num_classes=num_classes,
                    target_size=target_size,
                    class_indices_list=class_indices_list,
                    class_features_list=class_features_list,
                    full_class_mean=full_class_mean,
                    full_class_var=full_class_var,
                    lambda_cls=lambda_cls,
                    lambda_mean=lambda_mean,
                )
            )

            for method in ("herding", "learned_topk", "learned_group"):
                mask = load_mask_for_method(method, args.dataset, args.model_name, seed, kr, n_samples)
                if mask is None:
                    continue
                method_results[method].append(
                    compute_subset_scores(
                        mask,
                        sa_scores=sa_scores,
                        weights=weights,
                        div_metric=div_metric,
                        dds_metric=dds_metric,
                        div_loader=div_loader,
                        dds_loader=dds_loader,
                        image_adapter=image_adapter,
                        div_features=div_features,
                        dds_features=dds_features,
                        labels_t=labels_t,
                        labels_np=labels_np,
                        num_classes=num_classes,
                        target_size=int(mask.sum()),
                        class_indices_list=class_indices_list,
                        class_features_list=class_features_list,
                        full_class_mean=full_class_mean,
                        full_class_var=full_class_var,
                        lambda_cls=lambda_cls,
                        lambda_mean=lambda_mean,
                    )
                )

        table_rows: list[tuple[str, dict[str, float]]] = []
        for method in methods:
            vals = method_results[method]
            if not vals:
                print(f"[Warning] empty result for method={method}, kr={kr}")
                avg = {"raw_score": np.nan, "class_correction": np.nan, "mean_correction": np.nan, "adjusted_score": np.nan}
            else:
                avg = {
                    "raw_score": float(np.mean([v["raw_score"] for v in vals])),
                    "class_correction": float(np.mean([v["class_correction"] for v in vals])),
                    "mean_correction": float(np.mean([v["mean_correction"] for v in vals])),
                    "adjusted_score": float(np.mean([v["adjusted_score"] for v in vals])),
                }
            table_rows.append((method, avg))

        print_table(kr, table_rows)
        save_plot(args.dataset, kr, table_rows, output_dir)

    print("\nExample command:")
    print("python test.py --dataset cifar10 --krs 20,30,40,50,60 --seeds 22,42,96")


if __name__ == "__main__":
    main()
