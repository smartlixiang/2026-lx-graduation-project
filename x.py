from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataset.dataset_config import CIFAR10, CIFAR100
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.global_config import CONFIG
from utils.seed import set_seed
from utils.static_score_cache import get_or_compute_static_scores

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXED_SEED = 42
EVAL_SEEDS = [22, 42, 96]
KEEP_RATIOS = list(range(20, 91, 10))
METHODS = ["topk", "random", "herding", "E2LN", "GraNd", "Forgetting", "MoSo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按已有 mask 计算子集综合分数(1)与分布偏移程度(2)")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=[CIFAR10, CIFAR100])
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weight-group", type=str, default="naive", choices=["naive", "learned"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
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


def load_mask(method: str, dataset: str, seed: int, kr: int, n_samples: int, weight_group: str) -> np.ndarray | None:
    method_aliases: dict[str, list[str]] = {
        "topk": [f"{weight_group}_topk", "topk"],
        "random": ["random"],
        "herding": ["herding"],
        "E2LN": ["E2LN"],
        "GraNd": ["GraNd"],
        "Forgetting": ["Forgetting"],
        "MoSo": ["MoSo", "moso"],
    }
    aliases = method_aliases.get(method, [method])
    candidates: list[Path] = []
    for alias in aliases:
        candidates.extend(
            [
                PROJECT_ROOT / "mask" / alias / dataset / str(seed) / f"mask_{kr}.npz",
                PROJECT_ROOT / "result" / alias / dataset / str(seed) / f"mask_{kr}.npz",
            ]
        )

    for path in candidates:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as loaded:
            if "mask" in loaded:
                mask = np.asarray(loaded["mask"], dtype=np.uint8)
            else:
                key = next(iter(loaded.files), None)
                if key is None:
                    continue
                mask = np.asarray(loaded[key], dtype=np.uint8)
        if mask.shape[0] == n_samples:
            return mask
    return None


def compute_subset_metrics(
    selected_mask: np.ndarray,
    *,
    sa_scores: np.ndarray,
    dds_scores: np.ndarray,
    div_metric: Div,
    div_loader: DataLoader,
    image_adapter,
    div_features,
    labels_t: torch.Tensor,
    weights: dict[str, float],
    labels_np: np.ndarray,
    num_classes: int,
    image_features_np: np.ndarray,
    full_class_mean: np.ndarray,
) -> tuple[float, float]:
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
    selected = selected_mask.astype(bool)
    subset_score = float(
        np.sum(
            (
                weights["sa"] * sa_scores
                + weights["dds"] * dds_scores
                + weights["div"] * div_dyn
            )[selected],
            dtype=np.float64,
        )
    )

    shift_values: list[float] = []
    for class_id in range(num_classes):
        class_selected = selected & (labels_np == class_id)
        if not np.any(class_selected):
            continue
        mu_sub = np.mean(image_features_np[class_selected], axis=0, dtype=np.float32)
        shift_values.append(float(np.linalg.norm(mu_sub - full_class_mean[class_id])))
    distribution_shift = float(np.mean(shift_values)) if shift_values else 0.0
    return subset_score, distribution_shift


def main() -> None:
    args = parse_args()
    set_seed(FIXED_SEED)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    dataset_for_names = _build_dataset(args.dataset, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    labels = np.asarray(dataset_for_names.targets)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    n_samples = len(dataset_for_names)

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
        num_samples=n_samples,
        compute_fn=_compute_scores,
    )

    sa_scores = np.asarray(static_scores["sa"], dtype=np.float32)
    dds_scores = np.asarray(static_scores["dds"], dtype=np.float32)
    div_features, _ = div_metric._encode_images(div_loader, image_adapter)
    image_features_np = div_features.detach().cpu().numpy().astype(np.float32)

    full_class_mean = np.zeros((len(class_names), image_features_np.shape[1]), dtype=np.float32)
    for class_id in range(len(class_names)):
        class_feats = image_features_np[labels == class_id]
        if class_feats.shape[0] > 0:
            full_class_mean[class_id] = np.mean(class_feats, axis=0, dtype=np.float32)

    rows: list[dict[str, object]] = []
    for kr in tqdm(KEEP_RATIOS, desc="Evaluating existing masks", unit="kr"):
        row: dict[str, object] = {"kr": kr}
        for method in METHODS:
            score_list: list[float] = []
            shift_list: list[float] = []
            for seed in EVAL_SEEDS:
                mask = load_mask(method, args.dataset, seed, kr, n_samples, args.weight_group)
                if mask is None:
                    continue
                score, shift = compute_subset_metrics(
                    mask,
                    sa_scores=sa_scores,
                    dds_scores=dds_scores,
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    div_features=div_features,
                    labels_t=labels_t,
                    weights=weights,
                    labels_np=labels,
                    num_classes=len(class_names),
                    image_features_np=image_features_np,
                    full_class_mean=full_class_mean,
                )
                score_list.append(score)
                shift_list.append(shift)
            row[f"{method}_1"] = float(np.mean(score_list)) if score_list else float("nan")
            row[f"{method}_2"] = float(np.mean(shift_list)) if shift_list else float("nan")
        rows.append(row)

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "group_baseline" / args.dataset / f"{args.weight_group}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["kr"] + [f"{method}_{idx}" for method in METHODS for idx in (1, 2)]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved baseline table to: {output_path}")


if __name__ == "__main__":
    main()
