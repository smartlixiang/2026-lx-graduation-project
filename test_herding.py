from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import Div  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.group_lambda import compute_balance_penalty  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="展示 CIFAR10/CIFAR100 上不同方法的 herding 修正项（按类别平均）"
    )
    parser.add_argument("--datasets", type=str, default=f"{CIFAR10},{CIFAR100}")
    parser.add_argument("--kr", type=str, default="20,30,50,70,90")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_strs(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def _build_dataset(dataset_name: str, transform):
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = data_root / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_mask(path: Path, n_samples: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"mask not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "mask" in data:
            mask = np.asarray(data["mask"], dtype=np.uint8)
        else:
            key = next(iter(data.files), None)
            if key is None:
                raise ValueError(f"empty npz file: {path}")
            mask = np.asarray(data[key], dtype=np.uint8)
    mask = mask.reshape(-1)
    if mask.shape[0] != n_samples:
        raise ValueError(f"mask length mismatch: {path}, got {mask.shape[0]}, expected {n_samples}")
    return (mask > 0).astype(np.uint8)


def _random_mask(n_samples: int, keep_ratio: int, rng: np.random.Generator) -> np.ndarray:
    k = max(1, min(n_samples, int(round(n_samples * keep_ratio / 100.0))))
    indices = rng.choice(n_samples, size=k, replace=False)
    mask = np.zeros(n_samples, dtype=np.uint8)
    mask[indices] = 1
    return mask


def _compute_herding_penalty_sum(
    mask: np.ndarray,
    labels: np.ndarray,
    feats: np.ndarray,
    full_mean: np.ndarray,
    full_var: np.ndarray,
    eps: float,
) -> float:
    selected = mask.astype(bool)
    penalty_sum = 0.0
    num_classes = full_mean.shape[0]
    for class_id in range(num_classes):
        cls_sel = selected & (labels == class_id)
        if int(np.sum(cls_sel)) <= 0:
            continue
        subset_mean = np.mean(feats[cls_sel], axis=0, dtype=np.float32)
        diff = subset_mean - full_mean[class_id]
        dist2 = float(np.dot(diff, diff))
        penalty_sum += dist2 / (float(full_var[class_id]) + eps)
    return float(penalty_sum)


def _compute_herding_penalty_avg(
    mask: np.ndarray,
    labels: np.ndarray,
    feats: np.ndarray,
    full_mean: np.ndarray,
    full_var: np.ndarray,
    eps: float,
) -> float:
    num_classes = full_mean.shape[0]
    penalty_sum = _compute_herding_penalty_sum(mask, labels, feats, full_mean, full_var, eps)
    return float(penalty_sum / float(num_classes))


def _format_table(title: str, dataset_name: str, krs: list[int], values: dict[str, dict[int, float]]) -> str:
    methods = ["random", "herding", "learned_topk", "learned_group", "naive_topk", "E2LN", "GraNd", "Forgetting"]
    rows = []
    for kr in krs:
        rows.append([
            str(kr),
            *(f"{values[m][kr]:.6f}" if np.isfinite(values[m][kr]) else "nan" for m in methods),
        ])

    headers = ["kr", *methods]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    lines = [f"\nDataset: {dataset_name} | {title}", fmt_row(headers), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    datasets_to_run = _parse_csv_strs(args.datasets)
    krs = _parse_csv_ints(args.kr)
    if not krs:
        raise ValueError("kr 列表不能为空")

    set_seed(args.seed)
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    for dataset_name in datasets_to_run:
        ds_plain = _build_dataset(dataset_name, transform=None)
        class_names = ds_plain.classes  # type: ignore[attr-defined]
        labels = np.asarray(ds_plain.targets, dtype=np.int64)
        n_samples = int(labels.shape[0])

        metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
        loader = DataLoader(
            _build_dataset(dataset_name, metric.extractor.preprocess),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        image_adapter, _, _ = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=metric.extractor.embed_dim,
            seed=args.seed,
            map_location=device,
        )
        image_adapter.to(device).eval()

        feats, _ = metric._encode_images(loader, image_adapter)
        feats_np = feats.detach().cpu().numpy().astype(np.float32) if isinstance(feats, torch.Tensor) else np.asarray(feats, dtype=np.float32)

        num_classes = len(class_names)
        full_mean = np.zeros((num_classes, feats_np.shape[1]), dtype=np.float32)
        full_var = np.zeros((num_classes,), dtype=np.float32)
        eps = 1e-8
        for class_id in range(num_classes):
            class_feats = feats_np[labels == class_id]
            if class_feats.shape[0] == 0:
                continue
            class_mean = np.mean(class_feats, axis=0, dtype=np.float32)
            diff = class_feats - class_mean
            sigma2 = float(np.mean(np.sum(diff * diff, axis=1)))
            full_mean[class_id] = class_mean
            full_var[class_id] = np.float32(max(sigma2, 0.0))

        methods = ["random", "herding", "learned_topk", "learned_group", "naive_topk", "E2LN", "GraNd", "Forgetting"]
        values_class_avg: dict[str, dict[int, float]] = {m: {} for m in methods}
        values_no_class_avg: dict[str, dict[int, float]] = {m: {} for m in methods}
        values_class_balance_raw: dict[str, dict[int, float]] = {m: {} for m in methods}
        rng = np.random.default_rng(args.random_seed)

        for kr in krs:
            random_mask = _random_mask(n_samples, kr, rng)
            random_target_size = int(np.sum(random_mask))
            values_class_avg["random"][kr] = _compute_herding_penalty_avg(random_mask, labels, feats_np, full_mean, full_var, eps)
            values_no_class_avg["random"][kr] = _compute_herding_penalty_sum(random_mask, labels, feats_np, full_mean, full_var, eps)
            values_class_balance_raw["random"][kr] = compute_balance_penalty(
                random_mask,
                labels,
                num_classes,
                random_target_size,
            )

            for mode in methods[1:]:
                mask_path = resolve_mask_path(mode, dataset_name, args.model_name, args.seed, kr)
                try:
                    mask = _load_mask(mask_path, n_samples)
                    target_size = int(np.sum(mask))
                    values_class_avg[mode][kr] = _compute_herding_penalty_avg(mask, labels, feats_np, full_mean, full_var, eps)
                    values_no_class_avg[mode][kr] = _compute_herding_penalty_sum(mask, labels, feats_np, full_mean, full_var, eps)
                    values_class_balance_raw[mode][kr] = compute_balance_penalty(
                        mask,
                        labels,
                        num_classes,
                        target_size,
                    )
                except Exception as exc:
                    print(f"[WARN] {dataset_name} kr={kr} mode={mode}: {exc}")
                    values_class_avg[mode][kr] = float("nan")
                    values_no_class_avg[mode][kr] = float("nan")
                    values_class_balance_raw[mode][kr] = float("nan")

        print(_format_table("按类别取均值", dataset_name, krs, values_class_avg))
        print(_format_table("不按类别取均值", dataset_name, krs, values_no_class_avg))
        print(_format_table("类别平衡修正项原始值", dataset_name, krs, values_class_balance_raw))


if __name__ == "__main__":
    main()
