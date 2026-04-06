"""Visualize dominant-direction (IDS-style) selection for the legacy DDS component."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET
from model.adapter import AdapterMLP, load_trained_adapters
from scoring import DifficultyDirection
from utils.global_config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and visualize PCA direction selection for the third static component "
            "(legacy DDS naming, IDS-style dominant-direction implementation)."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, choices=AVAILABLE_DATASETS)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--adapter-image-path", type=str, default=None)
    parser.add_argument("--adapter-text-path", type=str, default=None)
    parser.add_argument("--dds-important-eigval-ratio", type=float, default=0.5)
    return parser.parse_args()


def build_dataset(dataset_name: str, data_root: str, transform):
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = Path(data_root) / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_image_adapter(
    metric: DifficultyDirection,
    dataset_name: str,
    clip_model: str,
    seed: int,
    device: torch.device,
    adapter_image_path: str | None,
    adapter_text_path: str | None,
) -> AdapterMLP:
    image_adapter, _, _ = load_trained_adapters(
        dataset_name=dataset_name,
        clip_model=clip_model,
        input_dim=metric.extractor.embed_dim,
        seed=seed,
        map_location=device,
        adapter_image_path=adapter_image_path,
        adapter_text_path=adapter_text_path,
    )
    image_adapter = image_adapter.to(device).eval()
    return image_adapter  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else CONFIG.global_device

    metric_bootstrap = DifficultyDirection(
        class_names=["bootstrap"],
        clip_model=args.clip_model,
        device=device,
        important_eigval_ratio=args.dds_important_eigval_ratio,
    )

    dataset = build_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        transform=metric_bootstrap.extractor.preprocess,
    )
    class_names = [str(name) for name in dataset.classes]  # type: ignore[attr-defined]

    if not (0 <= args.class_id < len(class_names)):
        raise ValueError(
            f"class_id {args.class_id} out of range. Valid range: [0, {len(class_names) - 1}]"
        )

    metric = DifficultyDirection(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        important_eigval_ratio=args.dds_important_eigval_ratio,
    )

    image_adapter = load_image_adapter(
        metric=metric,
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        seed=args.seed,
        device=device,
        adapter_image_path=args.adapter_image_path,
        adapter_text_path=args.adapter_text_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    image_features, labels = metric._encode_images(loader, adapter=image_adapter)
    class_mask = labels == args.class_id
    if not class_mask.any():
        raise RuntimeError(f"No samples found for class_id={args.class_id}.")

    class_features = image_features[class_mask]
    if class_features.shape[0] <= 1:
        raise RuntimeError(
            f"class_id={args.class_id} has only {class_features.shape[0]} sample(s); PCA requires >= 2."
        )

    analysis = metric.analyze_principal_directions(class_features)
    eigvals = analysis["eigenvalues_desc"].detach().cpu().numpy()
    total_directions = int(analysis["total_directions"])
    selected_directions = int(analysis["selected_directions"])
    threshold = float(analysis["important_eigval_ratio"])
    selected_ratio = float(analysis["selected_ratio"])
    total_eig_sum = float(analysis["total_eigval_sum"])

    output = args.output or f"result/{args.dataset}_class{args.class_id}_ids_eigvals.png"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = range(1, total_directions + 1)
    plt.figure(figsize=(12, 5))
    plt.bar(x, eigvals, color="#4472C4", width=0.9)
    plt.axvline(
        x=selected_directions + 0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"selection cutoff ({selected_directions})",
    )
    plt.xlabel("Direction index (descending eigenvalue order)")
    plt.ylabel("Eigenvalue")
    plt.title(
        f"IDS Direction Selection | {args.dataset} | class {args.class_id} | "
        f"total={total_directions} | selected={selected_directions}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Dataset: {args.dataset}")
    print(f"Class ID: {args.class_id}")
    print(f"Total directions: {total_directions}")
    print(f"Selected directions: {selected_directions}")
    print(f"Important variance ratio threshold: {threshold:.2f}")
    print(f"Actual cumulative variance ratio: {selected_ratio:.4f}")
    if total_eig_sum <= 1e-12:
        print("Warning: total eigenvalue sum is near zero (degenerate spectrum).")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
