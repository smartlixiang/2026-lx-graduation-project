"""Train CLIP adapters for a dataset (image + text)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

import clip  # type: ignore
from tqdm import tqdm

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100
from model.adapter import AdapterMLP, CLIPFeatureExtractor, resolve_adapter_dir
from utils.global_config import CONFIG
from utils.seed import parse_seed_list, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP dataset adapters.")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--data-root", type=str, default=str(CONFIG.data_root))
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {}")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    return parser.parse_args()


def _build_dataset(dataset_name: str, data_root: str, transform) -> datasets.VisionDataset:
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _default_batch_size(dataset_name: str) -> int:
    if dataset_name in (CIFAR10, CIFAR100):
        return 256
    return 128


def train_for_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    extractor = CLIPFeatureExtractor(model_name=args.clip_model, device=device)
    dataset = _build_dataset(args.dataset, args.data_root, extractor.preprocess)
    class_names = dataset.classes  # type: ignore[attr-defined]
    prompts = [args.prompt_template.format(name) for name in class_names]

    batch_size = args.batch_size or _default_batch_size(args.dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    image_adapter = AdapterMLP(input_dim=extractor.embed_dim, hidden_dim=args.hidden_dim).to(device)
    text_adapter = AdapterMLP(input_dim=extractor.embed_dim, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(image_adapter.parameters()) + list(text_adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    base_text_features = extractor.encode_text(prompts).to(device)
    loss_fn = nn.CrossEntropyLoss()

    image_adapter.train()
    text_adapter.train()
    start_time = time.perf_counter()

    progress_bar = tqdm(
        total=args.epochs * len(loader),
        desc=f"adapter-train (seed={seed})",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )
    for epoch in range(args.epochs):
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            image_features = extractor.encode_image(images)
            image_features = image_adapter(image_features.to(device))
            text_features = text_adapter(base_text_features)
            logits = (image_features @ text_features.T) / args.temperature
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.update(1)
            progress_bar.set_postfix(
                epoch=f"{epoch + 1}/{args.epochs}",
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        progress_bar.write(
            f"[seed={seed}] epoch {epoch + 1}/{args.epochs} "
            f"avg_loss={epoch_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}"
        )
    progress_bar.close()

    total_time = time.perf_counter() - start_time

    output_dir = resolve_adapter_dir(args.dataset, seed)
    image_path = output_dir / "adapter_image.pt"
    text_path = output_dir / "adapter_context.pt"
    torch.save(image_adapter.state_dict(), image_path)
    torch.save(text_adapter.state_dict(), text_path)

    meta = {
        "dataset": args.dataset,
        "seed": seed,
        "clip_model": args.clip_model,
        "clip_version": getattr(clip, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "prompt_template": args.prompt_template,
        "num_classes": len(class_names),
        "num_samples": len(dataset),
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "lr_scheduler": {
            "type": "StepLR",
            "step_size": args.step_size,
            "gamma": args.gamma,
        },
        "adapter_image_path": str(image_path),
        "adapter_text_path": str(text_path),
        "time_seconds": total_time,
    }
    meta_path = output_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if multi_seed:
        print(f"[seed={seed}] saved adapters to {output_dir}")


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        train_for_seed(args, seed, multi_seed)


if __name__ == "__main__":
    main()
