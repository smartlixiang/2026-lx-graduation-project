"""Train a lightweight adapter on top of frozen CLIP image features."""
from __future__ import annotations

import argparse
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from model.adapter import AdapterMLP, CLIPFeatureExtractor
<<<<<<< ours
from utils.global import CONFIG
=======
from utils.global_config import CONFIG
>>>>>>> theirs


def build_dataloaders(dataset_name: str, preprocess, batch_size: int, num_workers: int) -> tuple[DataLoader, List[str]]:
    dataset_name = dataset_name.lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("Only cifar10/cifar100 are currently supported.")

    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    train_set = dataset_cls(root=str(CONFIG.data_root), train=True, download=True, transform=preprocess)
    classes = list(train_set.classes)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=CONFIG.pin_memory,
    )
    return train_loader, classes


def contrastive_loss(
    image_features: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = image_features @ text_features.t()
    logits = logits / temperature
    return nn.functional.cross_entropy(logits, labels)


def train_adapter(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = CLIPFeatureExtractor(args.clip_model, device=device)

    train_loader, class_names = build_dataloaders(
        args.dataset, clip_encoder.preprocess, args.batch_size, CONFIG.num_workers
    )
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_features = clip_encoder.encode_text(text_prompts)

    adapter = AdapterMLP(input_dim=clip_encoder.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    CONFIG.ensure_data_dir()
    adapter_dir = CONFIG.ensure_adapter_dir()
    save_path = adapter_dir / f"adapter_{args.dataset}_{args.clip_model.replace('/', '-')}.pt"

    adapter.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                image_features = clip_encoder.encode_image(images)
            adapted = adapter(image_features)
            loss = contrastive_loss(adapted, text_features, labels, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}")

    torch.save(adapter.state_dict(), save_path)
    print(f"Adapter weights saved to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP adapter with contrastive loss")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name: cifar10 or cifar100")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG.default_batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive temperature")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of adapter MLP")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_adapter(args)
