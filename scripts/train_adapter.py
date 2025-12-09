"""Train a lightweight adapter on top of frozen CLIP image features."""
from __future__ import annotations

import argparse
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP, CLIPFeatureExtractor
from utils.global_config import CONFIG


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


def visualize_adapter_effects(
    clip_encoder: CLIPFeatureExtractor,
    adapter: AdapterMLP,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    class_names: List[str],
    temperature: float,
    save_path: Path,
) -> None:
    """生成单样本 top-5 置信度对比图，直观展示适配器训练前后的预测差异。"""

    adapter_device = next(adapter.parameters()).device
    adapter.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images.to(adapter_device)
        labels = labels.to(adapter_device)

        image_features = clip_encoder.encode_image(images)
        adapted_features = adapter(image_features)

        logits_before = (image_features @ text_features.t()) / temperature
        logits_after = (adapted_features @ text_features.t()) / temperature
        probs_before = logits_before.softmax(dim=-1)
        probs_after = logits_after.softmax(dim=-1)

        baseline_pred = probs_before.argmax(dim=-1)
        adapted_pred = probs_after.argmax(dim=-1)
        baseline_acc = (baseline_pred == labels).float().mean().item()
        adapted_acc = (adapted_pred == labels).float().mean().item()

        sample_idx = 0
        gt_label = labels[sample_idx].item()

        topk_before = torch.topk(probs_before[sample_idx], k=5)
        topk_after = torch.topk(probs_after[sample_idx], k=5)

        top_labels_before = [class_names[i] for i in topk_before.indices.tolist()]
        top_labels_after = [class_names[i] for i in topk_after.indices.tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(range(5), topk_before.values.cpu().tolist(), color="#7FB3D5")
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(top_labels_before)
    axes[0].set_xlabel("概率")
    axes[0].set_title("训练前 (CLIP 原始特征)")

    axes[1].barh(range(5), topk_after.values.cpu().tolist(), color="#F5B041")
    axes[1].invert_yaxis()
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(top_labels_after)
    axes[1].set_xlabel("概率")
    axes[1].set_title("训练后 (Adapter 特征)")

    fig.suptitle(
        f"样本真实标签: {class_names[gt_label]} | 训练前 Top-1 准确率: {baseline_acc:.2%}, 训练后: {adapted_acc:.2%}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


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
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                image_features = clip_encoder.encode_image(images)
            adapted = adapter(image_features)
            loss = contrastive_loss(adapted, text_features, labels, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * images.size(0)
            total_loss += batch_loss
            progress.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(train_loader.dataset)
        progress.write(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f}")

    torch.save(adapter.state_dict(), save_path)
    print(f"Adapter weights saved to {save_path}")

    viz_loader = DataLoader(
        train_loader.dataset,
        batch_size=min(8, args.batch_size),
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=CONFIG.pin_memory,
    )
    viz_path = adapter_dir / f"adapter_{args.dataset}_{args.clip_model.replace('/', '-')}_viz.png"
    visualize_adapter_effects(
        clip_encoder,
        adapter,
        viz_loader,
        text_features,
        class_names,
        args.temperature,
        viz_path,
    )
    print(f"适配器训练效果对比图已保存至 {viz_path}")


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
