"""Train a lightweight adapter on top of frozen CLIP image features."""
from __future__ import annotations
from utils.global_config import CONFIG
from utils.seed import parse_seed_list, set_seed
from model.adapter import AdapterMLP, CLIPFeatureExtractor

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


def build_dataloaders(
    dataset_name: str,
    preprocess,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, List[str]]:
    dataset_name = dataset_name.lower()
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("Only cifar10/cifar100 are currently supported.")

    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    train_set = dataset_cls(root=str(CONFIG.data_root), train=True, download=True, transform=preprocess)
    classes = list(train_set.classes)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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

    # 避免中文字符缺失导致的告警，将字体限定为通用无衬线字体
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].barh(range(5), topk_before.values.cpu().tolist(), color="#7FB3D5")
    axes[0].invert_yaxis()
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(top_labels_before)
    axes[0].set_xlabel("Probability")
    axes[0].set_title("Before Training (CLIP features)")

    axes[1].barh(range(5), topk_after.values.cpu().tolist(), color="#F5B041")
    axes[1].invert_yaxis()
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(top_labels_after)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("After Training (Adapter features)")

    fig.suptitle(
        (
            "Ground truth: {gt} | Top-1 accuracy before: {before:.2%}, "
            "after: {after:.2%}"
        ).format(
            gt=class_names[gt_label], before=baseline_acc, after=adapted_acc
        ),
        fontsize=12,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def train_adapter(args: argparse.Namespace, seed: int, multi_seed: bool) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = CLIPFeatureExtractor(args.clip_model, device=device)

    train_loader, class_names = build_dataloaders(
        args.dataset,
        clip_encoder.preprocess,
        args.batch_size,
        CONFIG.num_workers,
        seed,
    )
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_features = clip_encoder.encode_text(text_prompts)

    adapter = AdapterMLP(input_dim=clip_encoder.embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    CONFIG.ensure_data_dir()
    adapter_dir = CONFIG.ensure_adapter_dir(args.dataset)
    seed_suffix = f"_seed{seed}" if multi_seed else ""
    save_path = adapter_dir / f"adapter_{args.dataset}_{args.clip_model.replace('/', '-')}{seed_suffix}.pt"

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
    viz_path = (
        adapter_dir
        / f"adapter_{args.dataset}_{args.clip_model.replace('/', '-')}{seed_suffix}_viz.png"
    )
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
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG.default_batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive temperature")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension of adapter MLP")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant")
    parser.add_argument(
        "--seed",
        type=str,
        default=",".join(str(s) for s in CONFIG.exp_seeds),
        help="随机种子，支持单个整数或逗号分隔列表",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = parse_seed_list(args.seed)
    multi_seed = len(seeds) > 1
    for seed in seeds:
        train_adapter(args, seed, multi_seed)
