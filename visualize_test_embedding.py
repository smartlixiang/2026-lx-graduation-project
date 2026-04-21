from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import BaseDataLoader
from model.model_config import get_model
from utils.global_config import CONFIG
from utils.path_rules import resolve_checkpoint_path
from utils.seed import set_seed


DATASET_NAME = "cifar10"
MODEL_NAME = "resnet50"
KEEP_RATIO = 50
FIXED_SEED = 22
METHODS = ["random", "MoSo", "YangCLIP", "learned_group"]
OUTPUT_PATH = Path("picture") / "cifar10_test_embedding_tsne_kr50.png"


@torch.no_grad()
def extract_penultimate_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    progress = tqdm(loader, desc="Extracting test embeddings", unit="batch")
    for images, labels in progress:
        images = images.to(device)

        # ResNet50 penultimate feature:
        # global average pooling 后、最终 fc 分类层之前的特征
        x = model.relu(model.bn1(model.conv1(images)))
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        all_features.append(x.detach().cpu().numpy().astype(np.float32))
        all_labels.append(np.asarray(labels, dtype=np.int64))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return features / norms


def fit_tsne(features: np.ndarray) -> np.ndarray:
    # CIFAR-10 test set 有 10000 个样本，这里采用较稳妥的参数
    reducer = TSNE(
        n_components=2,
        perplexity=30.0,
        init="pca",
        learning_rate="auto",
        random_state=FIXED_SEED,
    )
    coords = reducer.fit_transform(features)
    return coords.astype(np.float32)


def pairwise_max_distance(points: np.ndarray, chunk_size: int = 512) -> float:
    n = points.shape[0]
    if n <= 1:
        return 0.0

    max_dist = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]
        diff = chunk[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        current = float(np.sqrt(np.max(dist2)))
        if current > max_dist:
            max_dist = current
    return max_dist


def pairwise_min_distance(a: np.ndarray, b: np.ndarray, chunk_size: int = 512) -> float:
    min_dist = math.inf
    for start in range(0, a.shape[0], chunk_size):
        end = min(start + chunk_size, a.shape[0])
        chunk = a[start:end]
        diff = chunk[:, None, :] - b[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        current = float(np.sqrt(np.min(dist2)))
        if current < min_dist:
            min_dist = current
    return min_dist


def compute_dunn_index(coords: np.ndarray, labels: np.ndarray) -> float:
    class_ids = np.unique(labels)
    clusters = [coords[labels == class_id] for class_id in class_ids]

    max_intra = 0.0
    for cluster in clusters:
        intra = pairwise_max_distance(cluster)
        if intra > max_intra:
            max_intra = intra

    if max_intra <= 1e-12:
        return 0.0

    min_inter = math.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            inter = pairwise_min_distance(clusters[i], clusters[j])
            if inter < min_inter:
                min_inter = inter

    if not np.isfinite(min_inter):
        return 0.0
    return float(min_inter / max_intra)


def load_model_checkpoint(method: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = resolve_checkpoint_path(
        mode=method,
        dataset=DATASET_NAME,
        model=MODEL_NAME,
        seed=FIXED_SEED,
        keep_ratio=KEEP_RATIO,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for method='{method}': {checkpoint_path}\n"
            "Please run train_cifar10_checkpoint.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(MODEL_NAME)(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    set_seed(FIXED_SEED)
    device = CONFIG.global_device

    data_loader = BaseDataLoader(
        DATASET_NAME,
        data_path=Path("data"),
        batch_size=256,
        num_workers=CONFIG.num_workers,
        val_split=0.0,
        seed=FIXED_SEED,
    )
    _, _, test_loader = data_loader.load()

    class_names = list(test_loader.dataset.classes)  # type: ignore[attr-defined]
    cmap = plt.get_cmap("tab10")

    panel_results: dict[str, dict[str, np.ndarray | float]] = {}

    for method in METHODS:
        print(f"\n[Load] method={method}")
        model = load_model_checkpoint(method, device)

        features, labels = extract_penultimate_features(model, test_loader, device)
        features = l2_normalize(features)

        print(f"[t-SNE] method={method}")
        coords = fit_tsne(features)

        print(f"[Dunn] method={method}")
        dunn_index = compute_dunn_index(coords, labels)

        panel_results[method] = {
            "coords": coords,
            "labels": labels,
            "dunn_index": dunn_index,
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=220)
    axes = axes.flatten()

    for ax, method in zip(axes, METHODS):
        coords = panel_results[method]["coords"]  # type: ignore[index]
        labels = panel_results[method]["labels"]  # type: ignore[index]
        dunn_index = panel_results[method]["dunn_index"]  # type: ignore[index]

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap=cmap,
            s=5.0,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_title(
            f"{method}\nDunn Index = {float(dunn_index):.4e}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    handles = []
    for class_id, class_name in enumerate(class_names):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=class_name,
                markerfacecolor=cmap(class_id),
                markersize=7,
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )

    fig.suptitle(
        "CIFAR-10 test-set t-SNE of ResNet-50 embeddings trained on selected subsets (kr=50)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.985,
        top=0.91,
        bottom=0.11,
        wspace=0.10,
        hspace=0.18,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
