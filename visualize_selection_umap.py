from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import AVAILABLE_DATASETS, CIFAR10, CIFAR100, TINY_IMAGENET  # noqa: E402
from model.adapter import CLIPFeatureExtractor, load_trained_adapters  # noqa: E402
from train_after_selection import select_random_indices_by_class  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the full training set and selected subsets in a shared 2D manifold space "
            "using CLIP+adapter features."
        )
    )
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=AVAILABLE_DATASETS)
    parser.add_argument("--seed", type=int, default=22, help="Selection seed / adapter seed.")
    parser.add_argument("--kr", type=int, default=50, help="Keep ratio in percent.")
    parser.add_argument(
        "--methods",
        type=str,
        default="random,learned_topk,learned_group",
        help="Comma-separated methods to compare. 'random' is generated on the fly.",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help=(
            "Maximum number of points used to fit and draw the 2D manifold. "
            "All selected samples are kept; the rest are class-balanced subsampled."
        ),
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="umap",
        choices=["umap", "tsne", "pca"],
        help="Preferred 2D reducer. If umap is unavailable, the script falls back automatically.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--alpha-background",
        type=float,
        default=0.15,
        help="Alpha for the gray full-dataset background points.",
    )
    parser.add_argument(
        "--point-size-background",
        type=float,
        default=6.0,
        help="Marker size for background points.",
    )
    parser.add_argument(
        "--point-size-selected",
        type=float,
        default=10.0,
        help="Marker size for selected points.",
    )
    parser.add_argument(
        "--show-class-legend",
        action="store_true",
        help="Show class legend. Recommended mainly for CIFAR-10.",
    )
    parser.add_argument(
        "--save-embedding",
        action="store_true",
        help="Save the computed 2D coordinates as an .npz file next to the figure.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path. Default: result/visualization/<dataset>/seed_<seed>/umap_kr<kr>.png",
    )
    return parser.parse_args()


# ---------- dataset / mask utilities ----------

def _build_dataset(dataset_name: str, transform) -> datasets.VisionDataset:
    data_root = PROJECT_ROOT / "data"
    if dataset_name == CIFAR10:
        return datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=transform)
    if dataset_name == CIFAR100:
        return datasets.CIFAR100(root=str(data_root), train=True, download=False, transform=transform)
    if dataset_name == TINY_IMAGENET:
        train_root = data_root / "tiny-imagenet-200" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"tiny-imagenet train split not found: {train_root}")
        return datasets.ImageFolder(root=str(train_root), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_labels(dataset: torch.utils.data.Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.asarray([dataset[idx][1] for idx in range(len(dataset))])


def _load_mask(
    dataset_name: str,
    method: str,
    keep_ratio: int,
    seed: int,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    method = method.strip()
    if method == "random":
        selected_indices = select_random_indices_by_class(
            labels=labels,
            num_classes=num_classes,
            keep_ratio=keep_ratio,
            seed=seed,
        )
        mask = np.zeros(labels.shape[0], dtype=bool)
        mask[selected_indices] = True
        return mask

    mask_path = resolve_mask_path(
        mode=method,
        dataset=dataset_name,
        model="resnet50",
        seed=seed,
        keep_ratio=keep_ratio,
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found for method='{method}': {mask_path}")
    with np.load(mask_path) as data:
        if "mask" in data:
            raw = data["mask"]
        elif len(data.files) == 1:
            raw = data[data.files[0]]
        else:
            raise ValueError(f"Unrecognized mask format: {mask_path}")
    mask = np.asarray(raw).astype(bool)
    if mask.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mask length mismatch for method='{method}': got {mask.shape[0]}, expected {labels.shape[0]}"
        )
    return mask


# ---------- feature extraction ----------

def _build_loader(dataset_name: str, preprocess, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    dataset = _build_dataset(dataset_name, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def _extract_adapter_features(
    loader: DataLoader,
    extractor: CLIPFeatureExtractor,
    image_adapter: torch.nn.Module,
) -> tuple[np.ndarray, np.ndarray]:
    image_adapter.eval()
    adapter_device = next(image_adapter.parameters()).device
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, batch_labels in loader:
        feats = extractor.encode_image(images)
        feats = image_adapter(feats.to(adapter_device))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.detach().cpu().numpy().astype(np.float32))
        all_labels.append(np.asarray(batch_labels, dtype=np.int64))
    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


# ---------- sampling / reduction ----------

def _class_balanced_subsample_indices(
    labels: np.ndarray,
    must_keep: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    n = labels.shape[0]
    all_indices = np.arange(n, dtype=np.int64)
    if max_points <= 0 or max_points >= n:
        return all_indices

    must_keep = np.unique(np.asarray(must_keep, dtype=np.int64))
    if must_keep.size >= max_points:
        return np.sort(must_keep)

    rng = np.random.default_rng(seed)
    remaining_budget = max_points - must_keep.size
    keep_set = set(must_keep.tolist())

    sampled_extra: list[int] = []
    classes = np.unique(labels)
    class_pool: dict[int, np.ndarray] = {}
    for c in classes:
        idx = np.flatnonzero(labels == c)
        idx = idx[~np.isin(idx, must_keep)]
        class_pool[int(c)] = idx

    while remaining_budget > 0:
        progressed = False
        for c in classes:
            if remaining_budget <= 0:
                break
            pool = class_pool[int(c)]
            if pool.size == 0:
                continue
            choice_pos = int(rng.integers(0, pool.size))
            sampled_extra.append(int(pool[choice_pos]))
            pool = np.delete(pool, choice_pos)
            class_pool[int(c)] = pool
            remaining_budget -= 1
            progressed = True
            if remaining_budget <= 0:
                break
        if not progressed:
            break

    selected = np.sort(np.concatenate([must_keep, np.asarray(sampled_extra, dtype=np.int64)]))
    return selected


def _fit_reduce(features: np.ndarray, reducer_name: str, seed: int, args: argparse.Namespace):
    reducer_name = reducer_name.lower()
    if reducer_name == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric="cosine",
                random_state=seed,
                transform_seed=seed,
            )
            coords = reducer.fit_transform(features)
            return coords.astype(np.float32), "umap"
        except Exception:
            reducer_name = "tsne"

    if reducer_name == "tsne":
        try:
            from sklearn.manifold import TSNE

            perplexity = min(float(args.tsne_perplexity), max(5.0, (features.shape[0] - 1) / 3.0))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=seed,
            )
            coords = reducer.fit_transform(features)
            return coords.astype(np.float32), "tsne"
        except Exception:
            reducer_name = "pca"

    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2, random_state=seed)
    coords = reducer.fit_transform(features)
    return coords.astype(np.float32), "pca"


# ---------- plotting ----------

def _get_class_names(dataset) -> list[str]:
    if hasattr(dataset, "classes"):
        return [str(x) for x in dataset.classes]
    return [str(i) for i in np.unique(_extract_labels(dataset)).tolist()]


def _make_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    return (
        PROJECT_ROOT
        / "result"
        / "visualization"
        / args.dataset
        / f"seed_{args.seed}"
        / f"{args.reducer}_kr{args.kr}.png"
    )


def _plot_panels(
    coords: np.ndarray,
    labels: np.ndarray,
    display_indices: np.ndarray,
    methods: list[str],
    masks: dict[str, np.ndarray],
    class_names: list[str],
    reducer_used: str,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    display_labels = labels[display_indices]
    display_coords = coords
    num_classes = len(class_names)
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")

    ncols = len(methods) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    # Panel 0: full dataset colored by label.
    ax = axes[0]
    scatter = ax.scatter(
        display_coords[:, 0],
        display_coords[:, 1],
        c=display_labels,
        s=args.point_size_background,
        cmap=cmap,
        alpha=0.65,
        linewidths=0,
    )
    ax.set_title(f"Full Train Set\n{args.dataset}")
    ax.set_xticks([])
    ax.set_yticks([])

    for panel_idx, method in enumerate(methods, start=1):
        ax = axes[panel_idx]
        selected_display = masks[method][display_indices]
        # background: all points in gray
        ax.scatter(
            display_coords[:, 0],
            display_coords[:, 1],
            c="lightgray",
            s=args.point_size_background,
            alpha=args.alpha_background,
            linewidths=0,
        )
        # overlay: selected points colored by label
        if np.any(selected_display):
            ax.scatter(
                display_coords[selected_display, 0],
                display_coords[selected_display, 1],
                c=display_labels[selected_display],
                s=args.point_size_selected,
                cmap=cmap,
                alpha=0.95,
                linewidths=0,
            )
        sel_num = int(masks[method].sum())
        ax.set_title(f"{method}\nselected={sel_num}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Shared 2D {reducer_used.upper()} manifold of CLIP+adapter features | seed={args.seed} | keep_ratio={args.kr}%",
        fontsize=13,
    )

    if args.show_class_legend and num_classes <= 20:
        handles: list[Line2D] = []
        for class_id, class_name in enumerate(class_names):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=class_name,
                    markerfacecolor=cmap(class_id % cmap.N),
                    markersize=6,
                )
            )
        fig.legend(handles=handles, loc="lower center", ncol=min(5, num_classes), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else CONFIG.global_device
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one method is required.")

    probe_dataset = _build_dataset(args.dataset, transform=None)
    labels = _extract_labels(probe_dataset)
    class_names = _get_class_names(probe_dataset)
    num_classes = len(class_names)

    extractor = CLIPFeatureExtractor(model_name=args.clip_model, device=device)
    loader = _build_loader(
        dataset_name=args.dataset,
        preprocess=extractor.preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    image_adapter, _, adapter_paths = load_trained_adapters(
        dataset_name=args.dataset,
        clip_model=args.clip_model,
        input_dim=extractor.embed_dim,
        seed=args.seed,
        map_location=device,
    )
    image_adapter = image_adapter.to(device).eval()

    print(f"[1/4] Extracting CLIP+adapter features from {args.dataset} ...")
    features, feature_labels = _extract_adapter_features(loader, extractor, image_adapter)
    if not np.array_equal(labels, feature_labels):
        raise RuntimeError("Label order mismatch between dataset probe and feature extraction loader.")

    print(f"[2/4] Loading/generating masks for methods: {methods}")
    masks: dict[str, np.ndarray] = {}
    must_keep = np.array([], dtype=np.int64)
    for method in methods:
        mask = _load_mask(
            dataset_name=args.dataset,
            method=method,
            keep_ratio=args.kr,
            seed=args.seed,
            labels=labels,
            num_classes=num_classes,
        )
        masks[method] = mask
        must_keep = np.union1d(must_keep, np.flatnonzero(mask))

    print(f"[3/4] Building shared 2D manifold input set (max_points={args.max_points}) ...")
    display_indices = _class_balanced_subsample_indices(
        labels=labels,
        must_keep=must_keep,
        max_points=args.max_points,
        seed=args.seed,
    )
    display_features = features[display_indices]

    print(f"[4/4] Fitting reducer ({args.reducer}) on {display_features.shape[0]} points ...")
    coords, reducer_used = _fit_reduce(display_features, args.reducer, args.seed, args)

    output_path = _make_output_path(args)
    _plot_panels(
        coords=coords,
        labels=labels,
        display_indices=display_indices,
        methods=methods,
        masks=masks,
        class_names=class_names,
        reducer_used=reducer_used,
        args=args,
        output_path=output_path,
    )

    if args.save_embedding:
        npz_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            coords=coords.astype(np.float32),
            display_indices=display_indices.astype(np.int64),
            labels=labels[display_indices].astype(np.int64),
            methods=np.asarray(methods),
            reducer=np.asarray(reducer_used),
            adapter_image_path=np.asarray(str(adapter_paths["image_path"])),
        )
        print(f"Embedding saved to: {npz_path}")

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
