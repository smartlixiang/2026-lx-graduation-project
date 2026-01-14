"""
运行多样性覆盖度 (Div) 计算，并用更“可解释”的方式做 4 类可视化：

(1) 每类 Top/Bottom Div 样本拼图（直观看 Div 是否在挑“稀疏/覆盖”样本）
(2) 高 Div 样本 + 同类最近邻对照（看它到底“稀疏”还是“噪声/离群”）
(3) kNN 距离均值（k_distances）分布：总体直方图 + 各类箱线图（替代几乎必然平的 rank 直方图）
(4) 类内 2D 嵌入散点图：点颜色=Div（看高 Div 是否落在类内边缘/稀疏区域）

用法示例：
python test_diversity.py --data-root ./data --k 10 --chunk-size 1024
python test_diversity.py --adapter-path path/to/adapter.pt --embed-method pca
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP  # noqa: E402
from scoring import Div  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402


# -----------------------------
# args / utils
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试多样性覆盖度 (Div) 并输出更可解释的可视化结果")
    parser.add_argument("--data-root", type=str, default="./data", help="数据存放路径")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader 并行线程数")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument("--k", type=int, default=10, help="kNN 中的 k（前 k 个近邻距离均值）")
    parser.add_argument("--chunk-size", type=int, default=1024, help="相似度计算的分块大小")

    parser.add_argument("--adapter-path", type=str, default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt",
                        help="adapter 权重路径，留空则不使用 adapter")
    parser.add_argument("--device", type=str, default=None, help="计算设备，例如 cuda 或 cpu，默认跟随全局配置")

    # vis controls
    parser.add_argument("--vis-topk", type=int, default=16, help="(1) 每类 Top/Bottom 拼图各显示多少张")
    parser.add_argument("--neighbors-per-class", type=int, default=3, help="(2) 每类选多少个高 Div 样本做邻居对照")
    parser.add_argument("--neighbor-k", type=int, default=8, help="(2) 每个 anchor 显示多少个同类最近邻")
    parser.add_argument("--embed-method", type=str, default="pca", choices=["pca", "tsne", "umap"], help="(4) 2D 嵌入方法")
    parser.add_argument("--embed-max-per-class", type=int, default=2000, help="(4) 每类最多画多少点（太大很慢）")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录（默认 PROJECT_ROOT/test_Div_result）")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_adapter(adapter_path: Path, embed_dim: int, device: torch.device) -> AdapterMLP:
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter 权重: {adapter_path}")
    adapter = AdapterMLP(input_dim=embed_dim)
    state_dict = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()
    return adapter


def build_loader_preprocess(args: argparse.Namespace, preprocess, device: torch.device) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=preprocess,  # for CLIP
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: align indices with raw dataset
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def build_dataset_raw(args: argparse.Namespace) -> datasets.CIFAR10:
    # For visualization only (PIL image)
    return datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=None)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img)  # [0,1], CxHxW


def add_caption_below(img: Image.Image, caption: str, font: ImageFont.ImageFont | None = None) -> Image.Image:
    """Simple caption helper (optional)."""
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    pad = 6
    text_h = 18 if font is not None else 18
    w, h = img.size
    canvas = Image.new("RGB", (w, h + text_h + pad * 2), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, h + pad), caption, fill="black", font=font)
    return canvas


def save_grid_images(
    image_tensors: List[torch.Tensor],
    save_path: Path,
    nrow: int,
    pad_value: float = 1.0,
) -> None:
    if len(image_tensors) == 0:
        return
    grid = make_grid(image_tensors, nrow=nrow, padding=2, pad_value=pad_value)
    save_image(grid, save_path)


# -----------------------------
# core computations for neighbor / distances
# -----------------------------
@torch.no_grad()
def compute_class_knn_distances(
    class_features: torch.Tensor,
    anchor_local_idx: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given class_features [Nc, d] (assumed L2-normalized), compute nearest neighbors for one anchor.
    Returns (neighbor_local_indices, neighbor_distances) sorted by distance asc (excluding itself).
    """
    f = class_features
    a = f[anchor_local_idx: anchor_local_idx + 1]  # [1,d]
    sims = (a @ f.T).squeeze(0)  # [Nc]
    dists = torch.sqrt(torch.clamp(2.0 - 2.0 * sims, min=0.0))  # [Nc]
    dists[anchor_local_idx] = float("inf")
    k = min(topk, f.shape[0] - 1) if f.shape[0] > 1 else 0
    if k <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    vals, idxs = torch.topk(dists, k=k, largest=False)
    return idxs, vals


def compute_kdistance_all_with_div(div_metric: Div, loader: DataLoader, adapter: AdapterMLP | None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Use Div internals once:
    - encode all images -> image_features, labels
    - compute per-class kNN mean distances and quantile-normalized scores

    Returns: (scores_cpu, labels_cpu, k_distances_cpu, class_names)
    """
    # Use the existing implementation (returns cpu tensors and also includes image_features)
    result = div_metric.score_dataset(loader, adapter=adapter)
    return result.scores, result.labels, result.k_distances, result.class_names


# -----------------------------
# Visualizations (1)~(4)
# -----------------------------
def vis_top_bottom_per_class(
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    raw_dataset: datasets.CIFAR10,
    save_dir: Path,
    topk: int,
) -> None:
    """
    (1) 每类 Top/Bottom Div 样本拼图
    """
    out = save_dir / "vis_1_top_bottom"
    ensure_dir(out)

    for c, cname in enumerate(class_names):
        idxs = torch.where(labels == c)[0]
        if idxs.numel() == 0:
            continue

        cls_scores = scores[idxs]
        order = torch.argsort(cls_scores)  # asc

        bottom = idxs[order[: min(topk, order.numel())]].tolist()
        top = idxs[order[-min(topk, order.numel()):]].tolist()

        bottom_imgs = [pil_to_tensor(raw_dataset[i][0]) for i in bottom]
        top_imgs = [pil_to_tensor(raw_dataset[i][0]) for i in top]

        # grid size: try square-ish
        nrow = int(np.ceil(np.sqrt(topk)))
        save_grid_images(bottom_imgs, out / f"{c:02d}_{cname}_bottom_div.png", nrow=nrow)
        save_grid_images(top_imgs, out / f"{c:02d}_{cname}_top_div.png", nrow=nrow)


def vis_anchor_and_neighbors(
    div_metric: Div,
    scores: torch.Tensor,
    labels: torch.Tensor,
    k_distances: torch.Tensor,
    class_names: List[str],
    raw_dataset: datasets.CIFAR10,
    loader: DataLoader,
    adapter: AdapterMLP | None,
    save_dir: Path,
    anchors_per_class: int,
    neighbor_k: int,
) -> None:
    """
    (2) 高 Div 样本 + 同类最近邻对照

    说明：我们需要特征才能查最近邻。这里为了不改 scoring 代码，
    直接再次 encode 一遍（通常你跑 Div 本来就要 encode；这里重算一次可接受）。
    如果你希望“只算一次”，建议把 DivResult.image_features 保存下来并复用。
    """
    out = save_dir / "vis_2_neighbors"
    ensure_dir(out)

    # encode features once (on the device div_metric uses)
    # NOTE: _encode_images is "private" but用于分析脚本没问题
    with torch.no_grad():
        image_features, enc_labels = div_metric._encode_images(loader, adapter=adapter)  # type: ignore[attr-defined]

    # move to cpu for indexing
    image_features = image_features.detach().cpu()
    enc_labels = enc_labels.detach().cpu()

    # sanity: labels should match
    if enc_labels.shape == labels.shape and not torch.equal(enc_labels, labels):
        print("[WARN] encoded labels != result labels (unexpected). Proceeding with result labels.")

    font = ImageFont.load_default()

    for c, cname in enumerate(class_names):
        idxs = torch.where(labels == c)[0]
        if idxs.numel() <= 1:
            continue

        cls_scores = scores[idxs]
        # pick anchors among highest Div
        order_desc = torch.argsort(cls_scores, descending=True)
        anchor_globals = idxs[order_desc[: min(anchors_per_class, order_desc.numel())]].tolist()

        # build local feature array for this class
        cls_features = image_features[idxs]  # [Nc,d]

        # precompute mapping: global idx -> local idx
        global_to_local = {int(g): i for i, g in enumerate(idxs.tolist())}

        for a_i, gidx in enumerate(anchor_globals):
            local_anchor = global_to_local[int(gidx)]
            nn_local, nn_dist = compute_class_knn_distances(cls_features, local_anchor, topk=neighbor_k)

            # images: [anchor] + neighbors
            imgs: List[torch.Tensor] = [pil_to_tensor(raw_dataset[int(gidx)][0])]
            captions = [f"anchor Div={scores[gidx]:.3f}  d_mean={k_distances[gidx]:.3f}"]

            for j, (lidx, dist) in enumerate(zip(nn_local.tolist(), nn_dist.tolist())):
                ng = int(idxs[lidx].item())
                imgs.append(pil_to_tensor(raw_dataset[ng][0]))
                captions.append(f"nn{j + 1} Div={scores[ng]:.3f}  dist={dist:.3f}")

            # make a horizontal strip grid
            grid = make_grid(imgs, nrow=len(imgs), padding=2, pad_value=1.0)
            # convert to PIL for captioning
            grid_pil = transforms.ToPILImage()(grid)

            # one caption line (keep short; details go json)
            caption_line = f"{cname} | anchor={int(gidx)} | neighbor_k={neighbor_k}"
            grid_pil = add_caption_below(grid_pil, caption_line, font=font)

            save_path = out / f"{c:02d}_{cname}_anchor{a_i}_neighbors.png"
            grid_pil.save(save_path)

            # dump details
            detail = {
                "class": cname,
                "class_idx": c,
                "anchor_global_index": int(gidx),
                "anchor_div": float(scores[gidx]),
                "anchor_k_distance_mean": float(k_distances[gidx]),
                "neighbors": [],
            }
            for j, (lidx, dist) in enumerate(zip(nn_local.tolist(), nn_dist.tolist())):
                ng = int(idxs[lidx].item())
                detail["neighbors"].append(
                    {
                        "rank": j + 1,
                        "global_index": ng,
                        "div": float(scores[ng]),
                        "euclidean_dist": float(dist),
                    }
                )
            with open(out / f"{c:02d}_{cname}_anchor{a_i}_neighbors.json", "w", encoding="utf-8") as f:
                json.dump(detail, f, ensure_ascii=False, indent=2)


def vis_kdistance_distributions(
    k_distances: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_dir: Path,
) -> None:
    """
    (3) k_distances 分布：总体直方图 + 各类箱线图
    """
    out = save_dir / "vis_3_kdistance"
    ensure_dir(out)

    kd = k_distances.numpy()
    lb = labels.numpy()

    # overall histogram
    plt.figure()
    plt.hist(kd, bins=60)
    plt.title("kNN mean Euclidean distance distribution (all samples)")
    plt.xlabel("kNN mean distance")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out / "kdistance_hist_all.png", dpi=160)
    plt.close()

    # classwise boxplot (may be wide; still informative)
    data = []
    valid_names = []
    for c, cname in enumerate(class_names):
        vals = kd[lb == c]
        if vals.size == 0:
            continue
        data.append(vals)
        valid_names.append(cname)

    plt.figure(figsize=(max(10, len(valid_names) * 0.9), 5))
    plt.boxplot(data, showfliers=False)
    plt.title("kNN mean distance boxplot by class (fliers hidden)")
    plt.xlabel("class")
    plt.ylabel("kNN mean distance")
    plt.xticks(ticks=np.arange(1, len(valid_names) + 1), labels=valid_names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "kdistance_boxplot_by_class.png", dpi=160)
    plt.close()


def vis_embedding_scatter_per_class(
    div_metric: Div,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    loader: DataLoader,
    adapter: AdapterMLP | None,
    save_dir: Path,
    method: str,
    max_per_class: int,
    seed: int,
) -> None:
    """
    (4) 类内 2D 嵌入散点图：点颜色=Div
    - 默认 PCA：快，稳定，足够看“边缘/稀疏区域”现象
    - TSNE / UMAP：更强但更慢、依赖更多包
    """
    out = save_dir / "vis_4_embedding"
    ensure_dir(out)

    # encode features once
    with torch.no_grad():
        feats, enc_labels = div_metric._encode_images(loader, adapter=adapter)  # type: ignore[attr-defined]
    feats = feats.detach().cpu()
    enc_labels = enc_labels.detach().cpu()

    # pick embedder
    embedder = None
    if method == "pca":
        from sklearn.decomposition import PCA  # type: ignore

        def embed_fn(x: np.ndarray) -> np.ndarray:
            return PCA(n_components=2, random_state=seed).fit_transform(x)

        embedder = "PCA"
    elif method == "tsne":
        from sklearn.manifold import TSNE  # type: ignore

        def embed_fn(x: np.ndarray) -> np.ndarray:
            # TSNE is slow; keep it small
            return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(x)

        embedder = "t-SNE"
    else:  # umap
        try:
            import umap  # type: ignore

            def embed_fn(x: np.ndarray) -> np.ndarray:
                return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)

            embedder = "UMAP"
        except Exception as e:
            print(f"[WARN] umap-learn not available ({e}). Falling back to PCA.")
            from sklearn.decomposition import PCA  # type: ignore

            def embed_fn(x: np.ndarray) -> np.ndarray:
                return PCA(n_components=2, random_state=seed).fit_transform(x)

            embedder = "PCA(fallback)"

    for c, cname in enumerate(class_names):
        idxs = torch.where(labels == c)[0]
        if idxs.numel() < 5:
            continue

        # subsample for speed
        idx_list = idxs.tolist()
        if len(idx_list) > max_per_class:
            rng = random.Random(seed + c)
            idx_list = rng.sample(idx_list, k=max_per_class)

        X = feats[idx_list].numpy()
        y_color = scores[idx_list].numpy()

        Z = embed_fn(X)  # [n,2]

        plt.figure(figsize=(6, 5))
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=y_color, s=6)  # colormap default
        plt.colorbar(sc, label="Div score")
        plt.title(f"{embedder} embedding within class: {cname}")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.tight_layout()
        plt.savefig(out / f"{c:02d}_{cname}_{method}.png", dpi=160)
        plt.close()


# -----------------------------
# main
# -----------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "test_Div_result")
    ensure_dir(output_dir)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    adapter_path = Path(args.adapter_path) if args.adapter_path else None

    dataset_for_names = datasets.CIFAR10(root=args.data_root, train=True, download=True)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    div_metric = Div(
        class_names=class_names,
        k=args.k,
        clip_model=args.clip_model,
        device=device,
        chunk_size=args.chunk_size,
    )

    adapter = None
    resolved_adapter_path = None
    if adapter_path is not None:
        resolved_adapter_path = (PROJECT_ROOT / adapter_path).resolve()
        adapter = load_adapter(
            adapter_path=resolved_adapter_path,
            embed_dim=div_metric.extractor.embed_dim,
            device=device,
        )

    loader = build_loader_preprocess(args, preprocess=div_metric.extractor.preprocess, device=device)
    raw_dataset = build_dataset_raw(args)

    # compute div once
    start = time.perf_counter()
    scores, labels, k_distances, class_names = compute_kdistance_all_with_div(div_metric, loader, adapter)
    elapsed = time.perf_counter() - start

    # save tensors
    torch.save(scores, output_dir / "div_scores.pt")
    torch.save(labels, output_dir / "div_labels.pt")
    torch.save(k_distances, output_dir / "div_k_distances.pt")

    # write summary
    summary: Dict[str, object] = {
        "num_samples": int(scores.numel()),
        "div_score_mean": float(scores.mean()),
        "div_score_std": float(scores.std()),
        "div_score_min": float(scores.min()),
        "div_score_max": float(scores.max()),
        "kdist_mean": float(k_distances.mean()),
        "kdist_std": float(k_distances.std()),
        "kdist_min": float(k_distances.min()),
        "kdist_max": float(k_distances.max()),
        "elapsed_seconds": elapsed,
        "adapter_path": str(resolved_adapter_path) if resolved_adapter_path is not None else None,
        "device": str(device),
        "k": int(args.k),
        "chunk_size": int(args.chunk_size),
        "embed_method": args.embed_method,
        "seed": int(args.seed),
    }
    with open(output_dir / "div_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------------------------
    # (1) top/bottom per class
    # -------------------------
    vis_top_bottom_per_class(
        scores=scores,
        labels=labels,
        class_names=class_names,
        raw_dataset=raw_dataset,
        save_dir=output_dir,
        topk=args.vis_topk,
    )

    # -------------------------
    # (2) anchor + neighbors
    # -------------------------
    vis_anchor_and_neighbors(
        div_metric=div_metric,
        scores=scores,
        labels=labels,
        k_distances=k_distances,
        class_names=class_names,
        raw_dataset=raw_dataset,
        loader=loader,
        adapter=adapter,
        save_dir=output_dir,
        anchors_per_class=args.neighbors_per_class,
        neighbor_k=args.neighbor_k,
    )

    # -------------------------
    # (3) kNN mean distance distributions
    # -------------------------
    vis_kdistance_distributions(
        k_distances=k_distances,
        labels=labels,
        class_names=class_names,
        save_dir=output_dir,
    )

    # -------------------------
    # (4) embedding scatter per class
    # -------------------------
    vis_embedding_scatter_per_class(
        div_metric=div_metric,
        scores=scores,
        labels=labels,
        class_names=class_names,
        loader=loader,
        adapter=adapter,
        save_dir=output_dir,
        method=args.embed_method,
        max_per_class=args.embed_max_per_class,
        seed=args.seed,
    )

    print(f"[OK] Div 计算完成，用时 {elapsed:.2f}s")
    print(f"[OK] 输出目录: {output_dir}")
    print("[OK] 生成内容：")
    print("  - div_scores.pt / div_labels.pt / div_k_distances.pt / div_summary.json")
    print("  - vis_1_top_bottom/  (每类 top/bottom 拼图)")
    print("  - vis_2_neighbors/   (anchor + 最近邻对照 + json 细节)")
    print("  - vis_3_kdistance/   (kNN mean distance 总体直方图 + 各类箱线图)")
    print("  - vis_4_embedding/   (类内 2D 嵌入散点图，点颜色=Div)")


if __name__ == "__main__":
    main()
