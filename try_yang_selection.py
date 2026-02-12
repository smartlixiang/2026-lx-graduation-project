from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402

from calculate_my_mask import (  # noqa: E402
    build_score_loader,
    ensure_scoring_weights,
    load_scoring_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--weight-group",
        type=str,
        default="42",
        help="使用 scoring_weights.json 中的哪个权重组，如 naive / my_learned 等",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cut-ratios",
        type=int,
        nargs="+",
        default=[20, 30, 40, 50, 60, 70, 80, 90],
        help="裁剪比例列表，比如 20 30 40 表示 20%/30%/40%",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin + 1e-12:
        return np.ones_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + 1e-12)


def yang_optimize_mask(
    sim_scores: np.ndarray,
    dis_scores: np.ndarray,
    sr: float,
    lambda_: float = 0.1,
    beta_: float = 2.0,
    lr: float = 1e-3,
    num_epochs: int = 100_000,
    scale_factor: float = 100.0,
    tol: float = 1e-3,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    使用 YangCLIP 官方 optimize_selection.py 风格的损失，对每个样本的 w 做优化，
    最终返回长度 N 的 {0,1} mask（np.uint8），1 表示被选择。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = torch.from_numpy(sim_scores).to(device=device, dtype=torch.float32)
    dis = torch.from_numpy(dis_scores).to(device=device, dtype=torch.float32)
    n = sim.shape[0]
    k = int(round(sr * n))

    w = torch.zeros(n, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

    sim_norm = sim / (sim.mean() + 1e-12)
    dis_norm = dis / (dis.mean() + 1e-12)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        x = torch.sigmoid(scale_factor * w)

        loss1 = -(x * sim_norm).mean()
        loss2 = -(x * dis_norm).mean() * lambda_

        x_hard = (x > 0.5).float()
        ste_x = x_hard - x.detach() + x
        card_diff = (ste_x.sum() - float(k)) / float(n)
        loss3 = torch.sqrt(card_diff * card_diff + 1e-8) * beta_

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

        if loss3.item() < tol:
            break

    with torch.no_grad():
        x = torch.sigmoid(scale_factor * w)
        mask = (x > 0.5).to(torch.uint8).cpu().numpy()
    return mask


def topk_mask_from_scores(
    sim_scores: np.ndarray,
    dis_scores: np.ndarray,
    sr: float,
    lambda_: float = 0.1,
) -> np.ndarray:
    n = sim_scores.shape[0]
    k = int(round(sr * n))
    sim_norm = sim_scores / (sim_scores.mean() + 1e-12)
    dis_norm = dis_scores / (dis_scores.mean() + 1e-12)

    combined = sim_norm + lambda_ * dis_norm
    order = np.argsort(-combined)
    mask = np.zeros(n, dtype=np.uint8)
    mask[order[:k]] = 1
    return mask


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, CIFAR10)
    weights = load_scoring_weights(all_weights, args.weight_group)

    dataset_for_names = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=None
    )
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    labels = np.asarray(dataset_for_names.targets)
    num_samples = labels.shape[0]

    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )

    batch_size = 128
    num_workers = 4
    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )

    def resolve_adapter_and_num_samples() -> tuple[
        torch.nn.Module,
        torch.nn.Module,
        int,
        dict[str, Path],
        int,
    ]:
        adapter_seed = args.seed
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=CIFAR10,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=adapter_seed,
            map_location=device,
            adapter_image_path=None,
            adapter_text_path=None,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()
        return image_adapter, text_adapter, adapter_seed, adapter_paths, num_samples

    image_adapter, text_adapter, adapter_seed, adapter_paths, num_samples = (
        resolve_adapter_and_num_samples()
    )

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=image_adapter,
        ).scores
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=image_adapter,
        ).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter_image=image_adapter,
            adapter_text=text_adapter,
        ).scores
        return {
            "sa": np.asarray(sa_scores_local),
            "div": np.asarray(div_scores_local),
            "dds": np.asarray(dds_scores_local),
            "labels": np.asarray(dataset_for_names.targets),
        }

    static_scores = get_or_compute_static_scores(
        cache_root=PROJECT_ROOT / "static_scores",
        dataset=CIFAR10,
        seed=adapter_seed,
        clip_model=args.clip_model,
        adapter_image_path=str(adapter_paths["image_path"]),
        adapter_text_path=str(adapter_paths["text_path"]),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
        dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    sa_scores = np.asarray(static_scores["sa"], dtype=np.float32)
    div_scores = np.asarray(static_scores["div"], dtype=np.float32)
    dds_scores = np.asarray(static_scores["dds"], dtype=np.float32)
    labels = np.asarray(static_scores["labels"], dtype=np.int64)
    num_samples = labels.shape[0]

    sim_scores = normalize(sa_scores.copy())
    dis_scores = normalize(
        (weights["div"] * div_scores + weights["dds"] * dds_scores).astype(np.float32)
    )

    print(
        "seed | cr | yang_selected | topk_selected | intersection | union | jaccard | overlap_yang | overlap_topk"
    )
    print(
        "-----+----+--------------+--------------+-------------+-------+---------+---------------+--------------"
    )

    lambda_yang = 0.1
    beta_yang = 2.0

    for cr in args.cut_ratios:
        sr = cr / 100.0

        yang_mask = yang_optimize_mask(
            sim_scores=sim_scores,
            dis_scores=dis_scores,
            sr=sr,
            lambda_=lambda_yang,
            beta_=beta_yang,
            lr=1e-3,
            num_epochs=100_000,
            scale_factor=100.0,
            tol=1e-3,
            device=device,
        )

        topk_mask = topk_mask_from_scores(
            sim_scores=sim_scores,
            dis_scores=dis_scores,
            sr=sr,
            lambda_=lambda_yang,
        )

        assert yang_mask.shape == topk_mask.shape == (num_samples,)

        yang_selected = int(yang_mask.sum())
        topk_selected = int(topk_mask.sum())
        intersection = int(np.logical_and(yang_mask == 1, topk_mask == 1).sum())
        union = int(np.logical_or(yang_mask == 1, topk_mask == 1).sum())
        jaccard = intersection / union if union > 0 else 0.0
        overlap_yang = intersection / yang_selected if yang_selected > 0 else 0.0
        overlap_topk = intersection / topk_selected if topk_selected > 0 else 0.0

        print(
            f"{args.seed:4d} | {cr:2d} | "
            f"{yang_selected:12d} | {topk_selected:12d} | "
            f"{intersection:11d} | {union:5d} | "
            f"{jaccard:7.4f} | {overlap_yang:13.4f} | {overlap_topk:12.4f}"
        )


if __name__ == "__main__":
    main()
