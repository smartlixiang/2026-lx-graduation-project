from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP  # noqa: E402
from dataset.dataset_config import CIFAR10  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import parse_seed_list, set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate selection masks (CIFAR-10)")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument(
        "--cr",
        type=str,
        default="80",
        help="cut_ratio 列表（百分比），支持逗号分隔或单值",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt",
        help="adapter 权重路径",
    )
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(CONFIG.global_seed),
        help="随机种子列表，逗号分隔",
    )
    parser.add_argument(
        "--weight-group",
        type=str,
        default="naive",
        help="scoring_weights.json 中的权重组名",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        help="mask 保存时使用的方法名（默认按 weight-group 映射为 my_naive 或 my_learned）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50",
        help="mask 保存路径中的模型名称",
    )
    return parser.parse_args()


def parse_ratio_list(ratio_text: str) -> list[int]:
    cleaned = ratio_text.strip()
    if not cleaned:
        return []
    if "," in cleaned:
        items = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        items = [cleaned]
    return [int(item) for item in items]


def ensure_scoring_weights(path: Path, dataset_name: str) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, dict[str, float]]] = {}
    updated = False
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
    dataset_entry = data.get(dataset_name)
    if not isinstance(dataset_entry, dict):
        dataset_entry = {}
        updated = True
    naive = dataset_entry.get("naive")
    if not isinstance(naive, dict):
        naive = {}
        updated = True
    for key in ("dds", "div", "sa"):
        if key not in naive:
            naive[key] = 1.0
            updated = True
    dataset_entry["naive"] = naive
    data[dataset_name] = dataset_entry
    if updated or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        group_name: group
        for group_name, group in dataset_entry.items()
        if isinstance(group, dict)
    }


def load_scoring_weights(
    all_weights: dict[str, dict[str, object]],
    group: str,
) -> dict[str, float]:
    selected = all_weights.get(group)
    if selected is None:
        raise KeyError(f"未找到权重组: {group}")
    if not isinstance(selected, dict):
        raise ValueError(f"权重组 {group} 格式不正确。")
    required = {"dds", "div", "sa"}
    missing = required - selected.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"权重组 {group} 缺少必要键: {missing_str}")
    weights: dict[str, float] = {}
    for key in sorted(required):
        value = selected[key]
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"权重组 {group} 的 {key} 无法转换为 float。") from exc
    return weights


def build_score_loader(
    preprocess, data_root: str, device: torch.device, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=preprocess)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def select_topk_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    cut_ratio: int,
) -> tuple[np.ndarray, dict[int, int]]:
    if cut_ratio <= 0 or cut_ratio > 100:
        raise ValueError("cr 必须在 1-100 之间。")
    mask = np.zeros(scores.shape[0], dtype=np.uint8)
    selected_by_class: dict[int, int] = {}
    ratio = cut_ratio / 100.0
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size == 0:
            selected_by_class[class_id] = 0
            continue
        if cut_ratio == 100:
            num_select = class_indices.size
        else:
            num_select = max(1, int(class_indices.size * ratio))
        class_scores = scores[class_indices]
        topk_indices = class_indices[
            np.argpartition(-class_scores, num_select - 1)[:num_select]
        ]
        mask[topk_indices] = 1
        selected_by_class[class_id] = int(num_select)
    return mask, selected_by_class


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, CIFAR10)
    weights = load_scoring_weights(all_weights, args.weight_group)

    dataset_for_names = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=None
    )
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]

    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    div_metric = Div(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
    )
    sa_metric = SemanticAlignment(
        class_names=class_names, clip_model=args.clip_model, device=device
    )

    adapter = None
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter 权重: {adapter_path}")
    adapter = AdapterMLP(input_dim=dds_metric.extractor.embed_dim)
    state_dict = torch.load(adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)
    adapter.to(device)
    adapter.eval()

    batch_size = 128
    num_workers = 8

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess, args.data_root, device, batch_size, num_workers
    )

    dds_start = time.perf_counter()
    num_samples = len(dataset_for_names)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(
            tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
            adapter=adapter,
        ).scores
        div_scores_local = div_metric.score_dataset(
            tqdm(div_loader, desc="Scoring Div", unit="batch"),
            adapter=adapter,
        ).scores
        sa_scores_local = sa_metric.score_dataset(
            tqdm(sa_loader, desc="Scoring SA", unit="batch"),
            adapter=adapter,
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
        clip_model=args.clip_model,
        adapter_path=str(adapter_path),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    dds_scores = torch.from_numpy(static_scores["dds"])
    div_scores = torch.from_numpy(static_scores["div"])
    sa_scores = torch.from_numpy(static_scores["sa"])
    dds_time = time.perf_counter() - dds_start
    div_time = dds_time
    sa_time = dds_time

    if not (len(dds_scores) == len(div_scores) == len(sa_scores)):
        raise RuntimeError("三个指标的样本数不一致，无法合并。")

    total_scores = (
        weights["dds"] * dds_scores
        + weights["div"] * div_scores
        + weights["sa"] * sa_scores
    )
    labels = np.asarray(dataset_for_names.targets)
    total_scores_np = np.asarray(total_scores)
    if args.method.strip():
        method_name = args.method.strip()
    elif args.weight_group == "naive":
        method_name = "my_naive"
    else:
        method_name = "my_learned"
    cut_ratios = parse_ratio_list(args.cr)
    if not cut_ratios:
        raise ValueError("cr 参数不能为空。")
    seeds = parse_seed_list(args.seeds)
    if method_name == "my_naive":
        save_seeds = [CONFIG.global_seed]
    else:
        save_seeds = seeds
    for cut_ratio in cut_ratios:
        mask, selected_by_class = select_topk_mask(
            total_scores_np, labels, num_classes=len(class_names), cut_ratio=cut_ratio
        )
        total_time = time.perf_counter() - total_start
        for seed in save_seeds:
            set_seed(seed)
            mask_dir = (
                PROJECT_ROOT
                / "mask"
                / method_name
                / "cifar10"
                / args.model_name
                / str(seed)
            )
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"mask_{cut_ratio}.npz"
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))

            meta_info = {
                "dataset": "cifar10",
                "model_name": args.model_name,
                "method": method_name,
                "weight_group": args.weight_group,
                "clip_model": args.clip_model,
                "adapter_path": str(Path(args.adapter_path)),
                "cr": cut_ratio,
                "num_samples": int(mask.shape[0]),
                "selected_count": int(mask.sum()),
                "selected_by_class": selected_by_class,
                "selection_strategy": "topk_per_class",
                "seeds": save_seeds,
                "timing": {
                    "dds_seconds": dds_time,
                    "div_seconds": div_time,
                    "sa_seconds": sa_time,
                    "total_seconds": total_time,
                },
            }
            with open(mask_dir / "meta_info.json", "w", encoding="utf-8") as f:
                json.dump(meta_info, f, ensure_ascii=False, indent=2)

            print(f"seed={seed} | cr={cut_ratio} | selected={int(mask.sum())}")
            print(f"mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
