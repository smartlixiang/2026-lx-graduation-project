from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_config import CIFAR10  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw histograms of CIFAR-10 static scoring distributions."
    )
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_scoring_debug",
        help="输出目录",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-image-path",
        type=str,
        default=None,
        help="图像 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument(
        "--adapter-text-path",
        type=str,
        default=None,
        help="文本 adapter 权重路径（默认按 dataset/seed 规则）",
    )
    parser.add_argument(
        "--adapter-seed",
        type=int,
        default=CONFIG.global_seed,
        help="adapter 训练随机种子",
    )
    parser.add_argument(
        "--weight-seed",
        type=str,
        default="22",
        help="学习权重的随机种子（scoring_weights.json 键名）",
    )
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--bins", type=int, default=80, help="直方图 bins 数量")
    return parser.parse_args()


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
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"权重组 {group} 的权重和必须为正数。")
    if not np.isclose(total, 1.0):
        weights = {key: value / total for key, value in weights.items()}
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


def main() -> None:
    args = parse_args()
    set_seed(22)

    device = torch.device(args.device) if args.device is not None else CONFIG.global_device
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, CIFAR10)
    naive_weights = load_scoring_weights(all_weights, "naive")
    learned_weights = load_scoring_weights(all_weights, str(args.weight_seed))

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

    image_adapter, text_adapter, adapter_paths = load_trained_adapters(
        dataset_name=CIFAR10,
        clip_model=args.clip_model,
        input_dim=dds_metric.extractor.embed_dim,
        seed=args.adapter_seed,
        map_location=device,
        adapter_image_path=args.adapter_image_path,
        adapter_text_path=args.adapter_text_path,
    )
    image_adapter.to(device).eval()
    text_adapter.to(device).eval()

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

    num_samples = len(dataset_for_names)

    def _compute_scores() -> dict[str, np.ndarray]:
        dds_scores_local = dds_metric.score_dataset(dds_loader, adapter=image_adapter).scores
        div_scores_local = div_metric.score_dataset(div_loader, adapter=image_adapter).scores
        sa_scores_local = sa_metric.score_dataset(
            sa_loader, adapter_image=image_adapter, adapter_text=text_adapter
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
        adapter_image_path=str(adapter_paths["image_path"]),
        adapter_text_path=str(adapter_paths["text_path"]),
        div_k=div_metric.k,
        dds_k=dds_metric.k,
        prompt_template=sa_metric.prompt_template,
        num_samples=num_samples,
        compute_fn=_compute_scores,
    )

    dds_scores = torch.from_numpy(static_scores["dds"])
    div_scores = torch.from_numpy(static_scores["div"])
    sa_scores = torch.from_numpy(static_scores["sa"])

    if not (len(dds_scores) == len(div_scores) == len(sa_scores)):
        raise RuntimeError("三个指标的样本数不一致，无法合并。")

    total_naive = (
        naive_weights["dds"] * dds_scores
        + naive_weights["div"] * div_scores
        + naive_weights["sa"] * sa_scores
    )
    total_learned = (
        learned_weights["dds"] * dds_scores
        + learned_weights["div"] * div_scores
        + learned_weights["sa"] * sa_scores
    )

    score_sets = {
        "SA": np.asarray(sa_scores),
        "Div": np.asarray(div_scores),
        "DDS": np.asarray(dds_scores),
        "Naive Total": np.asarray(total_naive),
        f"Learned Total (seed {args.weight_seed})": np.asarray(total_learned),
    }

    for name, scores in score_sets.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores, bins=args.bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xlim(0.0, 1.0)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        output_path = output_dir / f"cifar10_{safe_name}_hist_seed_{args.weight_seed}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved histogram to: {output_path}")


if __name__ == "__main__":
    main()
