from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.adapter import AdapterMLP  # noqa: E402
from dataset.dataset_config import CIFAR10  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.normalizer import NORMALIZER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scoring selection effectiveness (CIFAR-10)")
    parser.add_argument("--data-root", type=str, default="./data", help="数据根目录")
    parser.add_argument("--cr", type=int, default=80, help="cut_ratio (百分比)")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP 模型规格")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapter_weights/cifar10/adapter_cifar10_ViT-B-32.pt",
        help="adapter 权重路径",
    )
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="随机种子列表，逗号分隔")
    parser.add_argument(
        "--weight-group",
        type=str,
        default="naive",
        help="scoring_weights.json 中的权重组名",
    )
    parser.add_argument(
        "--div-cdf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Div 指标是否启用 CDF 修正",
    )
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
    return weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def select_indices(scores: torch.Tensor, cr: int) -> list[int]:
    num_samples = scores.numel()
    select_num = int(math.ceil(num_samples * cr / 100.0))
    select_num = max(1, min(select_num, num_samples))
    topk = torch.topk(scores, k=select_num, largest=True).indices
    return topk.cpu().tolist()


def train_one_seed(
    seed: int,
    train_dataset,
    test_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = models.resnet50(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        progress = tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch + 1}/200", unit="batch")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    if args.cr <= 0 or args.cr > 100:
        raise ValueError("cr 必须在 1-100 之间。")

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
        div_cdf=args.div_cdf,
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
    dds_scores = dds_metric.score_dataset(
        tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
        adapter=adapter,
    ).scores
    dds_time = time.perf_counter() - dds_start

    div_start = time.perf_counter()
    div_scores = div_metric.score_dataset(
        tqdm(div_loader, desc="Scoring Div", unit="batch"),
        adapter=adapter,
    ).scores
    div_time = time.perf_counter() - div_start

    sa_start = time.perf_counter()
    sa_scores = sa_metric.score_dataset(
        tqdm(sa_loader, desc="Scoring SA", unit="batch"),
        adapter=adapter,
    ).scores
    sa_time = time.perf_counter() - sa_start

    if not (len(dds_scores) == len(div_scores) == len(sa_scores)):
        raise RuntimeError("三个指标的样本数不一致，无法合并。")

    total_scores = (
        weights["dds"] * dds_scores
        + weights["div"] * div_scores
        + weights["sa"] * sa_scores
    )
    selected_indices = select_indices(total_scores, args.cr)

    train_tfms = NORMALIZER.train_tfms(CIFAR10)
    eval_tfms = NORMALIZER.eval_tfms(CIFAR10)
    full_train_dataset = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_tfms
    )
    train_dataset = Subset(full_train_dataset, selected_indices)
    test_dataset = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=eval_tfms
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    accuracies: list[float] = []
    train_times: list[float] = []
    for seed in seeds:
        train_start = time.perf_counter()
        acc = train_one_seed(
            seed, train_dataset, test_dataset, device, batch_size, num_workers
        )
        train_time = time.perf_counter() - train_start
        print(f"seed={seed} train_time={train_time:.2f}s")
        accuracies.append(acc)
        train_times.append(train_time)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    total_time = time.perf_counter() - total_start

    result_dir = PROJECT_ROOT / "test_scoring_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.weight_group}_scoring_cr{args.cr}.json"
    result_payload = {
        "cr": args.cr,
        "selected_count": len(selected_indices),
        "weight_group": args.weight_group,
        "seeds": seeds,
        "accuracies": accuracies,
        "mean": mean_acc,
        "std": std_acc,
        "timing": {
            "dds_seconds": dds_time,
            "div_seconds": div_time,
            "sa_seconds": sa_time,
            "train_seconds": train_times,
            "total_seconds": total_time,
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    print(f"cr={args.cr} | selected={len(selected_indices)}")
    for seed, acc in zip(seeds, accuracies):
        print(f"seed={seed} acc={acc:.4f}")
    print(f"mean={mean_acc:.4f} std={std_acc:.4f}")
    print(
        "timing: "
        f"dds={dds_time:.2f}s "
        f"div={div_time:.2f}s "
        f"sa={sa_time:.2f}s "
        f"total={total_time:.2f}s"
    )
    print(f"结果保存至: {result_path}")


if __name__ == "__main__":
    main()
