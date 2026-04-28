from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets

from utils.global_config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load cached SA/Div/DDS scores only and visualize the highest/lowest "
            "samples of Tiny-ImageNet class 0 for each metric."
        )
    )
    parser.add_argument("--data-root", type=str, default=str(CONFIG.data_root))
    parser.add_argument("--dataset", type=str, default="tiny-imagenet", choices=["tiny-imagenet"])
    parser.add_argument(
        "--seed",
        type=int,
        default=96,
        help="Seed of the static score cache to load.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=128,
        help="Target class id in ImageFolder order (0-based).",
    )
    parser.add_argument(
        "--static-cache-root",
        type=str,
        default="static_scores",
        help="Root directory of cached static scores.",
    )
    parser.add_argument(
        "--div-k",
        type=float,
        default=0.05,
        help="Must match the local cached Div setting.",
    )
    parser.add_argument(
        "--dds-eigval-lower-bound",
        type=float,
        default=0.02,
        help="Must match the local cached DDS lower bound.",
    )
    parser.add_argument(
        "--dds-eigval-upper-bound",
        type=float,
        default=0.20,
        help="Must match the local cached DDS upper bound.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="picture/compare_static_extremes_tiny_imagenet_class0.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="Figure DPI. Use a high value for a high-resolution figure.",
    )
    return parser.parse_args()


def _format_percent(value: float) -> str:
    return f"{int(round(value * 100))}%"


def _format_div_k(div_k: float) -> str:
    if float(div_k).is_integer() and div_k >= 1:
        return str(int(div_k))
    if 0 < div_k < 1:
        return _format_percent(div_k)
    raise ValueError("div_k must be a positive integer or a ratio in (0,1).")


def build_cache_dir(
    cache_root: Path,
    dataset: str,
    seed: int,
    div_k: float,
    dds_lower: float,
    dds_upper: float,
) -> Path:
    param_dir = f"Div_{_format_div_k(div_k)}_DDS_[{_format_percent(dds_lower)}-{_format_percent(dds_upper)}]"
    return cache_root / dataset / str(int(seed)) / param_dir


def load_metric_cache(cache_dir: Path, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"{metric_name}_cache.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required cache file not found: {cache_path}\n"
            f"This script loads cached static scores only and will not recompute them."
        )

    data = np.load(cache_path, allow_pickle=False)
    if "scores" not in data or "labels" not in data:
        raise ValueError(f"Invalid cache file: {cache_path}. Expected 'scores' and 'labels' arrays.")

    scores = np.asarray(data["scores"])
    labels = np.asarray(data["labels"])

    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Invalid shapes in {cache_path}: scores.shape={scores.shape}, labels.shape={labels.shape}"
        )
    return scores, labels


def build_raw_dataset(data_root: str) -> datasets.ImageFolder:
    train_root = Path(data_root) / "tiny-imagenet-200" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Tiny-ImageNet train split not found: {train_root}")
    return datasets.ImageFolder(root=str(train_root), transform=None)


def pil_to_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # Defensive fallback, though ImageFolder with transform=None should already return PIL.
    return Image.fromarray(np.asarray(img)).convert("RGB")


def select_high_low(scores: np.ndarray, labels: np.ndarray, class_id: int) -> tuple[int, float, int, float]:
    class_indices = np.where(labels == class_id)[0]
    if class_indices.size == 0:
        raise ValueError(f"No samples found for class_id={class_id}.")

    class_scores = scores[class_indices]

    high_local = int(np.argmax(class_scores))
    low_local = int(np.argmin(class_scores))

    high_index = int(class_indices[high_local])
    low_index = int(class_indices[low_local])
    high_score = float(class_scores[high_local])
    low_score = float(class_scores[low_local])

    return high_index, high_score, low_index, low_score


def make_figure(
    dataset: datasets.ImageFolder,
    class_id: int,
    class_name: str,
    metric_results: list[dict[str, object]],
    output_path: Path,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), dpi=dpi)

    for row, result in enumerate(metric_results):
        metric_name = str(result["metric"])
        high_index = int(result["high_index"])
        low_index = int(result["low_index"])
        high_score = float(result["high_score"])
        low_score = float(result["low_score"])

        high_img, high_label = dataset[high_index]
        low_img, low_label = dataset[low_index]

        if int(high_label) != class_id or int(low_label) != class_id:
            raise RuntimeError(
                f"Selected sample does not belong to class_id={class_id} for metric {metric_name}."
            )

        high_img = pil_to_rgb(high_img)
        low_img = pil_to_rgb(low_img)

        ax_left = axes[row, 0]
        ax_right = axes[row, 1]

        ax_left.imshow(high_img)
        ax_left.axis("off")
        ax_left.set_title(
            f"{metric_name} high\nindex={high_index}, score={high_score:.6f}",
            fontsize=14,
            pad=10,
        )

        ax_right.imshow(low_img)
        ax_right.axis("off")
        ax_right.set_title(
            f"{metric_name} low\nindex={low_index}, score={low_score:.6f}",
            fontsize=14,
            pad=10,
        )

    fig.suptitle(
        f"Tiny-ImageNet Class {class_id} Comparison in Cached Static Metrics\n"
        f"Class name: {class_name}",
        fontsize=20,
        y=0.985,
    )

    legend_text = (
        "Figure meaning: each row corresponds to one cached static metric (SA, Div, DDS) for the same "
        f"Tiny-ImageNet class {class_id}. The left column shows the highest-scoring sample under that metric, "
        "and the right column shows the lowest-scoring sample. This comparison is used to directly inspect "
        "what visual patterns each metric tends to reward or suppress within the same class."
    )

    fig.text(
        0.5,
        0.02,
        legend_text,
        ha="center",
        va="bottom",
        fontsize=12,
        wrap=True,
    )

    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    cache_root = PROJECT_ROOT / args.static_cache_root
    cache_dir = build_cache_dir(
        cache_root=cache_root,
        dataset=args.dataset,
        seed=args.seed,
        div_k=args.div_k,
        dds_lower=args.dds_eigval_lower_bound,
        dds_upper=args.dds_eigval_upper_bound,
    )

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Static cache directory not found: {cache_dir}\n"
            f"This script loads cached static scores only and will not recompute them."
        )

    sa_scores, sa_labels = load_metric_cache(cache_dir, "SA")
    div_scores, div_labels = load_metric_cache(cache_dir, "Div")
    dds_scores, dds_labels = load_metric_cache(cache_dir, "DDS")

    if not (np.array_equal(sa_labels, div_labels) and np.array_equal(sa_labels, dds_labels)):
        raise RuntimeError("Label arrays in SA/Div/DDS cache files are inconsistent.")

    raw_dataset = build_raw_dataset(args.data_root)
    class_names = list(raw_dataset.classes)

    if not (0 <= args.class_id < len(class_names)):
        raise ValueError(
            f"class_id={args.class_id} out of range. Valid range is [0, {len(class_names) - 1}]."
        )

    if len(raw_dataset) != sa_scores.shape[0]:
        raise RuntimeError(
            f"Dataset size mismatch: len(dataset)={len(raw_dataset)}, len(cache)={sa_scores.shape[0]}"
        )

    class_name = class_names[args.class_id]

    metric_results: list[dict[str, object]] = []

    for metric_name, scores in (
        ("SA", sa_scores),
        ("Div", div_scores),
        ("DDS", dds_scores),
    ):
        high_index, high_score, low_index, low_score = select_high_low(
            scores=scores,
            labels=sa_labels,
            class_id=args.class_id,
        )
        metric_results.append(
            {
                "metric": metric_name,
                "high_index": high_index,
                "high_score": high_score,
                "low_index": low_index,
                "low_score": low_score,
            }
        )

    print(f"Using cache directory: {cache_dir}")
    print(f"Target class: id={args.class_id}, name={class_name}")
    for item in metric_results:
        print(
            f"{item['metric']}: "
            f"high(index={item['high_index']}, score={item['high_score']:.6f}), "
            f"low(index={item['low_index']}, score={item['low_score']:.6f})"
        )

    output_path = PROJECT_ROOT / args.output
    make_figure(
        dataset=raw_dataset,
        class_id=args.class_id,
        class_name=class_name,
        metric_results=metric_results,
        output_path=output_path,
        dpi=args.dpi,
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
