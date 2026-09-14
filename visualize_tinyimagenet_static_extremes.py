#!/usr/bin/env python3
"""
Visualize high- and low-scoring Tiny-ImageNet training samples.

The script:
1. Loads cached raw SA / Div / DDS(SV) scores.
2. Applies class-wise standard z-score to each static criterion.
3. Loads the learned criterion weights for the requested seed.
4. Computes the same static total score used by calculate_my_mask.py:
       Q = w_SA * z(SA) + w_Div * z(Div) + w_SV * z(SV)
5. Selects three classes and shows the top-3 / bottom-3 samples per class.
6. Saves a flat, paper-ready PDF and PNG.

Place this script anywhere inside the project repository and run it from
the repository environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets


DATASET_NAME = "tiny-imagenet"


def find_project_root(start: Path) -> Path:
    """Find the repository root from the script location or current directory."""
    candidates = [start.resolve(), Path.cwd().resolve()]
    checked: set[Path] = set()

    for candidate in candidates:
        for root in [candidate, *candidate.parents]:
            if root in checked:
                continue
            checked.add(root)
            if (
                (root / "weights" / "scoring_weights.json").exists()
                and (root / "utils" / "static_score_cache.py").exists()
                and (root / "data" / "tiny-imagenet-200").exists()
            ):
                return root

    raise FileNotFoundError(
        "Cannot locate the project root. Expected weights/scoring_weights.json, "
        "utils/static_score_cache.py, and data/tiny-imagenet-200."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Tiny-ImageNet high/low static-score samples."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed whose static-score cache and learned weights are used.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help=(
            "Optional comma-separated classes. Each item can be a numeric class id, "
            "Tiny-ImageNet WNID, or resolved English class name. "
            "Exactly three classes are required when this option is used."
        ),
    )
    parser.add_argument(
        "--class-seed",
        type=int,
        default=42,
        help="Seed for reproducibly choosing three classes when --classes is omitted.",
    )
    parser.add_argument(
        "--samples-per-side",
        type=int,
        default=3,
        help="Number of high-score and low-score samples shown per class.",
    )
    parser.add_argument(
        "--div-k",
        type=float,
        default=0.05,
        help="Div neighborhood setting used to resolve the static-score cache.",
    )
    parser.add_argument(
        "--sv-lower",
        type=float,
        default=0.02,
        help="Lower explained-variance ratio used by SV (DDS in cache naming).",
    )
    parser.add_argument(
        "--sv-upper",
        type=float,
        default=0.20,
        help="Upper explained-variance ratio used by SV (DDS in cache naming).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/tiny_imagenet_score_examples",
        help="Output path without extension. Both PDF and PNG are written.",
    )
    return parser.parse_args()


def load_metric_cache(cache_dir: Path, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_dir / f"{metric_name}_cache.npz"
    if not path.exists():
        raise FileNotFoundError(f"Static-score cache not found: {path}")

    data = np.load(path, allow_pickle=False)
    for key in ("scores", "labels", "indices"):
        if key not in data:
            raise KeyError(f"{path} does not contain required key '{key}'.")

    scores = np.asarray(data["scores"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    indices = np.asarray(data["indices"], dtype=np.int64)

    expected = np.arange(scores.shape[0], dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise ValueError(f"Unexpected sample ordering in {path}.")

    return scores, labels


def load_learned_weights(weights_path: Path, seed: int) -> dict[str, float]:
    with weights_path.open("r", encoding="utf-8") as f:
        all_weights = json.load(f)

    dataset_weights = all_weights.get(DATASET_NAME)
    if not isinstance(dataset_weights, dict):
        raise KeyError(f"No '{DATASET_NAME}' entry in {weights_path}.")

    entry = dataset_weights.get(str(seed))
    if not isinstance(entry, dict):
        available = [k for k in dataset_weights.keys() if k != "naive"]
        raise KeyError(
            f"No learned weights for seed={seed}. Available learned seeds: {available}"
        )

    # The implementation still stores Structural Variation under the historical
    # key "dds". In the paper and this script it is referred to as SV.
    weights = {
        "sa": float(entry["sa"]),
        "div": float(entry["div"]),
        "sv": float(entry["dds"]),
    }

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Learned weights must have a positive sum.")

    # Usually already normalized by the training code; normalize defensively.
    return {key: value / total for key, value in weights.items()}


def select_class_ids(
    classes_arg: str,
    class_seed: int,
    wnids: list[str],
    display_names: list[str],
) -> list[int]:
    num_classes = len(wnids)

    if not classes_arg.strip():
        rng = np.random.default_rng(class_seed)
        chosen = rng.choice(num_classes, size=3, replace=False)
        return sorted(int(x) for x in chosen)

    tokens = [item.strip() for item in classes_arg.split(",") if item.strip()]
    if len(tokens) != 3:
        raise ValueError("--classes must contain exactly three comma-separated items.")

    name_to_id = {name.casefold(): i for i, name in enumerate(display_names)}
    wnid_to_id = {wnid.casefold(): i for i, wnid in enumerate(wnids)}

    result: list[int] = []
    for token in tokens:
        if token.isdigit():
            class_id = int(token)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"Class id out of range: {class_id}")
        elif token.casefold() in wnid_to_id:
            class_id = wnid_to_id[token.casefold()]
        elif token.casefold() in name_to_id:
            class_id = name_to_id[token.casefold()]
        else:
            raise ValueError(
                f"Cannot resolve class '{token}'. Use a class id, WNID, "
                "or the first English name from Tiny-ImageNet words.txt."
            )
        result.append(class_id)

    if len(set(result)) != 3:
        raise ValueError("The three requested classes must be distinct.")

    return result


def main() -> None:
    args = parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import project utilities only after the root is known.
    from utils.class_name_utils import resolve_class_names_for_prompts
    from utils.score_utils import standard_zscore_by_class
    from utils.static_score_cache import resolve_static_score_cache_dir

    data_root = project_root / "data"
    train_root = data_root / "tiny-imagenet-200" / "train"
    weights_path = project_root / "weights" / "scoring_weights.json"

    dataset = datasets.ImageFolder(root=str(train_root), transform=None)
    wnids = list(dataset.classes)
    display_names = resolve_class_names_for_prompts(
        dataset_name=DATASET_NAME,
        data_root=data_root,
        class_names=wnids,
    )

    cache_dir = resolve_static_score_cache_dir(
        cache_root=project_root / "static_scores",
        dataset=DATASET_NAME,
        seed=args.seed,
        div_k=args.div_k,
        dds_eigval_lower_bound=args.sv_lower,
        dds_eigval_upper_bound=args.sv_upper,
    )

    sa_raw, labels_sa = load_metric_cache(cache_dir, "SA")
    div_raw, labels_div = load_metric_cache(cache_dir, "Div")
    sv_raw, labels_sv = load_metric_cache(cache_dir, "DDS")

    if not (
        np.array_equal(labels_sa, labels_div)
        and np.array_equal(labels_sa, labels_sv)
    ):
        raise ValueError("Labels in SA / Div / DDS caches are inconsistent.")

    labels = labels_sa
    dataset_targets = np.asarray(dataset.targets, dtype=np.int64)
    if not np.array_equal(labels, dataset_targets):
        raise ValueError(
            "Static-score cache ordering does not match the current Tiny-ImageNet "
            "ImageFolder ordering."
        )

    weights = load_learned_weights(weights_path, args.seed)

    # Match calculate_my_mask.py exactly:
    # each static criterion is z-scored within each class before weighting.
    sa_z = standard_zscore_by_class(sa_raw, labels)
    div_z = standard_zscore_by_class(div_raw, labels)
    sv_z = standard_zscore_by_class(sv_raw, labels)

    total_scores = (
        weights["sa"] * sa_z
        + weights["div"] * div_z
        + weights["sv"] * sv_z
    ).astype(np.float32)

    class_ids = select_class_ids(
        classes_arg=args.classes,
        class_seed=args.class_seed,
        wnids=wnids,
        display_names=display_names,
    )

    n = int(args.samples_per_side)
    if n <= 0:
        raise ValueError("--samples-per-side must be positive.")

    # Flat layout: class label | high x N | gap | low x N
    # For the requested 3+3 layout, 12.0 x 4.25 is intentionally wide/flat.
    fig = plt.figure(figsize=(12.0, 4.25))
    width_ratios = [1.55] + [1.0] * n + [0.32] + [1.0] * n
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2 * n + 2,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.40,
    )

    for row, class_id in enumerate(class_ids):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < 2 * n:
            raise ValueError(
                f"Class {class_id} has only {class_indices.size} samples; "
                f"at least {2 * n} are required."
            )

        class_scores = total_scores[class_indices]
        order = np.argsort(class_scores, kind="mergesort")

        low_indices = class_indices[order[:n]]
        high_indices = class_indices[order[-n:]][::-1]

        # Left-most class name.
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        readable_name = textwrap.fill(display_names[class_id], width=14)
        label_ax.text(
            0.96,
            0.50,
            readable_name,
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

        # High-score samples on the left.
        for col, sample_idx in enumerate(high_indices, start=1):
            ax = fig.add_subplot(gs[row, col])
            image, _ = dataset[int(sample_idx)]
            ax.imshow(image)
            ax.set_title(
                f"{total_scores[sample_idx]:+.2f}",
                fontsize=9,
                fontweight="bold",
                pad=2,
            )
            ax.axis("off")

        # Spacer column.
        gap_ax = fig.add_subplot(gs[row, n + 1])
        gap_ax.axis("off")

        # Low-score samples on the right.
        for offset, sample_idx in enumerate(low_indices):
            col = n + 2 + offset
            ax = fig.add_subplot(gs[row, col])
            image, _ = dataset[int(sample_idx)]
            ax.imshow(image)
            ax.set_title(
                f"{total_scores[sample_idx]:+.2f}",
                fontsize=9,
                fontweight="bold",
                pad=2,
            )
            ax.axis("off")

    # No global title, as intended for the paper figure.
    fig.subplots_adjust(left=0.015, right=0.995, top=0.96, bottom=0.025)

    output_base = Path(args.output)
    if not output_base.is_absolute():
        output_base = project_root / output_base
    output_base.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print("[Weights]")
    print(
        f"  SA={weights['sa']:.6f}, "
        f"Div={weights['div']:.6f}, "
        f"SV={weights['sv']:.6f}"
    )
    print("[Selected classes]")
    for class_id in class_ids:
        print(
            f"  id={class_id:3d} | wnid={wnids[class_id]} | "
            f"name={display_names[class_id]}"
        )
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")


if __name__ == "__main__":
    main()