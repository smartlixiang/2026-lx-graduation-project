from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calculate_my_mask import (  # noqa: E402
    _build_dataset,
    _get_or_compute_group_mean_stats,
    _mean_stats_cache_path,
    build_score_loader,
    ensure_scoring_weights,
    load_scoring_weights,
)
from dataset.dataset_config import CIFAR10, CIFAR100  # noqa: E402
from model.adapter import load_trained_adapters  # noqa: E402
from scoring import DifficultyDirection, Div, SemanticAlignment  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.group_lambda import compute_balance_penalty  # noqa: E402
from utils.path_rules import resolve_mask_path  # noqa: E402
from utils.static_score_cache import get_or_compute_static_scores  # noqa: E402


@dataclass
class EvalResult:
    raw_score: float
    class_penalty: float
    mean_penalty: float
    adjusted_score: float


def parse_int_csv(text: str) -> list[int]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [int(x) for x in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare subset scores for multiple selection methods.")
    parser.add_argument("--dataset", type=str, default=CIFAR10, choices=[CIFAR10, CIFAR100])
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--seeds", type=str, default="22,42,96")
    parser.add_argument("--krs", type=str, default="20,30,40,50,60")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lambda-M", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="test")
    return parser.parse_args()


def estimate_lambda_local(
    *,
    seed: int,
    n_samples: int,
    target_size: int,
    eval_score_fn,
    penalty_fn,
    mean_penalty_fn,
    M: int = 100,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    注意：这里是 test.py 的实验版 lambda 估计。
    与 utils/group_lambda.py 的正式逻辑不同：
    - 仅使用 std 比值，不使用 min(std, mean)
    - 不使用 dataset beta
    - 两项统一 0.4 缩放
    """
    s_values: list[float] = []
    pen_values: list[float] = []
    mean_pen_values: list[float] = []

    for sample_idx in tqdm(range(1, M + 1), desc=f"Lambda sampling (seed={seed})", leave=False):
        sample_seed = int(seed) * sample_idx
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(n_samples, size=target_size, replace=False)
        mask = np.zeros(n_samples, dtype=np.uint8)
        mask[idx] = 1
        s_values.append(float(eval_score_fn(mask)))
        pen_values.append(float(penalty_fn(mask)))
        mean_pen_values.append(float(mean_penalty_fn(mask)))

    sigma_s = float(np.std(np.asarray(s_values, dtype=np.float64)))
    sigma_pen = float(np.std(np.asarray(pen_values, dtype=np.float64)))
    sigma_mean_pen = float(np.std(np.asarray(mean_pen_values, dtype=np.float64)))

    lambda_cls = float(0.4 * sigma_s / (sigma_pen + eps))
    lambda_mean = float(0.4 * sigma_s / (sigma_mean_pen + eps))

    return {
        "lambda_cls": lambda_cls,
        "lambda_mean": lambda_mean,
        "sigma_S": sigma_s,
        "sigma_pen": sigma_pen,
        "sigma_mean_pen": sigma_mean_pen,
    }


def load_mask_or_raise(dataset: str, model_name: str, seed: int, kr: int, method: str) -> np.ndarray:
    mode_map = {
        "herding": "herding",
        "learned_topk": "learned_topk",
        "learned_group": "learned_group",
    }
    mode = mode_map[method]
    mask_path = resolve_mask_path(
        mode=mode,
        dataset=dataset,
        model=model_name,
        seed=seed,
        keep_ratio=kr,
    )
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: dataset={dataset}, seed={seed}, kr={kr}, method={method}, path={mask_path}"
        )
    loaded = np.load(mask_path, allow_pickle=False)
    if "mask" not in loaded:
        raise KeyError(f"Mask file missing 'mask' key: {mask_path}")
    return np.asarray(loaded["mask"], dtype=np.uint8)


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset.strip().lower()
    seeds = parse_int_csv(args.seeds)
    krs = parse_int_csv(args.krs)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else CONFIG.global_device

    dataset_for_names = _build_dataset(dataset_name, transform=None)
    class_names = dataset_for_names.classes  # type: ignore[attr-defined]
    labels_np = np.asarray(dataset_for_names.targets, dtype=np.int64)
    num_classes = len(class_names)
    n_samples = int(labels_np.shape[0])

    dds_metric = DifficultyDirection(class_names=class_names, clip_model=args.clip_model, device=device)
    div_metric = Div(class_names=class_names, clip_model=args.clip_model, device=device)
    sa_metric = SemanticAlignment(class_names=class_names, clip_model=args.clip_model, device=device)

    dds_loader = build_score_loader(
        dds_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    div_loader = build_score_loader(
        div_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )
    sa_loader = build_score_loader(
        sa_metric.extractor.preprocess,
        dataset_name,
        device,
        args.batch_size,
        args.num_workers,
    )

    weights_path = PROJECT_ROOT / "weights" / "scoring_weights.json"
    all_weights = ensure_scoring_weights(weights_path, dataset_name)

    methods = ["random", "herding", "learned_topk", "learned_group"]

    for kr in krs:
        per_method_seed_metrics: dict[str, list[EvalResult]] = {m: [] for m in methods}
        lambdas: list[tuple[float, float]] = []

        for seed in seeds:
            weights = load_scoring_weights(all_weights, "learned", seed)
            image_adapter, text_adapter, adapter_paths = load_trained_adapters(
                dataset_name=dataset_name,
                clip_model=args.clip_model,
                input_dim=dds_metric.extractor.embed_dim,
                seed=seed,
                map_location=device,
            )
            image_adapter.to(device).eval()
            text_adapter.to(device).eval()

            def _compute_scores() -> dict[str, np.ndarray]:
                dds_scores_local = dds_metric.score_dataset(
                    tqdm(dds_loader, desc=f"Scoring DDS (seed={seed})", unit="batch", leave=False),
                    adapter=image_adapter,
                ).scores
                div_scores_local = div_metric.score_dataset(
                    tqdm(div_loader, desc=f"Scoring Div (seed={seed})", unit="batch", leave=False),
                    adapter=image_adapter,
                ).scores
                sa_scores_local = sa_metric.score_dataset(
                    tqdm(sa_loader, desc=f"Scoring SA (seed={seed})", unit="batch", leave=False),
                    adapter_image=image_adapter,
                    adapter_text=text_adapter,
                ).scores
                return {
                    "sa": np.asarray(sa_scores_local),
                    "div": np.asarray(div_scores_local),
                    "dds": np.asarray(dds_scores_local),
                    "labels": labels_np,
                }

            static_scores = get_or_compute_static_scores(
                cache_root=PROJECT_ROOT / "static_scores",
                dataset=dataset_name,
                seed=seed,
                clip_model=args.clip_model,
                adapter_image_path=str(adapter_paths["image_path"]),
                adapter_text_path=str(adapter_paths["text_path"]),
                div_k=div_metric.k,
                dds_k=dds_metric.k,
                dds_eigval_lower_bound=dds_metric.eigval_lower_bound,
                dds_eigval_upper_bound=dds_metric.eigval_upper_bound,
                prompt_template=sa_metric.prompt_template,
                num_samples=n_samples,
                compute_fn=_compute_scores,
            )
            sa_scores_np = np.asarray(static_scores["sa"], dtype=np.float32)

            labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
            div_features_t, _ = div_metric._encode_images(div_loader, image_adapter)
            dds_features_t, _ = dds_metric._encode_images(dds_loader, image_adapter)
            div_features_np = (
                div_features_t.detach().cpu().numpy() if isinstance(div_features_t, torch.Tensor) else np.asarray(div_features_t)
            ).astype(np.float32)

            mean_stats_cache_path = _mean_stats_cache_path(
                dataset_name=dataset_name,
                clip_model=args.clip_model,
                adapter_image_path=str(adapter_paths["image_path"]),
            )
            full_class_mean, full_class_var = _get_or_compute_group_mean_stats(
                cache_path=mean_stats_cache_path,
                image_features=div_features_np,
                labels=labels_np,
                num_classes=num_classes,
            )
            class_indices_list = [np.flatnonzero(labels_np == class_id).astype(np.int64) for class_id in range(num_classes)]
            class_features_list = [div_features_np[class_indices] for class_indices in class_indices_list]

            target_size = int(round((kr / 100.0) * n_samples))
            target_size = min(n_samples, max(1, target_size))

            cache: dict[bytes, tuple[float, float, float, float]] = {}

            def mean_penalty_fn(mask: np.ndarray) -> float:
                selected = mask.astype(bool)
                penalty_sum = 0.0
                for class_id in range(num_classes):
                    class_indices = class_indices_list[class_id]
                    if class_indices.size == 0:
                        continue
                    class_selected = selected[class_indices]
                    count = int(np.sum(class_selected))
                    if count <= 0:
                        continue
                    subset_sum = np.sum(class_features_list[class_id][class_selected], axis=0, dtype=np.float32)
                    subset_mean = subset_sum / float(count)
                    diff = subset_mean - full_class_mean[class_id]
                    dist2 = float(np.dot(diff, diff))
                    penalty_sum += dist2 / (float(full_class_var[class_id]) + 1e-8)
                return float(penalty_sum)

            def evaluate(mask: np.ndarray, lambda_cls: float, lambda_mean: float) -> EvalResult:
                key = np.asarray(mask, dtype=np.uint8).tobytes()
                if key in cache:
                    raw_score, class_pen, mean_pen, _ = cache[key]
                    adjusted = raw_score - lambda_cls * class_pen - lambda_mean * mean_pen
                    return EvalResult(raw_score, class_pen, mean_pen, adjusted)

                div_dyn = np.asarray(
                    div_metric.score_dataset_dynamic(
                        div_loader,
                        adapter=image_adapter,
                        selected_mask=mask,
                        image_features=div_features_t,
                        labels=labels_t,
                    ).scores,
                    dtype=np.float32,
                )
                dds_dyn = np.asarray(
                    dds_metric.score_dataset_dynamic(
                        dds_loader,
                        adapter=image_adapter,
                        selected_mask=mask,
                        image_features=dds_features_t,
                        labels=labels_t,
                    ).scores,
                    dtype=np.float32,
                )
                merged = weights["sa"] * sa_scores_np + weights["div"] * div_dyn + weights["dds"] * dds_dyn
                raw_score = float(np.sum(merged[mask.astype(bool)]))
                class_pen = float(compute_balance_penalty(mask, labels_np, num_classes, target_size))
                mean_pen = float(mean_penalty_fn(mask))
                adjusted = raw_score - lambda_cls * class_pen - lambda_mean * mean_pen
                cache[key] = (raw_score, class_pen, mean_pen, adjusted)
                return EvalResult(raw_score, class_pen, mean_pen, adjusted)

            lambda_info = estimate_lambda_local(
                seed=seed,
                n_samples=n_samples,
                target_size=target_size,
                eval_score_fn=lambda mask: evaluate(mask, 0.0, 0.0).raw_score,
                penalty_fn=lambda mask: compute_balance_penalty(mask, labels_np, num_classes, target_size),
                mean_penalty_fn=mean_penalty_fn,
                M=max(10, min(args.lambda_M, n_samples)),
                eps=1e-8,
            )
            lambda_cls = float(lambda_info["lambda_cls"])
            lambda_mean = float(lambda_info["lambda_mean"])
            lambdas.append((lambda_cls, lambda_mean))

            rng = np.random.default_rng(seed)
            random_idx = rng.choice(n_samples, size=target_size, replace=False)
            random_mask = np.zeros(n_samples, dtype=np.uint8)
            random_mask[random_idx] = 1
            per_method_seed_metrics["random"].append(evaluate(random_mask, lambda_cls, lambda_mean))

            for method in ["herding", "learned_topk", "learned_group"]:
                mask = load_mask_or_raise(dataset_name, args.model_name, seed, kr, method)
                if int(mask.sum()) != target_size:
                    raise ValueError(
                        f"Mask size mismatch: method={method}, dataset={dataset_name}, seed={seed}, kr={kr}, "
                        f"expected={target_size}, got={int(mask.sum())}"
                    )
                per_method_seed_metrics[method].append(evaluate(mask, lambda_cls, lambda_mean))

        lambda_cls_mean = float(np.mean([x[0] for x in lambdas]))
        lambda_mean_mean = float(np.mean([x[1] for x in lambdas]))

        print("=" * 92)
        print(f"dataset={dataset_name} | kr={kr}")
        print(f"lambda_cls(mean over seeds)={lambda_cls_mean:.4f} | lambda_mean(mean over seeds)={lambda_mean_mean:.4f}")
        print("-" * 92)
        print(
            f"{'method':<16}{'raw_score_mean':>16}{'class_corr_mean':>18}{'herd_corr_mean':>18}{'adjusted_score_mean':>22}"
        )
        print("-" * 92)

        bar_raw, bar_cls_corr, bar_mean_corr = [], [], []
        for method in methods:
            vals = per_method_seed_metrics[method]
            raw_mean = float(np.mean([v.raw_score for v in vals]))
            cls_corr_mean = float(np.mean([lambda_cls_mean * v.class_penalty for v in vals]))
            mean_corr_mean = float(np.mean([lambda_mean_mean * v.mean_penalty for v in vals]))
            adjusted_mean = float(np.mean([v.adjusted_score for v in vals]))
            print(f"{method:<16}{raw_mean:>16.4f}{cls_corr_mean:>18.4f}{mean_corr_mean:>18.4f}{adjusted_mean:>22.4f}")
            bar_raw.append(raw_mean)
            bar_cls_corr.append(cls_corr_mean)
            bar_mean_corr.append(mean_corr_mean)

        x = np.arange(len(methods))
        width = 0.24
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x - width, bar_raw, width=width, label="raw_score", color="#4C72B0")
        ax.bar(x, bar_cls_corr, width=width, label="class_correction", color="#DD8452")
        ax.bar(x + width, bar_mean_corr, width=width, label="herd_correction", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Score / Correction")
        ax.set_title(f"Subset score decomposition | dataset={dataset_name}, kr={kr}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        out_path = out_dir / f"{kr}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
