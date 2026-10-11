"""Ablate each dynamic component used to construct the calibration target.

The script reuses the existing adapter and static-score caches.  By default it
rebuilds each ablated A/C/T pseudo-label and refits the three static weights;
``--reuse-saved-weights`` instead loads validated weights from
``weights/ablation_weights.json`` and skips all dynamic-target work.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calculate_my_mask import (
    _build_dataset,
    build_score_loader,
    parse_ratio_list,
    select_group_mask,
)
from dataset.dataset_config import CIFAR10, CIFAR100
from learn_scoring_weights import (
    resolve_default_proxy_epochs,
    resolve_dynamic_component_cache_path,
)
from model.adapter import load_trained_adapters
from scoring import DifficultyDirection, Div, SemanticAlignment
from utils.class_name_utils import resolve_class_names_for_prompts
from utils.global_config import CONFIG
from utils.path_rules import resolve_mask_path
from utils.score_utils import standard_zscore, standard_zscore_by_class
from utils.seed import parse_seed_list, set_seed
from utils.static_score_cache import get_or_compute_static_scores
from weights.calibration import fit_softplus_ratio_regression


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "ablation_weights.json"
MAIN_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "scoring_weights.json"
DYNAMIC_COMPONENTS = ("A", "C", "T")
STATIC_COMPONENTS = ("sa", "div", "dds")
ABLATION_METHODS = {
    "A": "ablation_A",
    "C": "ablation_C",
    "T": "ablation_T",
}
DEFAULT_DATASETS = (CIFAR10, CIFAR100)
DEFAULT_SEEDS = (22, 42, 96)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learned-group ablations for pseudo-label components A/C/T."
    )
    parser.add_argument("--dataset", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--component", default=",".join(DYNAMIC_COMPONENTS))
    parser.add_argument("--kr", default="20,30,40,50,60,70,80,90")
    parser.add_argument("--seed", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--skip-saved", action="store_true")
    parser.add_argument("--group-candidate-pool-size", type=int, default=10)
    parser.add_argument("--group-init-count", type=int, default=5)
    parser.add_argument("--proxy-model", default="resnet18")
    parser.add_argument("--proxy-epochs", type=int, default=None)
    parser.add_argument(
        "--ratio-lambda",
        type=float,
        default=None,
        help=(
            "Regression ratio regularization. By default, reuse the value "
            "recorded for the same dataset/seed in weights/scoring_weights.json."
        ),
    )
    parser.add_argument("--regression-learning-rate", type=float, default=2e-3)
    parser.add_argument("--regression-max-iter", type=int, default=10000)
    parser.add_argument("--regression-tol", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument(
        "--reuse-saved-weights",
        action="store_true",
        help=(
            "Skip dynamic-target construction and regression; load sa/div/dds "
            "from weights/ablation_weights.json and generate masks directly."
        ),
    )
    return parser


def parse_dataset_list(value: str) -> list[str]:
    datasets = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not datasets:
        raise ValueError("--dataset must contain at least one dataset.")
    unsupported = [name for name in datasets if name not in DEFAULT_DATASETS]
    if unsupported:
        raise ValueError(
            f"Unsupported normal dataset(s): {unsupported}; "
            f"expected a subset of {DEFAULT_DATASETS}."
        )
    return list(dict.fromkeys(datasets))


def parse_component_list(value: str) -> list[str]:
    components = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not components:
        raise ValueError("--component must contain at least one of A,C,T.")
    unsupported = [name for name in components if name not in DYNAMIC_COMPONENTS]
    if unsupported:
        raise ValueError(
            f"Unsupported dynamic component(s): {unsupported}; "
            f"expected a subset of {DYNAMIC_COMPONENTS}."
        )
    return list(dict.fromkeys(components))


def load_dynamic_components(
    dataset: str,
    proxy_model: str,
    seed: int,
    epochs: int,
    num_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    """Load final normalized A/C/T values from the canonical local caches."""
    values: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    expected_shape = (int(num_samples),)

    for name in DYNAMIC_COMPONENTS:
        path = resolve_dynamic_component_cache_path(
            dataset, proxy_model, seed, epochs, name
        )
        if not path.is_file():
            raise FileNotFoundError(f"Required dynamic cache is missing: {path}")
        with np.load(path, allow_pickle=False) as cache:
            if "final_normalized" not in cache:
                raise ValueError(
                    f"Dynamic cache has no final_normalized array: {path}"
                )
            component = np.asarray(cache["final_normalized"], dtype=np.float64)
        if component.shape != expected_shape:
            raise ValueError(
                f"{path}: expected final_normalized shape {expected_shape}, "
                f"got {component.shape}"
            )
        if not np.all(np.isfinite(component)):
            raise ValueError(f"{path}: final_normalized contains NaN/inf")
        values[name] = component
        paths[name] = path

    return values, paths


def build_ablated_dynamic_target(
    component_values: dict[str, np.ndarray], omitted: str
) -> tuple[np.ndarray, tuple[str, str]]:
    """Average the two retained components and standardize the pseudo-label."""
    included = tuple(name for name in DYNAMIC_COMPONENTS if name != omitted)
    if len(included) != 2:
        raise ValueError(f"Unknown omitted component: {omitted}")
    utility_raw = np.mean(
        np.stack([component_values[name] for name in included], axis=0), axis=0
    )
    return standard_zscore(utility_raw).astype(np.float64), included


def load_weight_file() -> dict:
    if not WEIGHTS_PATH.exists():
        return {}
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Weight file must contain a JSON object: {WEIGHTS_PATH}")
    return data


def load_saved_ablation_weights(
    data: dict,
    *,
    dataset: str,
    method: str,
    seed: int,
    omitted_component: str,
) -> dict[str, float]:
    """Load and validate one dynamic-component ablation weight triplet."""
    try:
        entry = data[dataset][method][str(seed)]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "Saved ablation weights are missing: "
            f"file={WEIGHTS_PATH}, dataset={dataset}, method={method}, seed={seed}"
        ) from exc
    if not isinstance(entry, dict):
        raise ValueError(
            f"Saved weight entry must be an object: {dataset}/{method}/{seed}"
        )

    expected_components = tuple(
        name for name in DYNAMIC_COMPONENTS if name != omitted_component
    )
    saved_components = entry.get("dynamic_components")
    if saved_components is not None and set(saved_components) != set(expected_components):
        raise ValueError(
            f"{dataset}/{method}/{seed}: expected dynamic_components "
            f"{expected_components}, got {saved_components}"
        )

    weights: dict[str, float] = {}
    for name in STATIC_COMPONENTS:
        if name not in entry:
            raise ValueError(f"{dataset}/{method}/{seed}: missing weight {name}")
        try:
            value = float(entry[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset}/{method}/{seed}: weight {name} is not numeric"
            ) from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{dataset}/{method}/{seed}: invalid weight {name}={value}"
            )
        weights[name] = value

    weight_sum = float(sum(weights.values()))
    if not np.isclose(weight_sum, 1.0, atol=1e-4):
        raise ValueError(
            f"{dataset}/{method}/{seed}: weights must sum to one, got {weight_sum}"
        )
    return weights


def resolve_ratio_lambda(
    requested: float | None,
    *,
    dataset: str,
    seed: int,
) -> float:
    """Use an explicit value or recover the value used by the main experiment."""
    if requested is not None:
        value = float(requested)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("--ratio-lambda must be finite and non-negative.")
        return value

    if MAIN_WEIGHTS_PATH.is_file():
        with open(MAIN_WEIGHTS_PATH, "r", encoding="utf-8") as file:
            main_data = json.load(file)
        try:
            value = float(main_data[dataset][str(seed)]["ratio_lambda"])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value >= 0.0:
            print(
                f"[ratio-lambda] reuse main experiment: "
                f"dataset={dataset}, seed={seed}, value={value}"
            )
            return value

    fallback = 1e-2
    print(
        f"[ratio-lambda] no saved main-experiment value for "
        f"dataset={dataset}, seed={seed}; use code default {fallback}"
    )
    return fallback


def save_weight_entry(
    data: dict,
    *,
    dataset: str,
    method: str,
    seed: int,
    weights: dict[str, float],
    bias: float,
    included_components: tuple[str, str],
    component_cache_paths: dict[str, Path],
    args: argparse.Namespace,
    ratio_lambda: float,
    proxy_epochs: int,
    fit_result: dict[str, object],
) -> None:
    dataset_entry = data.setdefault(dataset, {})
    method_entry = dataset_entry.setdefault(method, {})
    method_entry[str(seed)] = {
        "experiment": method,
        "static_components": ["SA", "Div", "DDS"],
        "dynamic_components": list(included_components),
        "sa": float(weights["sa"]),
        "div": float(weights["div"]),
        "dds": float(weights["dds"]),
        "bias": float(bias),
        "ratio_lambda": float(ratio_lambda),
        "proxy_model": args.proxy_model,
        "proxy_epochs": int(proxy_epochs),
        "proxy_training_seed": int(seed),
        "dynamic_component_cache_paths": {
            name: str(component_cache_paths[name]) for name in included_components
        },
        "adapter_seed": int(seed),
        "diagnostics": {
            "weight_solver": "softplus_ratio_no_simplex",
            "raw_weights": [
                float(value)
                for value in np.asarray(
                    fit_result["raw_weights"], dtype=np.float64
                )
            ],
            "mse": float(fit_result["mse"]),
            "ratio_regularizer": float(fit_result["ratio_regularizer"]),
            "iterations": int(fit_result["iterations"]),
            "weights_sum": float(sum(weights.values())),
        },
    }

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = WEIGHTS_PATH.with_suffix(".json.tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(WEIGHTS_PATH)


def run_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    seeds: list[int],
    keep_ratios: list[int],
    omitted_components: list[str],
    weight_data: dict,
) -> None:
    device = torch.device(args.device) if args.device else CONFIG.global_device
    proxy_epochs = (
        int(args.proxy_epochs)
        if args.proxy_epochs is not None
        else resolve_default_proxy_epochs(dataset_name)
    )
    dataset = _build_dataset(dataset_name, transform=None)
    class_names = list(
        resolve_class_names_for_prompts(
            dataset_name=dataset_name,
            data_root=PROJECT_ROOT / "data",
            class_names=dataset.classes,
        )
    )
    labels = np.asarray(dataset.targets, dtype=np.int64)

    dds_metric = DifficultyDirection(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    div_metric = Div(
        class_names=class_names, clip_model=args.clip_model, device=device
    )
    sa_metric = SemanticAlignment(
        class_names=class_names,
        clip_model=args.clip_model,
        device=device,
        dataset_name=dataset_name,
        data_root=str(PROJECT_ROOT / "data"),
        debug_prompts=args.debug_prompts,
    )
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

    for seed in seeds:
        set_seed(seed)
        image_adapter, text_adapter, adapter_paths = load_trained_adapters(
            dataset_name=dataset_name,
            clip_model=args.clip_model,
            input_dim=dds_metric.extractor.embed_dim,
            seed=seed,
            map_location=device,
        )
        image_adapter.to(device).eval()
        text_adapter.to(device).eval()

        def compute_scores() -> dict[str, np.ndarray]:
            return {
                "dds": np.asarray(
                    dds_metric.score_dataset(
                        tqdm(dds_loader, desc="Scoring DDS", unit="batch"),
                        adapter=image_adapter,
                    ).scores
                ),
                "div": np.asarray(
                    div_metric.score_dataset(
                        tqdm(div_loader, desc="Scoring Div", unit="batch"),
                        adapter=image_adapter,
                    ).scores
                ),
                "sa": np.asarray(
                    sa_metric.score_dataset(
                        tqdm(sa_loader, desc="Scoring SA", unit="batch"),
                        adapter_image=image_adapter,
                        adapter_text=text_adapter,
                    ).scores
                ),
                "labels": labels,
            }

        scores = get_or_compute_static_scores(
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
            num_samples=len(dataset),
            compute_fn=compute_scores,
        )
        if not np.array_equal(labels, scores["labels"]):
            raise ValueError("Static cache labels do not match the current dataset")

        static_features: np.ndarray | None = None
        component_values: dict[str, np.ndarray] | None = None
        component_paths: dict[str, Path] | None = None
        ratio_lambda: float | None = None
        if not args.reuse_saved_weights:
            static_features = np.stack(
                [
                    standard_zscore_by_class(scores[name], labels)
                    for name in STATIC_COMPONENTS
                ],
                axis=1,
            ).astype(np.float64)
            component_values, component_paths = load_dynamic_components(
                dataset_name,
                args.proxy_model,
                seed,
                proxy_epochs,
                len(dataset),
            )
            ratio_lambda = resolve_ratio_lambda(
                args.ratio_lambda, dataset=dataset_name, seed=seed
            )

        for omitted in omitted_components:
            method = ABLATION_METHODS[omitted]
            if args.reuse_saved_weights:
                weights = load_saved_ablation_weights(
                    weight_data,
                    dataset=dataset_name,
                    method=method,
                    seed=seed,
                    omitted_component=omitted,
                )
                print(
                    f"[weights reused] dataset={dataset_name}, seed={seed}, "
                    f"method={method}, weights={weights}"
                )
            else:
                if (
                    static_features is None
                    or component_values is None
                    or component_paths is None
                    or ratio_lambda is None
                ):
                    raise RuntimeError("Weight-learning inputs were not initialized.")
                dynamic_target, included = build_ablated_dynamic_target(
                    component_values, omitted
                )
                fit_result = fit_softplus_ratio_regression(
                    static_features,
                    dynamic_target,
                    ratio_lambda,
                    args.regression_learning_rate,
                    args.regression_max_iter,
                    args.regression_tol,
                    device,
                )
                fitted = np.asarray(
                    fit_result["normalized_weights"], dtype=np.float64
                )
                if fitted.shape != (len(STATIC_COMPONENTS),):
                    raise RuntimeError(
                        f"Unexpected fitted weight shape for {method}: {fitted.shape}"
                    )
                if not np.isclose(fitted.sum(), 1.0, atol=1e-6):
                    raise RuntimeError(
                        f"Fitted weights do not sum to one for {method}"
                    )
                weights = dict(zip(STATIC_COMPONENTS, fitted.tolist()))
                save_weight_entry(
                    weight_data,
                    dataset=dataset_name,
                    method=method,
                    seed=seed,
                    weights=weights,
                    bias=float(fit_result["bias"]),
                    included_components=included,
                    component_cache_paths=component_paths,
                    args=args,
                    ratio_lambda=ratio_lambda,
                    proxy_epochs=proxy_epochs,
                    fit_result=fit_result,
                )
                print(
                    f"[weights fitted] dataset={dataset_name}, seed={seed}, "
                    f"method={method}, included={included}, weights={weights}"
                )

            for keep_ratio in keep_ratios:
                path = resolve_mask_path(
                    mode=method,
                    dataset=dataset_name,
                    model=args.model_name,
                    seed=seed,
                    keep_ratio=keep_ratio,
                )
                if args.skip_saved and path.exists():
                    print(f"[skip] {path}")
                    continue

                start = time.perf_counter()
                mask, _, stats = select_group_mask(
                    scores["sa"],
                    div_metric=div_metric,
                    div_loader=div_loader,
                    image_adapter=image_adapter,
                    labels=labels,
                    weights=weights,
                    num_classes=len(class_names),
                    keep_ratio=keep_ratio,
                    device=device,
                    seed=seed,
                    weight_group="learned",
                    div_static_scores=scores["div"],
                    dds_static_scores=scores["dds"],
                    group_candidate_pool_size=args.group_candidate_pool_size,
                    group_init_count=args.group_init_count,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, mask=mask.astype(np.uint8))
                print(
                    f"[saved] {path} selected={int(mask.sum())} "
                    f"elapsed={time.perf_counter() - start:.2f}s "
                    f"subset_score={stats['subset_comprehensive_score']:.6f} "
                    f"distribution_shift={stats['distribution_shift']:.6f}"
                )


def main() -> None:
    args = build_parser().parse_args()
    datasets = parse_dataset_list(args.dataset)
    seeds = parse_seed_list(args.seed)
    keep_ratios = parse_ratio_list(args.kr)
    omitted_components = parse_component_list(args.component)
    if not seeds:
        raise ValueError("--seed must contain at least one seed.")
    if not keep_ratios:
        raise ValueError("--kr must contain at least one keep ratio.")
    if args.reuse_saved_weights and not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"--reuse-saved-weights requires an existing file: {WEIGHTS_PATH}"
        )

    weight_data = load_weight_file()
    for dataset_name in datasets:
        run_dataset(
            args,
            dataset_name,
            seeds,
            keep_ratios,
            omitted_components,
            weight_data,
        )


if __name__ == "__main__":
    main()
