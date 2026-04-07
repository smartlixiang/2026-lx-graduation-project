"""Shared dataset-specific training recipes."""
from __future__ import annotations

import argparse


DATASET_DEFAULT_TRAINING_CONFIGS = {
    "cifar10": {
        "epochs": 200,
        "batch_size": 128,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_milestones": [60, 120, 160],
        "lr_gamma": 0.2,
        "execution": {
            "use_amp": False,
            "reference_effective_batch_size": None,
        },
    },
    "cifar100": {
        "epochs": 200,
        "batch_size": 128,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_milestones": [60, 120, 160],
        "lr_gamma": 0.2,
        "execution": {
            "use_amp": False,
            "reference_effective_batch_size": None,
        },
    },
    "tiny-imagenet": {
        "epochs": 90,
        "batch_size": 256,
        "init_lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_milestones": [30, 60],
        "lr_gamma": 0.1,
        "execution": {
            "default_physical_batch_size": 64,
            "use_amp": True,
            "reference_effective_batch_size": 256,
        },
    },
}


def get_default_training_config(dataset_name: str) -> dict[str, int | float | list[int]]:
    try:
        config = DATASET_DEFAULT_TRAINING_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset for training config: {dataset_name}") from exc
    return {
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "init_lr": float(config["init_lr"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "lr_milestones": list(config["lr_milestones"]),
        "lr_gamma": float(config["lr_gamma"]),
    }


def apply_dataset_training_defaults(
    args: argparse.Namespace,
    *,
    lr_attr: str,
) -> argparse.Namespace:
    defaults = get_default_training_config(args.dataset)
    batch_size_unspecified = args.batch_size is None

    if args.epochs is None:
        args.epochs = defaults["epochs"]

    if getattr(args, lr_attr) is None:
        setattr(args, lr_attr, defaults["init_lr"])
    if args.momentum is None:
        args.momentum = defaults["momentum"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.lr_milestones is None:
        args.lr_milestones = defaults["lr_milestones"]
    if args.lr_gamma is None:
        args.lr_gamma = defaults["lr_gamma"]

    execution_config = DATASET_DEFAULT_TRAINING_CONFIGS[args.dataset]["execution"]
    if args.dataset == "tiny-imagenet":
        if batch_size_unspecified:
            args.batch_size = int(execution_config["default_physical_batch_size"])
        args.use_amp = bool(execution_config["use_amp"])
        args.reference_effective_batch_size = int(execution_config["reference_effective_batch_size"])
        args.physical_batch_size = int(args.batch_size)
        if args.physical_batch_size <= 64:
            args.grad_accum_steps = 4
        elif args.physical_batch_size <= 128:
            args.grad_accum_steps = 2
        else:
            args.grad_accum_steps = 1
        args.effective_batch_size = args.physical_batch_size * args.grad_accum_steps
    else:
        if batch_size_unspecified:
            args.batch_size = defaults["batch_size"]
        args.use_amp = bool(execution_config["use_amp"])
        args.reference_effective_batch_size = args.batch_size
        args.grad_accum_steps = 1
        args.physical_batch_size = args.batch_size
        args.effective_batch_size = args.batch_size
    return args

