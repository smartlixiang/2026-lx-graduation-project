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


def _config_payload(config: dict) -> dict[str, int | float | list[int]]:
    return {
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "init_lr": float(config["init_lr"]),
        "momentum": float(config["momentum"]),
        "weight_decay": float(config["weight_decay"]),
        "lr_milestones": list(config["lr_milestones"]),
        "lr_gamma": float(config["lr_gamma"]),
    }


def get_default_training_config(dataset_name: str) -> dict[str, int | float | list[int]]:
    try:
        config = DATASET_DEFAULT_TRAINING_CONFIGS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset for training config: {dataset_name}") from exc
    return _config_payload(config)



def get_proxy_training_config(dataset_name: str) -> dict[str, int | float | list[int]]:
    """Return proxy-only training defaults without changing target-model recipes."""
    base = get_default_training_config(dataset_name)
    if dataset_name in {"cifar10", "cifar100"}:
        base["epochs"] = 160
        base["lr_milestones"] = [60, 120]
    elif dataset_name == "tiny-imagenet":
        base["epochs"] = 80
        base["lr_milestones"] = [30, 60]
    else:
        raise ValueError(f"Unsupported dataset for proxy training config: {dataset_name}")
    return base


def apply_proxy_training_defaults(args: argparse.Namespace, *, lr_attr: str = "lr") -> argparse.Namespace:
    proxy_defaults = get_proxy_training_config(args.dataset)
    original_getter = globals()["get_default_training_config"]
    try:
        globals()["get_default_training_config"] = lambda dataset_name: proxy_defaults if dataset_name == args.dataset else original_getter(dataset_name)
        return apply_dataset_training_defaults(args, lr_attr=lr_attr)
    finally:
        globals()["get_default_training_config"] = original_getter


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

