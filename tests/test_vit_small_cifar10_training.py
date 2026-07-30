from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision import transforms

from train_after_selection import (
    apply_dataset_defaults,
    apply_model_specific_transforms,
    build_optimizer_and_scheduler,
    build_training_metadata,
    build_vit_small_cifar10_transforms,
    validate_vit_checkpoint,
)


def make_args(dataset="cifar10", model="vit_small", **overrides):
    values = dict(
        dataset=dataset, model=model, epochs=None, batch_size=None, init_lr=None,
        momentum=None, weight_decay=None, lr_milestones=None, lr_gamma=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_vit_small_cifar10_defaults():
    args = apply_dataset_defaults(make_args())
    assert (args.epochs, args.batch_size, args.init_lr, args.weight_decay) == (200, 512, 1e-4, 0.0)
    assert (args.optimizer_name, args.scheduler_name, args.use_amp) == ("adam", "cosine", True)
    assert (args.physical_batch_size, args.effective_batch_size, args.grad_accum_steps) == (512, 512, 1)


def test_vit_small_cli_numeric_overrides():
    args = apply_dataset_defaults(make_args(epochs=100, batch_size=256, init_lr=2e-4, weight_decay=1e-5))
    assert (args.epochs, args.batch_size, args.init_lr, args.weight_decay) == (100, 256, 2e-4, 1e-5)
    assert (args.optimizer_name, args.scheduler_name) == ("adam", "cosine")


def test_resnet50_cifar10_defaults_unchanged():
    args = apply_dataset_defaults(make_args(model="resnet50"))
    assert (args.epochs, args.batch_size, args.init_lr) == (200, 128, 0.1)
    assert (args.momentum, args.weight_decay) == (0.9, 5e-4)
    assert (args.lr_milestones, args.lr_gamma, args.use_amp) == ([60, 120, 160], 0.2, False)


def test_vit_optimizer_is_adam():
    args = apply_dataset_defaults(make_args())
    optimizer, scheduler = build_optimizer_and_scheduler(args, nn.Linear(2, 2))
    assert isinstance(optimizer, Adam)
    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert isinstance(scheduler, CosineAnnealingLR)
    assert (scheduler.T_max, scheduler.eta_min) == (200, 0.0)


def test_resnet_optimizer_remains_sgd():
    args = apply_dataset_defaults(make_args(model="resnet50"))
    optimizer, scheduler = build_optimizer_and_scheduler(args, nn.Linear(2, 2))
    assert isinstance(optimizer, SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert isinstance(scheduler, MultiStepLR)
    assert list(scheduler.milestones.elements()) == [60, 120, 160]
    assert scheduler.gamma == 0.2


def test_vit_train_transform():
    train, _ = build_vit_small_cifar10_transforms()
    assert [type(item) for item in train.transforms] == [
        transforms.RandAugment, transforms.RandomCrop, transforms.RandomHorizontalFlip,
        transforms.ToTensor, transforms.Normalize,
    ]
    assert (train.transforms[0].num_ops, train.transforms[0].magnitude) == (2, 14)
    assert (train.transforms[1].size, train.transforms[1].padding) == ((32, 32), 4)
    assert train.transforms[-1].mean == (0.4914, 0.4822, 0.4465)
    assert train.transforms[-1].std == (0.2023, 0.1994, 0.2010)


def test_vit_test_transform():
    _, test = build_vit_small_cifar10_transforms()
    assert [type(item) for item in test.transforms] == [transforms.ToTensor, transforms.Normalize]
    assert test.transforms[-1].mean == (0.4914, 0.4822, 0.4465)
    assert test.transforms[-1].std == (0.2023, 0.1994, 0.2010)


@pytest.mark.parametrize("dataset,model,expected", [
    ("cifar10", "vit_small", True), ("cifar10", "resnet50", False),
    ("cifar100", "vit_small", False),
])
def test_transform_is_model_specific(dataset, model, expected):
    train = Namespace(transform="original-train")
    test = Namespace(transform="original-test")
    assert apply_model_specific_transforms(make_args(dataset, model), train, test) is expected
    assert isinstance(train.transform, transforms.Compose) is expected
    if not expected:
        assert (train.transform, test.transform) == ("original-train", "original-test")


@pytest.mark.parametrize("checkpoint", [{}, {"training_recipe": "old_sgd_recipe"}])
def test_incompatible_vit_checkpoint_is_rejected(checkpoint):
    args = apply_dataset_defaults(make_args())
    with pytest.raises(RuntimeError, match=r"legacy\.pt.*training_recipe"):
        validate_vit_checkpoint(checkpoint, Path("legacy.pt"), args)


def test_vit_result_metadata():
    args = apply_dataset_defaults(make_args())
    metadata = build_training_metadata(args, runtime_use_amp=True)
    assert metadata["optimizer"] == "Adam"
    assert metadata["optimizer_config"] == {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
    assert metadata["lr_schedule"] == {"type": "CosineAnnealingLR", "T_max": 200, "eta_min": 0.0}
    assert metadata["augmentation"]["randaugment_num_ops"] == 2
    assert metadata["normalization"]["std"] == [0.2023, 0.1994, 0.2010]
    assert metadata["use_amp"] is True
    assert (args.epochs, args.physical_batch_size, args.effective_batch_size) == (200, 512, 512)
