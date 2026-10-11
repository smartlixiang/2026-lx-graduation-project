"""Training utilities for the author-code SL-ViT recipe.

Reference: aanna0701/SPT_LSA_ViT at 54bd796acd28fa11deac21a97b14b154379b34e7.
AutoAugment operation semantics follow the author's DeepVoltaire-derived policy.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Sampler
from torchvision import transforms


def load_sl_vit_config(dataset: str) -> dict:
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError(f"sl_vit supports only cifar10 and cifar100, received {dataset}.")
    path = Path(__file__).resolve().parents[1] / "configs" / f"sl_vit_{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class OfficialCIFAR10Policy:
    """Author's 25 policies, including its PIL fill/interpolation/rotation rules."""

    policies = (
        (.1, "invert", 7, .2, "contrast", 6), (.7, "rotate", 2, .3, "translateX", 9),
        (.8, "sharpness", 1, .9, "sharpness", 3), (.5, "shearY", 8, .7, "translateY", 9),
        (.5, "autocontrast", 8, .9, "equalize", 2), (.2, "shearY", 7, .3, "posterize", 7),
        (.4, "color", 3, .6, "brightness", 7), (.3, "sharpness", 9, .7, "brightness", 9),
        (.6, "equalize", 5, .5, "equalize", 1), (.6, "contrast", 7, .6, "sharpness", 5),
        (.7, "color", 7, .5, "translateX", 8), (.3, "equalize", 7, .4, "autocontrast", 8),
        (.4, "translateY", 3, .2, "sharpness", 6), (.9, "brightness", 6, .2, "color", 8),
        (.5, "solarize", 2, .0, "invert", 3), (.2, "equalize", 0, .6, "autocontrast", 0),
        (.2, "equalize", 8, .6, "equalize", 4), (.9, "color", 9, .6, "equalize", 6),
        (.8, "autocontrast", 4, .2, "solarize", 8), (.1, "brightness", 3, .7, "color", 0),
        (.4, "solarize", 5, .9, "autocontrast", 3), (.9, "translateY", 9, .7, "translateY", 9),
        (.9, "autocontrast", 2, .8, "solarize", 3), (.8, "equalize", 8, .1, "invert", 3),
        (.7, "translateY", 9, .9, "autocontrast", 1),
    )

    @staticmethod
    def _apply(image, operation, index):
        if operation.startswith("shear") or operation.startswith("translate"):
            if operation.startswith("shear"):
                value = np.linspace(0, .3, 10)[index] * random.choice([-1, 1])
                matrix = (1, value, 0, 0, 1, 0) if operation == "shearX" else (1, 0, 0, value, 1, 0)
                return image.transform(image.size, Image.Transform.AFFINE, matrix,
                                       Image.Resampling.BICUBIC, fillcolor=(128, 128, 128))
            value = np.linspace(0, 150/331, 10)[index]
            axis = 0 if operation == "translateX" else 1
            value *= image.size[axis] * random.choice([-1, 1])
            matrix = (1, 0, value, 0, 1, 0) if axis == 0 else (1, 0, 0, 0, 1, value)
            return image.transform(image.size, Image.Transform.AFFINE, matrix, fillcolor=(128, 128, 128))
        if operation == "rotate":
            rotated = image.convert("RGBA").rotate(np.linspace(0, 30, 10)[index])
            return Image.composite(rotated, Image.new("RGBA", rotated.size, (128,)*4), rotated).convert(image.mode)
        enhancers = {"color": ImageEnhance.Color, "contrast": ImageEnhance.Contrast,
                     "brightness": ImageEnhance.Brightness, "sharpness": ImageEnhance.Sharpness}
        if operation in enhancers:
            magnitude = np.linspace(0, .9, 10)[index]
            return enhancers[operation](image).enhance(1 + magnitude * random.choice([-1, 1]))
        if operation == "posterize":
            return ImageOps.posterize(image, int(np.round(np.linspace(8, 4, 10))[index]))
        if operation == "solarize":
            return ImageOps.solarize(image, np.linspace(256, 0, 10)[index])
        return {"invert": ImageOps.invert, "equalize": ImageOps.equalize,
                "autocontrast": ImageOps.autocontrast}[operation](image)

    def __call__(self, image):
        policy = self.policies[random.randint(0, len(self.policies)-1)]
        for probability, operation, magnitude in (policy[:3], policy[3:]):
            if random.random() < probability:
                image = self._apply(image, operation, magnitude)
        return image


class OfficialRandomErasing:
    """Uniform area/aspect sampling, 100 attempts, mean fill after normalization."""

    def __init__(self, mean, probability=.25, min_area=.02, max_area=.4, min_aspect=.3):
        self.mean = tuple(mean)
        self.probability = probability
        self.min_area, self.max_area, self.min_aspect = min_area, max_area, min_aspect

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image
        height, width = image.shape[-2:]
        for _ in range(100):
            area = random.uniform(self.min_area, self.max_area) * height * width
            aspect = random.uniform(self.min_aspect, 1/self.min_aspect)
            h, w = round(math.sqrt(area*aspect)), round(math.sqrt(area/aspect))
            if w < width and h < height:
                row, column = random.randint(0, height-h), random.randint(0, width-w)
                image[:, row:row+h, column:column+w] = image.new_tensor(self.mean)[:, None, None]
                break
        return image


def build_sl_vit_transforms(config: dict):
    aug, norm = config["augmentation"], config["normalization"]
    train = transforms.Compose([
        transforms.RandomHorizontalFlip(aug["horizontal_flip_probability"]),
        transforms.RandomCrop(aug["crop_size"], padding=aug["crop_padding"]),
        OfficialCIFAR10Policy(), transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
        OfficialRandomErasing(norm["mean"], **aug["random_erasing"]),
    ])
    test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm["mean"], norm["std"])])
    return train, test


class RepeatedAugmentationBatchSampler(Sampler):
    """Upstream RA semantics: concatenate fresh permutations to 3N, then batch.

    Local RNG and set_epoch make the ordering deterministic when resuming.
    Every occurrence still calls the dataset's random transform independently.
    """

    def __init__(self, dataset_len: int, batch_size: int, factor: int = 3, seed: int = 0):
        if min(dataset_len, batch_size, factor) <= 0:
            raise ValueError("Dataset length, batch size and RA factor must be positive.")
        self.dataset_len, self.batch_size = dataset_len, batch_size
        self.factor, self.seed, self.epoch = factor, seed, 0
        if len(self) == 0:
            raise ValueError("SL-ViT drop_last=True produces no batches; increase selection size or reduce --batch_size.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.dataset_len * self.factor // self.batch_size

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = np.concatenate([rng.permutation(self.dataset_len) for _ in range(self.factor)])
        for start in range(0, len(self)*self.batch_size, self.batch_size):
            yield indices[start:start+self.batch_size].tolist()


class BatchCosineWarmup(torch.optim.lr_scheduler.LRScheduler):
    """Author's linear warmup + single cosine cycle, stepped after each batch.

    At the completed cycle boundary the author scheduler resets to min_lr.
    There are no additional cycles within the configured training duration.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-6):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps.")
        self.total_steps, self.warmup_steps = total_steps, warmup_steps
        self.max_lr, self.min_lr = max_lr, min_lr
        super().__init__(optimizer)
        # Match upstream initialization even for a zero-warmup override.
        for group in optimizer.param_groups:
            group["lr"] = min_lr
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self):
        step = self.last_epoch % self.total_steps
        if step < self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            factor = (1 + math.cos(math.pi*(step-self.warmup_steps)/(self.total_steps-self.warmup_steps))) / 2
        return [self.min_lr + (self.max_lr-self.min_lr)*factor for _ in self.optimizer.param_groups]


def mix_sl_vit_batch(images, labels, config: dict):
    aug = config["augmentation"]
    if np.random.rand() >= aug["mix_probability"]:
        return images, labels, labels, 1.0
    cutmix = np.random.rand() < aug["cutmix_switch_probability"]
    alpha = aug["cutmix_beta"] if cutmix else aug["mixup_alpha"]
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    if not cutmix:
        images = lam*images + (1-lam)*images[permutation]
    else:
        height, width = images.shape[-2:]
        box_h, box_w = int(height*np.sqrt(1-lam)), int(width*np.sqrt(1-lam))
        center_h, center_w = np.random.randint(height), np.random.randint(width)
        top, bottom = np.clip([center_h-box_h//2, center_h+box_h//2], 0, height)
        left, right = np.clip([center_w-box_w//2, center_w+box_w//2], 0, width)
        images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
        lam = 1 - ((bottom-top)*(right-left)/(height*width))
    return images, labels, labels[permutation], float(lam)


def sl_vit_training_metadata(args, runtime_use_amp: bool) -> dict:
    config = args.sl_vit_config
    return {
        "training_recipe": config["training_recipe"], "source": config["source"],
        "optimizer": "AdamW",
        "optimizer_config": {"lr": args.init_lr, "betas": config["betas"], "eps": config["eps"], "weight_decay": args.weight_decay},
        "lr_schedule": {"type": "BatchCosineWarmup", "step_unit": "batch", "epochs": args.epochs,
                        "warmup_epochs": args.warmup_epochs, "min_lr": config["min_lr"]},
        "augmentation": config["augmentation"], "normalization": config["normalization"],
        "label_smoothing": config["label_smoothing"], "stochastic_depth": config["stochastic_depth"],
        "use_amp": runtime_use_amp, "pretrained": False,
        "evaluation_protocol": "project_last_10_epochs_mean",
    }
