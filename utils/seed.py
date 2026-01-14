"""Seed utilities for reproducible experiments."""
from __future__ import annotations

import ast
import random
from typing import Iterable

import numpy as np
import torch


def parse_seed_list(seed: int | str | Iterable[int]) -> list[int]:
    """Parse a seed argument into a list of integers.

    Accepts a single int, comma-separated string (e.g. "22,42,96"),
    or a Python list/tuple string (e.g. "[22, 42, 96]").
    """

    if isinstance(seed, int):
        return [seed]
    if isinstance(seed, (list, tuple)):
        return [int(s) for s in seed]
    if isinstance(seed, str):
        value = seed.strip()
        if not value:
            return []
        if value.startswith("[") or value.startswith("("):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, int):
                return [int(parsed)]
            if isinstance(parsed, (list, tuple)):
                return [int(s) for s in parsed]
            raise ValueError(f"Unsupported seed literal: {seed}")
        if "," in value:
            return [int(s.strip()) for s in value.split(",") if s.strip()]
        return [int(value)]
    raise TypeError(f"Unsupported seed argument: {seed!r}")


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["parse_seed_list", "set_seed"]
