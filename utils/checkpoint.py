"""Shared command-line defaults for training checkpoint saves."""
from __future__ import annotations

import argparse


DEFAULT_CHECKPOINT_INTERVAL = 5


def _positive_interval(value: str) -> int:
    try:
        interval = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint interval must be a positive integer") from exc
    if interval <= 0:
        raise argparse.ArgumentTypeError("checkpoint interval must be a positive integer")
    return interval


def add_checkpoint_interval_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint_interval",
        "--checkpoint-interval",
        type=_positive_interval,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="每隔多少轮覆盖保存最新 checkpoint（默认：5，必须为正整数）",
    )
