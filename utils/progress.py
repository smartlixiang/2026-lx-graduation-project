"""Helpers for stable multi-line tqdm layouts."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def create_persistent_bar(
    total: int,
    desc: str,
    *,
    position: int,
    initial: int = 0,
) -> tqdm:
    return tqdm(
        total=total,
        desc=desc,
        position=position,
        initial=initial,
        leave=True,
        dynamic_ncols=True,
    )


def create_transient_batch_bar(
    iterable: Iterable[T],
    desc: str,
    *,
    position: int,
) -> tqdm:
    return tqdm(
        iterable,
        desc=desc,
        unit="batch",
        position=position,
        leave=False,
        dynamic_ncols=True,
    )


class PersistentStatusLine:
    """A single persistent line for concise state updates."""

    def __init__(self, text: str, *, position: int) -> None:
        self._bar = tqdm(
            total=1,
            desc=text,
            bar_format="{desc}",
            position=position,
            leave=True,
            dynamic_ncols=True,
        )
        self._bar.n = 1
        self._bar.refresh()

    def update(self, text: str) -> None:
        self._bar.set_description_str(text)
        self._bar.n = 1
        self._bar.refresh()

    def close(self) -> None:
        self._bar.close()
