"""Path naming rules derived from experiment parameters."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_proxy_log_dir(
    dataset: str,
    seed: int | None = None,
    *,
    proxy_model: str = "resnet18",
    epochs: int,
    root: Path | str | None = None,
) -> Path:
    """Return the canonical proxy-CV log directory.

    New rule:
        weights/proxy_logs/[dataset]/[proxy_model]/[seed]/[max_epochs]

    ``seed`` is the actual random seed used by train_proxy.py for fold
    construction, model initialization, dataloader shuffling and trajectory
    logging.  Keeping it in the path prevents different proxy runs from
    overwriting each other or sharing downstream dynamic caches by accident.
    """
    if seed is None:
        raise ValueError(
            "resolve_proxy_log_dir now requires an explicit seed. "
            "Expected path rule: [dataset]/[proxy_model]/[seed]/[max_epochs]."
        )
    base = Path(root) if root is not None else PROJECT_ROOT / "weights" / "proxy_logs"
    return base / dataset / proxy_model / str(int(seed)) / str(int(epochs))


def resolve_checkpoint_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    keep_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "checkpoint"
    return base / mode / dataset / model / str(seed) / f"checkpoint_{int(keep_ratio)}.pt"


def resolve_mask_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    keep_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "mask"
    return base / mode / dataset / str(seed) / f"mask_{int(keep_ratio)}.npz"


def resolve_result_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    keep_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "result"
    return base / mode / dataset / model / str(seed) / f"result_{int(keep_ratio)}.json"
