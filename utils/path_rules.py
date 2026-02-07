"""Path naming rules derived from experiment parameters."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_proxy_log_dir(
    dataset: str,
    seed: int,
    *,
    proxy_model: str = "resnet18",
    epochs: int,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "weights" / "proxy_logs"
    return base / dataset / proxy_model / str(seed) / str(int(epochs))


def resolve_checkpoint_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    cut_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "checkpoint"
    return base / mode / dataset / model / str(seed) / f"checkpoint_{int(cut_ratio)}.pt"


def resolve_mask_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    cut_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "mask"
    return base / mode / dataset / model / str(seed) / f"mask_{int(cut_ratio)}.npz"


def resolve_result_path(
    mode: str,
    dataset: str,
    model: str,
    seed: int,
    cut_ratio: int,
    *,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root is not None else PROJECT_ROOT / "result"
    return base / mode / dataset / model / str(seed) / f"result_{int(cut_ratio)}.json"
