"""Test reading proxy logs and computing ForgettingScore."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weights import ForgettingScore  # noqa: E402
from utils.global_config import CONFIG  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_path",
        type=Path,
        default=Path("weights/proxy_logs/cifar10_resnet18_2026_01_12_14_31.npz"),
        help="Path to the proxy log .npz file.",
    )
    parser.add_argument("--seed", type=int, default=CONFIG.global_seed, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    scorer = ForgettingScore(args.npz_path)

    start = time.perf_counter()
    result = scorer.compute()
    elapsed = time.perf_counter() - start

    print(f"Loaded: {args.npz_path}")
    print(f"Samples: {result.scores.shape[0]}")
    print(f"Start epoch: {result.start_epoch}")
    print(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
    print(f"Elapsed: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
