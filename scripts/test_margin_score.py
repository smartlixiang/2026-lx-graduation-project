"""Test reading proxy logs and computing MarginScore."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weights import MarginScore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_path",
        type=Path,
        default=Path("weights/proxy_logs/cifar10_resnet18_2026_01_12_14_31.npz"),
        help="Path to the proxy log .npz file.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Margin threshold for boundary samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = MarginScore(args.npz_path, delta=args.delta)

    start = time.perf_counter()
    result = scorer.compute()
    elapsed = time.perf_counter() - start

    print(f"Loaded: {args.npz_path}")
    print(f"Samples: {result.scores.shape[0]}")
    print(f"Delta: {result.delta}")
    print(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
    print(f"Elapsed: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
