from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model.adapter import load_trained_adapters
from utils.global_config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class DDSCompareResult:
    raw_band: np.ndarray
    raw_top80: np.ndarray
    norm_band: np.ndarray
    norm_top80: np.ndarray
    diff_signed: np.ndarray
    diff_abs: np.ndarray
    labels: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "signed_diff_mean": float(np.mean(self.diff_signed)),
            "signed_diff_var": float(np.var(self.diff_signed)),
            "abs_diff_mean": float(np.mean(self.diff_abs)),
            "abs_diff_var": float(np.var(self.diff_abs)),
        }


class IndependentDDSComparator:
    """
    Compare two independently implemented DDS variants on the same features:
    1) band DDS: use PCA directions whose individual eigenvalue ratio lies in [lower, upper]
    2) top80 DDS: use top principal directions whose cumulative eigenvalue ratio reaches target_ratio
    """

    def __init__(
        self,
        clip_model: str,
        device: torch.device,
        pca_cov_reg: float = 1e-6,
        band_lower: float = 0.02,
        band_upper: float = 0.20,
        top_ratio: float = 0.80,
    ) -> None:
        from model.adapter import CLIPFeatureExtractor  # local import to match repo style

        self.device = device
        self.extractor = CLIPFeatureExtractor(model_name=clip_model, device=device)
        self.pca_cov_reg = float(pca_cov_reg)
        self.band_lower = float(band_lower)
        self.band_upper = float(band_upper)
        self.top_ratio = float(top_ratio)

    def encode_dataset(
        self,
        dataloader: DataLoader,
        image_adapter,
    ) -> tuple[np.ndarray, np.ndarray]:
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        image_adapter.eval()
        adapter_device = next(image_adapter.parameters()).device

        with torch.no_grad():
            for images, batch_labels in tqdm(dataloader, desc="Encoding CIFAR100", unit="batch"):
                image_features = self.extractor.encode_image(images)
                image_features = image_adapter(image_features.to(adapter_device))
                features.append(image_features.detach().cpu())
                labels.append(batch_labels.detach().cpu())

        feat = torch.cat(features, dim=0).numpy().astype(np.float32, copy=False)
        lab = torch.cat(labels, dim=0).numpy().astype(np.int64, copy=False)
        return feat, lab

    def _compute_pca(
        self,
        class_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if class_features.ndim != 2:
            raise ValueError("class_features must be 2D.")
        n, d = class_features.shape
        mean = np.mean(class_features, axis=0, dtype=np.float64)
        centered = class_features.astype(np.float64, copy=False) - mean[None, :]
        if n <= 1:
            eigvals = np.zeros((d,), dtype=np.float64)
            eigvecs = np.eye(d, dtype=np.float64)
            return centered, eigvals, eigvecs

        cov = (centered.T @ centered) / (n - 1)
        cov += self.pca_cov_reg * np.eye(d, dtype=np.float64)

        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        return centered, eigvals, eigvecs

    def _select_dirs_band(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
        total = float(np.sum(eigvals))
        if total <= 0:
            return eigvecs[:, -1:].copy()

        ratios = eigvals / total
        mask = (ratios >= self.band_lower) & (ratios <= self.band_upper)
        if np.any(mask):
            return eigvecs[:, mask].copy()

        target = 0.5 * (self.band_lower + self.band_upper)
        idx = int(np.argmin(np.abs(ratios - target)))
        return eigvecs[:, idx : idx + 1].copy()

    def _select_dirs_top80(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
        total = float(np.sum(eigvals))
        if total <= 0:
            return eigvecs[:, -1:].copy()

        eigvals_desc = eigvals[::-1]
        eigvecs_desc = eigvecs[:, ::-1]
        ratios_desc = eigvals_desc / total
        cumulative = np.cumsum(ratios_desc)
        keep = int(np.searchsorted(cumulative, self.top_ratio, side="left")) + 1
        keep = min(max(keep, 1), eigvals_desc.shape[0])
        return eigvecs_desc[:, :keep].copy()

    @staticmethod
    def _score_from_dirs(centered: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        proj = centered @ dirs
        return np.mean(np.abs(proj), axis=1, dtype=np.float64)

    @staticmethod
    def _quantile_minmax_per_class(
        values: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        low_q: float = 0.002,
        high_q: float = 0.998,
    ) -> np.ndarray:
        out = np.zeros_like(values, dtype=np.float64)
        for c in range(num_classes):
            mask = labels == c
            if not np.any(mask):
                continue
            cls_vals = values[mask]
            q_low = float(np.quantile(cls_vals, low_q))
            q_high = float(np.quantile(cls_vals, high_q))
            if abs(q_high - q_low) <= 1e-12:
                out[mask] = 0.5
                continue
            scaled = (cls_vals - q_low) / (q_high - q_low)
            out[mask] = np.clip(scaled, 0.0, 1.0)
        return out

    def compare(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
    ) -> DDSCompareResult:
        n = features.shape[0]
        raw_band = np.zeros((n,), dtype=np.float64)
        raw_top80 = np.zeros((n,), dtype=np.float64)

        for class_id in tqdm(range(num_classes), desc="Computing two DDS variants", unit="class"):
            mask = labels == class_id
            if not np.any(mask):
                continue

            cls_features = features[mask]
            centered, eigvals, eigvecs = self._compute_pca(cls_features)

            dirs_band = self._select_dirs_band(eigvals, eigvecs)
            dirs_top80 = self._select_dirs_top80(eigvals, eigvecs)

            raw_band[mask] = self._score_from_dirs(centered, dirs_band)
            raw_top80[mask] = self._score_from_dirs(centered, dirs_top80)

        norm_band = self._quantile_minmax_per_class(raw_band, labels, num_classes)
        norm_top80 = self._quantile_minmax_per_class(raw_top80, labels, num_classes)

        diff_signed = norm_band - norm_top80
        diff_abs = np.abs(diff_signed)

        return DDSCompareResult(
            raw_band=raw_band,
            raw_top80=raw_top80,
            norm_band=norm_band,
            norm_top80=norm_top80,
            diff_signed=diff_signed,
            diff_abs=diff_abs,
            labels=labels,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two independent DDS implementations on CIFAR100 after quantile min-max normalization."
    )
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--seed", type=int, default=96, help="Adapter seed.")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--band-lower", type=float, default=0.02)
    parser.add_argument("--band-upper", type=float, default=0.20)
    parser.add_argument("--top-ratio", type=float, default=0.80)
    parser.add_argument("--pca-cov-reg", type=float, default=1e-6)

    parser.add_argument(
        "--save-json",
        type=str,
        default="picture/dds_compare_cifar100_stats.json",
        help="Path to save summary statistics as json.",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default="picture/dds_compare_cifar100_arrays.npz",
        help="Path to save normalized DDS arrays and differences.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else CONFIG.global_device

    dataset = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        download=True,
        transform=None,
    )

    comparator = IndependentDDSComparator(
        clip_model=args.clip_model,
        device=device,
        pca_cov_reg=args.pca_cov_reg,
        band_lower=args.band_lower,
        band_upper=args.band_upper,
        top_ratio=args.top_ratio,
    )

    image_adapter, _text_adapter, adapter_paths = load_trained_adapters(
        dataset_name="cifar100",
        clip_model=args.clip_model,
        input_dim=comparator.extractor.embed_dim,
        seed=args.seed,
        map_location=device,
    )
    image_adapter = image_adapter.to(device).eval()

    loader = DataLoader(
        datasets.CIFAR100(
            root=args.data_root,
            train=True,
            download=True,
            transform=comparator.extractor.preprocess,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    features, labels = comparator.encode_dataset(loader, image_adapter)
    result = comparator.compare(features, labels, num_classes=len(dataset.classes))
    summary = result.summary()

    print("\n=== DDS comparison on CIFAR100 (after per-class quantile min-max normalization) ===")
    print(f"band DDS version: individual eigenvalue ratio in [{args.band_lower:.4f}, {args.band_upper:.4f}]")
    print(f"top80 DDS version: cumulative leading eigenvalue ratio = {args.top_ratio:.4f}")
    print(f"adapter seed: {args.seed}")
    print(f"image adapter path: {adapter_paths['image_path']}")
    print(f"signed diff mean (band - top80): {summary['signed_diff_mean']:.8f}")
    print(f"signed diff var  (band - top80): {summary['signed_diff_var']:.8f}")
    print(f"abs diff mean: {summary['abs_diff_mean']:.8f}")
    print(f"abs diff var : {summary['abs_diff_var']:.8f}")

    save_json_path = PROJECT_ROOT / args.save_json
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    with save_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "cifar100",
                "seed": int(args.seed),
                "clip_model": args.clip_model,
                "band_lower": float(args.band_lower),
                "band_upper": float(args.band_upper),
                "top_ratio": float(args.top_ratio),
                "pca_cov_reg": float(args.pca_cov_reg),
                "image_adapter_path": str(adapter_paths["image_path"]),
                **summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_npz_path = PROJECT_ROOT / args.save_npz
    save_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_npz_path,
        labels=result.labels.astype(np.int64),
        raw_band=result.raw_band.astype(np.float32),
        raw_top80=result.raw_top80.astype(np.float32),
        norm_band=result.norm_band.astype(np.float32),
        norm_top80=result.norm_top80.astype(np.float32),
        diff_signed=result.diff_signed.astype(np.float32),
        diff_abs=result.diff_abs.astype(np.float32),
    )

    print(f"Saved summary json to: {save_json_path}")
    print(f"Saved arrays npz to: {save_npz_path}")


if __name__ == "__main__":
    main()
