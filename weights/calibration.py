"""Shared calibration primitives used by the main and unseen experiments."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from utils.score_utils import standard_zscore
from weights.dynamic_utils import DynamicComponentResult


def build_dynamic_target(
    component_results: dict[str, DynamicComponentResult],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = {name: component_results[name].final_normalized for name in ("A", "C", "T")}
    num_samples = components["A"].shape[0]
    for name, values in components.items():
        if values.shape != (num_samples,):
            raise ValueError(f"{name}_final_normalized shape mismatch: {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}_final_normalized contains NaN/inf values.")
    utility_raw = sum(components.values()).astype(np.float64) / 3.0
    return standard_zscore(utility_raw).astype(np.float64), components


def fit_softplus_ratio_regression(
    features: np.ndarray,
    targets: np.ndarray,
    ratio_lambda: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
    device: torch.device,
) -> dict[str, object]:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets shape mismatch.")
    features_t = torch.as_tensor(features, dtype=torch.float64, device=device)
    targets_t = torch.as_tensor(targets, dtype=torch.float64, device=device)
    theta = torch.zeros(features.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.tensor(float(np.mean(targets)), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta, bias], lr=float(learning_rate))
    prev_loss: float | None = None
    iterations = 0
    final_mse = final_ratio = 0.0
    for iteration in tqdm(range(int(max_iter)), desc="fit softplus-ratio", unit="iter", leave=False):
        optimizer.zero_grad()
        raw_weights = torch.nn.functional.softplus(theta) + 1e-8
        pred = features_t @ raw_weights + bias
        mse = torch.mean((pred - targets_t) ** 2)
        ratio_reg = torch.var(raw_weights, unbiased=False) / (torch.mean(raw_weights) ** 2 + 1e-8)
        loss = mse + float(ratio_lambda) * ratio_reg
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        final_mse = float(mse.detach().cpu())
        final_ratio = float(ratio_reg.detach().cpu())
        iterations = iteration + 1
        if prev_loss is not None and abs(prev_loss - loss_value) < float(tol):
            break
        prev_loss = loss_value
    with torch.no_grad():
        raw = (torch.nn.functional.softplus(theta) + 1e-8).cpu().numpy().astype(np.float64)
        pred = (features_t @ torch.as_tensor(raw, device=device) + bias).cpu().numpy().astype(np.float64)
    return {
        "raw_weights": raw,
        "normalized_weights": raw / float(raw.sum()),
        "bias": float(bias.detach().cpu()),
        "mse": final_mse,
        "ratio_regularizer": final_ratio,
        "iterations": iterations,
        "pred": pred,
    }
