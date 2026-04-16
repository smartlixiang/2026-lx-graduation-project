"""Migrate legacy dynamic_components.npz into per-component cache files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

COMPONENTS = ("A", "C", "D", "E")


def iter_old_dynamic_cache_files(root: Path):
    yield from sorted(root.rglob("dynamic_components.npz"))


def parse_old_cache_path(old_path: Path, root: Path) -> tuple[str, str, int]:
    rel = old_path.relative_to(root)
    if len(rel.parts) != 4 or rel.parts[-1] != "dynamic_components.npz":
        raise ValueError(f"unexpected cache path layout: {old_path}")
    dataset, proxy_model, epochs_str, _ = rel.parts
    try:
        epochs = int(epochs_str)
    except ValueError as exc:
        raise ValueError(f"invalid epochs segment in path: {old_path}") from exc
    return dataset, proxy_model, epochs


def _read_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(()).item()
    return value


def build_component_payload_from_legacy(
    *,
    legacy: np.lib.npyio.NpzFile,
    component_name: str,
    dataset_from_path: str,
    proxy_model_from_path: str,
    epochs_from_path: int,
) -> dict[str, np.ndarray]:
    required_common = ["seed", "labels", "proxy_log_path"]
    required_component = [
        f"{component_name}_raw_foldwise",
        f"{component_name}_fold_normalized",
        f"{component_name}_aggregated",
        f"{component_name}_final_normalized",
    ]
    missing = [k for k in required_common + required_component if k not in legacy.files]
    if missing:
        raise ValueError(f"legacy cache missing keys for {component_name}: {missing}")

    labels = legacy["labels"].astype(np.int64)
    num_samples = int(labels.shape[0])
    dataset = str(_read_scalar(legacy, "dataset", dataset_from_path))
    proxy_training_seed = int(_read_scalar(legacy, "seed"))
    seed_free = bool(_read_scalar(legacy, "seed_free", True))
    proxy_log_path = str(_read_scalar(legacy, "proxy_log_path"))

    payload = {
        "component_name": np.array(component_name, dtype=np.str_),
        "dataset": np.array(dataset, dtype=np.str_),
        "proxy_model": np.array(proxy_model_from_path, dtype=np.str_),
        "proxy_training_seed": np.array(proxy_training_seed, dtype=np.int64),
        "seed_free": np.array(seed_free),
        "epochs": np.array(int(epochs_from_path), dtype=np.int64),
        "proxy_log_path": np.array(proxy_log_path, dtype=np.str_),
        "num_samples": np.array(num_samples, dtype=np.int64),
        "labels": labels,
        "raw_foldwise": legacy[f"{component_name}_raw_foldwise"].astype(np.float32),
        "fold_normalized": legacy[f"{component_name}_fold_normalized"].astype(np.float32),
        "aggregated": legacy[f"{component_name}_aggregated"].astype(np.float32),
        "final_normalized": legacy[f"{component_name}_final_normalized"].astype(np.float32),
    }
    return payload


def validate_new_component_payload(payload: dict[str, np.ndarray], expected_component: str) -> None:
    required = {
        "component_name",
        "dataset",
        "proxy_training_seed",
        "seed_free",
        "proxy_log_path",
        "num_samples",
        "labels",
        "raw_foldwise",
        "fold_normalized",
        "aggregated",
        "final_normalized",
        "proxy_model",
        "epochs",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"payload missing keys: {sorted(missing)}")

    component_name = str(payload["component_name"].item())
    if component_name != expected_component:
        raise ValueError(f"component mismatch: expect {expected_component}, got {component_name}")

    labels = np.asarray(payload["labels"])
    final_norm = np.asarray(payload["final_normalized"])
    num_samples = int(np.asarray(payload["num_samples"]).item())
    if labels.ndim != 1:
        raise ValueError("labels must be 1D.")
    if num_samples != labels.shape[0]:
        raise ValueError(f"num_samples mismatch with labels length: {num_samples} vs {labels.shape[0]}")
    if final_norm.shape != (num_samples,):
        raise ValueError(f"final_normalized shape mismatch: {final_norm.shape}")
    if not np.all(np.isfinite(final_norm)):
        raise ValueError("final_normalized contains NaN/inf.")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels contains NaN/inf.")


def _extract_payload_from_new_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _existing_cache_matches(path: Path, payload: dict[str, np.ndarray], component_name: str) -> bool:
    if not path.is_file():
        return False
    try:
        existing = _extract_payload_from_new_cache(path)
        validate_new_component_payload(existing, expected_component=component_name)
    except Exception:
        return False

    checks = [
        np.array_equal(existing["labels"], payload["labels"]),
        np.allclose(existing["final_normalized"], payload["final_normalized"], rtol=0.0, atol=0.0),
        str(existing["dataset"].item()) == str(payload["dataset"].item()),
        int(existing["proxy_training_seed"].item()) == int(payload["proxy_training_seed"].item()),
        int(existing["num_samples"].item()) == int(payload["num_samples"].item()),
    ]
    return all(checks)


def write_component_cache(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def migrate_one_legacy_cache(old_path: Path, root: Path, *, dry_run: bool = False, verbose: bool = False) -> bool:
    try:
        dataset, proxy_model, epochs = parse_old_cache_path(old_path, root)
        with np.load(old_path, allow_pickle=True) as legacy:
            payloads = {
                component: build_component_payload_from_legacy(
                    legacy=legacy,
                    component_name=component,
                    dataset_from_path=dataset,
                    proxy_model_from_path=proxy_model,
                    epochs_from_path=epochs,
                )
                for component in COMPONENTS
            }
    except Exception as exc:
        print(f"[SKIP] failed to migrate old cache: {old_path} ({exc})")
        return False

    target_paths = {component: old_path.with_name(f"{component}.npz") for component in COMPONENTS}
    try:
        for component, payload in payloads.items():
            validate_new_component_payload(payload, expected_component=component)
            target_path = target_paths[component]
            if _existing_cache_matches(target_path, payload, component_name=component):
                if verbose:
                    print(f"[OK] already migrated component {component}: {target_path}")
                continue
            if dry_run:
                print(f"[DRY-RUN] migrate {component}: {old_path} -> {target_path}")
                continue
            write_component_cache(target_path, payload)
            if verbose:
                print(f"[OK] wrote component {component}: {target_path}")
    except Exception as exc:
        print(f"[SKIP] failed to migrate old cache: {old_path} ({exc})")
        return False

    if dry_run:
        print(f"[DRY-RUN] would remove old cache: {old_path}")
        return True

    try:
        for component in COMPONENTS:
            payload = _extract_payload_from_new_cache(target_paths[component])
            validate_new_component_payload(payload, expected_component=component)
        old_path.unlink()
        print(f"[OK] migrated and removed old cache: {old_path}")
        return True
    except Exception as exc:
        print(f"[SKIP] failed to migrate old cache: {old_path} ({exc})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate dynamic_components.npz to per-component cache files.")
    parser.add_argument("--root", type=str, default="weights/dynamic_cache", help="Dynamic cache root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned migrations without writing/deleting files.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed migration logs.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[INFO] root does not exist, nothing to migrate: {root}")
        return

    old_files = list(iter_old_dynamic_cache_files(root))
    if not old_files:
        print(f"[INFO] no legacy caches found under: {root}")
        return

    ok_count = 0
    skip_count = 0
    for old_path in old_files:
        migrated = migrate_one_legacy_cache(old_path, root, dry_run=args.dry_run, verbose=args.verbose)
        if migrated:
            ok_count += 1
        else:
            skip_count += 1

    print(f"[DONE] processed={len(old_files)}, migrated={ok_count}, skipped={skip_count}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
