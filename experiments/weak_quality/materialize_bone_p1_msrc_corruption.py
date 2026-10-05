"""Materialize the fixed canonical-row MSRC weak-quality artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.weak_quality.bone_p1_msrc_protocol import (
    CORRUPTION_SEED,
    FEATURE_DIMS,
    K,
    N,
    TARGET_SNR_DB,
    V,
    build_corruption_mask,
    pair_counts,
    validate_mask,
)
from utils.data_util import D_MSRC, get_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/bone_p1/msrc_snr2p5_exact2of4_seed20"
ARTIFACT_NAME = "bone_p1_msrc_corruption.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_clean_views(clean_views: Sequence[np.ndarray]) -> None:
    if len(clean_views) != V:
        raise ValueError(f"expected {V} views, got {len(clean_views)}")
    for view_index, view in enumerate(clean_views):
        if view.shape != (N, FEATURE_DIMS[view_index]):
            raise ValueError(
                f"view {view_index} must be {(N, FEATURE_DIMS[view_index])}, got {view.shape}"
            )
        if not np.isfinite(view).all():
            raise ValueError(f"clean view {view_index} contains NaN or Inf")


def corrupt_standardized_views(
    clean_views: Sequence[np.ndarray], mask: np.ndarray, corruption_seed: int = CORRUPTION_SEED,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """Apply SNR-calibrated Gaussian noise without accepting or inspecting labels.

    Args:
        clean_views: x_clean[v], each [N, D_v].
        mask: [N, V], with True where view v is corrupted for row i.

    Returns:
        x_corrupt[v] [N, D_v], noise[v] [N, D_v], alpha_per_view, achieved_snr_db.
    """
    _validate_clean_views(clean_views)
    validate_mask(mask)
    # default_rng is deliberately reserved for row assignment in build_corruption_mask.
    noise_rng = np.random.RandomState(corruption_seed)
    corrupt_views: List[np.ndarray] = []
    noise_views: List[np.ndarray] = []
    alphas: List[float] = []
    achieved_snr_db: List[float] = []
    snr_scale = 10.0 ** (TARGET_SNR_DB / 10.0)

    for view_index, clean in enumerate(clean_views):
        omega = mask[:, view_index]
        noise = noise_rng.normal(0.0, 1.0, size=clean.shape).astype(clean.dtype, copy=False)
        signal_energy = float(np.sum(np.square(clean[omega]), dtype=np.float64))
        noise_energy = float(np.sum(np.square(noise[omega]), dtype=np.float64))
        if not np.isfinite(signal_energy) or not np.isfinite(noise_energy) or noise_energy <= 0.0:
            raise ValueError(f"invalid energy for view {view_index}")
        alpha = float(np.sqrt(signal_energy / (noise_energy * snr_scale)))
        corrupt = clean.copy()
        corrupt[omega] += alpha * noise[omega]
        if not np.isfinite(corrupt).all():
            raise ValueError(f"corruption produced NaN or Inf for view {view_index}")
        if not np.array_equal(clean[~omega], corrupt[~omega]):
            raise ValueError(f"uncorrupted locations changed in view {view_index}")
        if not np.any(clean[omega] != corrupt[omega]):
            raise ValueError(f"corrupted locations did not change in view {view_index}")
        perturbation_energy = float(np.sum(np.square(corrupt[omega] - clean[omega]), dtype=np.float64))
        achieved = float(10.0 * np.log10(signal_energy / perturbation_energy))
        if not np.isfinite(achieved) or abs(achieved - TARGET_SNR_DB) > 0.05:
            raise ValueError(f"view {view_index} achieved SNR {achieved}, outside tolerance")
        corrupt_views.append(corrupt)
        noise_views.append(noise)
        alphas.append(alpha)
        achieved_snr_db.append(achieved)
    return corrupt_views, noise_views, alphas, achieved_snr_db


def _audit(
    mask: np.ndarray, alphas: Sequence[float], achieved_snr_db: Sequence[float], clean_views: Sequence[np.ndarray],
    corrupt_views: Sequence[np.ndarray],
) -> Dict[str, object]:
    unchanged = [bool(np.array_equal(clean[~mask[:, v]], corrupt[~mask[:, v]])) for v, (clean, corrupt) in
                 enumerate(zip(clean_views, corrupt_views))]
    return {
        "N": N,
        "V": V,
        "K": K,
        "feature_dims": list(FEATURE_DIMS),
        "target_snr_db": TARGET_SNR_DB,
        "achieved_snr_db": [float(value) for value in achieved_snr_db],
        "corruption_seed": CORRUPTION_SEED,
        "total_corrupted_pairs": int(mask.sum()),
        "corrupted_count_per_view": [int(value) for value in mask.sum(axis=0)],
        "pair_counts": pair_counts(mask),
        "alpha_per_view": [float(value) for value in alphas],
        "clean_locations_exactly_unchanged": bool(all(unchanged)),
        "clean_locations_exactly_unchanged_per_view": unchanged,
        "labels_used_for_corruption": False,
        "post_corruption_standardization": False,
        "artifact_schema": {
            "x_clean_v*": "numeric ndarray [N, D_v]",
            "x_corrupt_v*": "numeric ndarray [N, D_v]",
            "corruption_mask": "bool ndarray [N, V]",
            "labels": "numeric ndarray [N] (evaluation only)",
            "row_ids": "int64 ndarray [N]",
        },
    }


def _validate_audit(audit: Dict[str, object]) -> None:
    if audit["total_corrupted_pairs"] != 420 or audit["corrupted_count_per_view"] != [105] * V:
        raise ValueError("audit mask counts are invalid")
    if audit["pair_counts"] != {f"{a}-{b}": 35 for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))}:
        raise ValueError("audit pair counts are invalid")
    if not audit["clean_locations_exactly_unchanged"]:
        raise ValueError("audit reports changed clean locations")
    if audit["labels_used_for_corruption"] or audit["post_corruption_standardization"]:
        raise ValueError("forbidden corruption behavior recorded")
    if any(abs(float(value) - TARGET_SNR_DB) > 0.05 for value in audit["achieved_snr_db"]):
        raise ValueError("audit SNR is outside tolerance")


def materialize(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, Path]:
    """Create the sealed artifact. Validation precedes any publishable output."""
    output_dir = Path(output_dir)
    # Canonical rows are loaded then standardized exactly once; shuffle is intentionally never called.
    dataset = get_dataset(D_MSRC)
    dataset.standard_data()
    clean_views = [np.ascontiguousarray(view.T.copy()) for view in dataset.data]  # x_clean[v]: [N, D_v]
    labels = np.ascontiguousarray(dataset.label.copy())
    _validate_clean_views(clean_views)
    if labels.shape != (N,) or not np.issubdtype(labels.dtype, np.number):
        raise ValueError("MSRC labels do not match the fixed artifact schema")

    mask = build_corruption_mask()
    corrupt_views, _noise_views, alphas, achieved_snr_db = corrupt_standardized_views(clean_views, mask)
    audit = _audit(mask, alphas, achieved_snr_db, clean_views, corrupt_views)
    _validate_audit(audit)  # Fail before a possibly formal artifact is written.

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bone_p1_materialize_", dir=output_dir) as staging_name:
        staging = Path(staging_name)
        artifact_path = staging / ARTIFACT_NAME
        arrays: Dict[str, np.ndarray] = {
            "corruption_mask": mask,
            "labels": labels,
            "row_ids": np.arange(N, dtype=np.int64),
        }
        for view_index in range(V):
            arrays[f"x_clean_v{view_index}"] = clean_views[view_index]
            arrays[f"x_corrupt_v{view_index}"] = corrupt_views[view_index]
        np.savez_compressed(artifact_path, **arrays)
        with np.load(artifact_path, allow_pickle=False) as archive:
            if any(archive[key].dtype == object for key in archive.files):
                raise ValueError("artifact contains forbidden object array")
        audit_path = staging / "corruption_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal = {
            "sha256": {
                "bone_p1_msrc_corruption.npz": sha256_file(artifact_path),
                "corruption_audit.json": sha256_file(audit_path),
                "bone_p1_msrc_protocol.py": sha256_file(REPO_ROOT / "experiments/weak_quality/bone_p1_msrc_protocol.py"),
                "materialize_bone_p1_msrc_corruption.py": sha256_file(Path(__file__).resolve()),
                "method/bone.py": sha256_file(REPO_ROOT / "method/bone.py"),
                "method/amv.py": sha256_file(REPO_ROOT / "method/amv.py"),
            }
        }
        seal_path = staging / "corruption_seal.json"
        seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        published = {
            "artifact": output_dir / ARTIFACT_NAME,
            "audit": output_dir / "corruption_audit.json",
            "seal": output_dir / "corruption_seal.json",
        }
        for temporary, final in ((artifact_path, published["artifact"]), (audit_path, published["audit"]),
                                 (seal_path, published["seal"])):
            os.replace(temporary, final)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = materialize(args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
