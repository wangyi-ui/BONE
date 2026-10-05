"""CPU-only P3-A4 mask artifact materializer; it never trains or corrupts features."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from experiments.utility_relation import bone_p3_a4_corruption_protocol as protocol
from utils.data_util import dataset_info


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3" / "corruptions"
READY_DATASETS = ("MNIST-USPS", "BDGP", "Caltech-5V", "HandWrite", "COIL20", "MSRC", "ALOI")
BLOCKED_DATASETS = ("CCV", "YouTubeFace50")
P1_ARTIFACT = REPO_ROOT / "outputs" / "bone_p1" / "msrc_snr2p5_exact2of4_seed20" / "bone_p1_msrc_corruption.npz"
P1_ARTIFACT_SHA256 = "2b05b7b5adfe872c60475fe8d3d9f4959229a57ffb1eb50ff29adb015f38db87"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_mask_npz(path: Path, mask: np.ndarray, raw_row_ids: np.ndarray) -> None:
    """Write exactly the two required arrays with reproducible ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in (("corruption_mask.npy", mask), ("raw_row_ids.npy", raw_row_ids)):
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(array), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _legacy_msrc_mask() -> tuple[np.ndarray, np.ndarray]:
    if sha256_file(P1_ARTIFACT) != P1_ARTIFACT_SHA256:
        raise ValueError("historical P1 MSRC artifact SHA mismatch")
    with np.load(P1_ARTIFACT, allow_pickle=False) as archive:
        mask = np.asarray(archive["corruption_mask"], dtype=bool)
        raw_row_ids = np.asarray(archive["row_ids"], dtype=np.int64)
    protocol.validate_balanced_half_view_mask(mask, "MSRC", protocol.CORRUPTION_SEED)
    return mask, raw_row_ids


def _record(dataset_name: str, mask: np.ndarray, raw_row_ids: np.ndarray, source_sha: str, mode: str, npz_path: Path) -> dict[str, object]:
    n, v = mask.shape
    row_sums = mask.sum(axis=1)
    record = {
        "dataset_name": dataset_name,
        "N": n,
        "V": v,
        "snr_db": protocol.TARGET_SNR_DB,
        "corruption_seed": protocol.CORRUPTION_SEED,
        "mask_mode": mode,
        "row_id_semantics": "RAW_FILE_ROW_ID",
        "raw_row_ids_sha256": hashlib.sha256(np.ascontiguousarray(raw_row_ids).tobytes()).hexdigest(),
        "total_corrupted_cells": int(mask.sum()),
        "corruption_fraction": float(mask.mean()),
        "row_sum_histogram": {str(int(value)): int(count) for value, count in zip(*np.unique(row_sums, return_counts=True))},
        "pattern_counts": protocol.pattern_counts(mask),
        "per_view_counts": [int(value) for value in mask.sum(axis=0)],
        "mask_sha256": protocol.mask_sha256(mask),
        "source_dataset_sha256": source_sha,
        "historical_noise_formula_source": protocol.HISTORICAL_NOISE_FORMULA_SOURCE,
        "preprocessing_insertion_point": "native cast/transpose -> native shared shuffle with explicit raw-ID mapping -> per-view StandardScaler -> corruption mask/noise -> no post-corruption standardization",
        "training_seed_independent": True,
        "npz_path": _display_path(npz_path),
        "npz_sha256": sha256_file(npz_path),
    }
    return record


def _display_path(path: Path) -> str:
    """Keep repository audits relocatable without rejecting test output paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def materialize_dataset(dataset_name: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Materialize only a raw-row mask for one approved dataset."""
    if dataset_name not in READY_DATASETS:
        raise ValueError(f"{dataset_name} is not an approved P3-A4 materialization target")
    if dataset_name == "MSRC":
        mask, raw_row_ids = _legacy_msrc_mask()
        source_sha, mode = P1_ARTIFACT_SHA256, "LEGACY_FROZEN"
    else:
        if dataset_name not in dataset_info:
            raise ValueError(f"missing source metadata for {dataset_name}")
        n, v, _k, _dims = dataset_info[dataset_name]
        source_path = DATA_DIR / f"{dataset_name}.mat"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        mask = protocol.build_balanced_half_view_mask(dataset_name, n, v)
        raw_row_ids = np.arange(n, dtype=np.int64)
        source_sha, mode = sha256_file(source_path), "P3_A4_SHA_BALANCED_HALF_VIEW_V1"
    if not np.array_equal(raw_row_ids, np.arange(mask.shape[0], dtype=np.int64)):
        raise ValueError("raw row ID mapping must be canonical arange(N)")
    protocol.validate_balanced_half_view_mask(mask, dataset_name)
    npz_path = output_dir / f"{dataset_name}_mask.npz"
    write_deterministic_mask_npz(npz_path, mask, raw_row_ids)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"corruption_mask", "raw_row_ids"}:
            raise ValueError("mask artifact schema must contain only mask and raw row IDs")
        if not np.array_equal(archive["corruption_mask"], mask) or not np.array_equal(archive["raw_row_ids"], raw_row_ids):
            raise ValueError("mask artifact contents do not match the protocol")
    record = _record(dataset_name, mask, raw_row_ids, source_sha, mode, npz_path)
    (output_dir / f"{dataset_name}_mask_audit.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def materialize_all(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    """Materialize mask-only artifacts for exactly the seven READY datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [materialize_dataset(dataset_name, output_dir) for dataset_name in READY_DATASETS]
    blocked = [
        {"dataset_name": dataset_name, "status": "BLOCKED_MISSING_DATA", "mask_materialized": False,
         "reason": "dataset file is absent; no mask is guessed or generated"}
        for dataset_name in BLOCKED_DATASETS
    ]
    audit = {
        "protocol": "P3-A4 balanced half-view masks; MSRC uses the historical frozen P1 artifact",
        "records": records,
        "blocked": blocked,
        "mask_only": True,
        "no_feature_corruption_materialized": True,
        "no_training": True,
        "no_gpu": True,
    }
    (output_dir / "corruption_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    audit = materialize_all(build_parser().parse_args().output_dir)
    print(json.dumps({"materialized": [record["dataset_name"] for record in audit["records"]], "blocked": [record["dataset_name"] for record in audit["blocked"]]}, indent=2))


if __name__ == "__main__":
    main()
