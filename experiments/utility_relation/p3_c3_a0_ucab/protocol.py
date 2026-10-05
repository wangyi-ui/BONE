"""Frozen protocol constants and read-only provenance verification for UCAB."""

from __future__ import annotations

import hashlib
from pathlib import Path


STAGE = "P3-C3-A0"
ARMS = ("CURRENT", "TRUE_BUDGET", "MEAN_BUDGET", "SHUFFLE_BUDGET")
HISTORICAL_PAIRED_SEEDS = (20, 30, 50)
CONFIRMATORY_SEEDS = (140, 150, 170)
DATASETS = ("ALOI", "CCV")
DATASET_CONFIG = {
    "CCV": {"optimizer": "Adam", "semantic_lr": 1e-5, "semantic_exposure": 3},
    "ALOI": {"optimizer": "Adam", "semantic_lr": 1e-4, "semantic_exposure": 1},
}

ROOT = Path(__file__).resolve().parents[3]
DEFINITION_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_definition_preregistered_20261004"
CONFIRMATORY_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_confirmatory_preregistered_20261004"
LEGACY_SEMANTIC_CORE = ROOT / "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py"
LEGACY_SEMANTIC_CORE_SHA256 = "330ac8999db35cae64092fcfeb0dc8dc900370996001a82d214a1eca90233f6b"


def sha256_file(path: Path) -> str:
    """Return a file hash without modifying protocol or source artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "FREEZE_MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"FREEZE_MANIFEST_MISSING:{manifest}")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = directory / name.lstrip(" *")
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"FREEZE_MANIFEST_MISMATCH:{path}")


def _status_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _parent_source_expected() -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for line in (DEFINITION_FREEZE / "PARENT_SOURCE_SHA256.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, raw_path = line.split(maxsplit=1)
        path = Path(raw_path.strip())
        expected[path if path.is_absolute() else ROOT / path] = digest
    return expected


def verify_preregistered_protocol() -> dict[str, str]:
    """Fail closed unless both freezes and every required parent hash match."""
    _verify_manifest(DEFINITION_FREEZE)
    _verify_manifest(CONFIRMATORY_FREEZE)
    definition = _status_values(DEFINITION_FREEZE / "STATUS.txt")
    confirmatory = _status_values(CONFIRMATORY_FREEZE / "STATUS.txt")
    if definition.get("DEFINITION_FROZEN") != "true":
        raise RuntimeError("DEFINITION_NOT_FROZEN")
    if confirmatory.get("STATUS") != "P3_C3_A0_UCAB_CONFIRMATORY_PREREGISTERED":
        raise RuntimeError("CONFIRMATORY_STATUS_MISMATCH")
    if confirmatory.get("NEXT_STAGE") != "UCAB_IMPLEMENTATION":
        raise RuntimeError("CONFIRMATORY_NEXT_STAGE_MISMATCH")

    expected = _parent_source_expected()
    legacy_expected = expected.get(LEGACY_SEMANTIC_CORE)
    if legacy_expected != LEGACY_SEMANTIC_CORE_SHA256:
        raise RuntimeError("LEGACY_SEMANTIC_CORE_EXPECTED_HASH_MISMATCH")
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"PARENT_SOURCE_HASH_MISMATCH:{path}")
    return {str(path): digest for path, digest in expected.items()}
