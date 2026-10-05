"""Immutable protocol and integrity checks for P3-C4-A0.

This package is deliberately diagnostic-only: it has no optimiser construction,
no metric imports, and no full-label input surface.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

STAGE = "P3-C4-A0"
DIAGNOSTIC_ONLY = True
NEW_METHOD = HYPERPARAMETER_SELECTION = GT_ALLOWED = FORMAL_CLUSTERING_METRICS = False
DATASETS = ("HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV")
SEEDS = (20, 30, 50)
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs/bone_p3/p3_c4_a0_utility_action_alignment"
LOG_ROOT = ROOT / "logs/bone_p3/p3_c4_a0_utility_action_alignment"
TARGET_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_confirmatory_fail_closed_20261005"

# The sources defining the frozen native and CURRENT semantic primitives.
FROZEN_SOURCE_HASHES = {
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
    "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py": "330ac8999db35cae64092fcfeb0dc8dc900370996001a82d214a1eca90233f6b",
    "experiments/utility_relation/p3_c3_a0_ucab/semantic_epoch.py": "5feb47cdf4ba02964c3e6baefdf3de11bb25b4c843ee8632d85f20e931152f20",
    "experiments/utility_relation/p3_c3_a0_ucab/protocol.py": "a729a49266daa968c86da194729d683825c655ecdb7fb1fda8467c9509a31db1",
    "experiment_freeze/bone_p3_c3_a0_ucab_confirmatory_fail_closed_20261005/FREEZE_MANIFEST.sha256": "1612cc938c7618175b91ad1818b810fb12aaed7f519ef1f4f55ea01650ea06f0",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_sources() -> dict[str, str]:
    """Fail closed when any owned frozen input drifted."""
    observed = {name: sha256_file(ROOT / name) for name in FROZEN_SOURCE_HASHES}
    if observed != FROZEN_SOURCE_HASHES:
        raise RuntimeError(f"FROZEN_SOURCE_HASH_MISMATCH:{observed}")
    # Hashing the manifest alone is insufficient: verify every artifact it owns.
    for line in (TARGET_FREEZE / "FREEZE_MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        artifact = TARGET_FREEZE / relative.lstrip(" *")
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"TARGET_C3_FREEZE_MANIFEST_MISMATCH:{artifact}")
    status = (TARGET_FREEZE / "STATUS.txt").read_text()
    if "FINAL_CONFIRMATORY_VERDICT=FAIL_CLOSED" not in status:
        raise RuntimeError("TARGET_C3_FREEZE_STATUS_MISMATCH")
    return observed


def manifest_cells() -> tuple[tuple[str, int], ...]:
    return tuple((dataset, seed) for dataset in DATASETS for seed in SEEDS)


def require_cell(dataset: str, seed: int) -> None:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("UNREGISTERED_DATASET_OR_TRAINING_SEED")
