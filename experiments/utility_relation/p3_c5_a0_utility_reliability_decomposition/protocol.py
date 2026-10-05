"""Frozen, non-tunable protocol for P3-C5-A0."""
from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.utility_relation.p3_c4_a0_utility_action_alignment.protocol import DATASETS, SEEDS

STAGE = "P3-C5-A0"
DIAGNOSTIC_ONLY = True
GAMMA_PROHIBITED = True
ROOT = Path(__file__).resolve().parents[3]
PARENT_FREEZE = ROOT / "experiment_freeze/bone_p3_c4_a0_full24_scientific_route_review_20261005"
OUTPUT_ROOT = ROOT / "outputs/bone_p3/p3_c5_a0_utility_reliability_decomposition"

COMPONENTS = ("closure", "conf_gen", "support_ver")
REQUIRED_CARRIER_FIELDS = frozenset((
    "query_rows", "anchor_rows", "T_rel", "U_cycle", "query_U_cycle",
    "closure", "conf_gen", "support_ver",
))
REQUIRED_OUTPUT_KEYS = frozenset((
    "stage", "dataset", "training_seed", "DIAGNOSTIC_ONLY", "full_GT_loaded",
    "formal_clustering_metrics_computed", "training_performed", "parameter_update_performed",
    "U_COMPONENT_RECONSTRUCTION_PARITY", "CLOSURE_ZERO_IMPLIES_U_ZERO",
    "A_CLOSURE_1", "A_CLOSURE_0", "Delta_CLOSURE",
    "A_CONF_Q1", "A_CONF_Q2", "A_CONF_Q3", "A_CONF_Q4", "A_CONF_Q5",
    "Delta_CONF_Q5_Q1", "Spearman_CONF_ALIGNMENT",
    "A_SUPPORT_Q1", "A_SUPPORT_Q2", "A_SUPPORT_Q3", "A_SUPPORT_Q4", "A_SUPPORT_Q5",
    "Delta_SUPPORT_Q5_Q1", "Spearman_SUPPORT_ALIGNMENT",
    "C4_U_DeltaAlignment_Q5_Q1_replication", "C4_U_Spearman_quintile_alignment_replication",
    "C4_U_REPLICATION_PASS", "closure_action_count_0", "closure_action_count_1",
    "conf_conditioned_action_count", "support_conditioned_action_count", "invalid_zero_norm_count",
    "valid_batch_count", "detach_contract", "model_state_sha256_before",
    "model_state_sha256_after", "model_state_unchanged", "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
))

# Preregistered gates; deliberately not exposed through a command line.
CELL_COUNT = 24
POSITIVE_CELL_MINIMUM = 16
POSITIVE_DATASET_MINIMUM = 6
ROUTE_STRUCTURAL_AWARE = "STRUCTURAL_AWARE_ACTION_VALUE_DESIGN"
ROUTE_SALVAGE = "MINIMAL_COMPONENT_SALVAGE_CONFIRMATION"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "FREEZE_MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("PARENT_FREEZE_MANIFEST_MISSING")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        artifact = directory / relative.lstrip(" *")
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise RuntimeError(f"PARENT_FREEZE_MANIFEST_MISMATCH:{artifact}")


def _verify_checksum_listing(path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, raw_path = line.split(maxsplit=1)
        source = Path(raw_path.strip())
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"PARENT_C4_SOURCE_MISMATCH:{source}")
        observed[str(source)] = expected
    if not observed:
        raise RuntimeError("PARENT_C4_CHECKSUM_LIST_EMPTY")
    return observed


def verify_parent_freeze() -> dict[str, dict[str, str]]:
    """Verify parent manifest plus the C4 source and test receipts, read-only."""
    _verify_manifest(PARENT_FREEZE)
    sources = _verify_checksum_listing(PARENT_FREEZE / "C4_SOURCE_SHA256.txt")
    tests = _verify_checksum_listing(PARENT_FREEZE / "C4_TEST_SHA256.txt")
    status = (PARENT_FREEZE / "STATUS.txt").read_text()
    conclusion = (PARENT_FREEZE / "SCIENTIFIC_CONCLUSION.txt").read_text()
    if ("STATUS=P3_C4_A0_FULL24_SCIENTIFIC_DIAGNOSTIC_FROZEN" not in status or
            "RELATIVE_U_ALIGNMENT_SUPPORTED=false" not in conclusion):
        raise RuntimeError("PARENT_C4_FREEZE_STATUS_MISMATCH")
    return {"C4_SOURCE_SHA256": sources, "C4_TEST_SHA256": tests}


def manifest_cells() -> tuple[tuple[str, int], ...]:
    return tuple((dataset, seed) for dataset in DATASETS for seed in SEEDS)


def require_cell(dataset: str, seed: int) -> None:
    if (dataset, seed) not in set(manifest_cells()):
        raise ValueError("UNREGISTERED_DATASET_OR_TRAINING_SEED")
