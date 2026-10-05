"""Fail-closed shared provenance and sealing helpers for UCAB validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol

from . import protocol as ucab_protocol


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs/bone_p3/p3_c3_a0_ucab/dataset_validation"
DEFINITION_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_definition_preregistered_20261004"
CONFIRMATORY_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_confirmatory_preregistered_20261004"
CPU_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_cpu_implementation_pass_20261004"
SMOKE_FREEZE = ROOT / "experiment_freeze/bone_p3_c3_a0_ucab_engineering_smoke_pass_20261004"
FREEZES = (DEFINITION_FREEZE, CONFIRMATORY_FREEZE, CPU_FREEZE, SMOKE_FREEZE)
HISTORICAL_ARMS = ("CURRENT", "TRUE_BUDGET")
CONFIRMATORY_ARMS = ("CURRENT", "TRUE_BUDGET", "MEAN_BUDGET", "SHUFFLE_BUDGET")
METRICS = ("ACC", "NMI", "ARI")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def read_json(path: str | Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"INVALID_{label}:{path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"INVALID_{label}:{path}")
    return value


def _verify_manifest(directory: Path) -> None:
    manifest = directory / "FREEZE_MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"FREEZE_MANIFEST_MISSING:{manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, raw_name = line.split(maxsplit=1)
            path = directory / raw_name.lstrip(" *")
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"FREEZE_MANIFEST_MISMATCH:{path}")


def _verify_hash_list(path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, raw_name = line.split(maxsplit=1)
            candidate = Path(raw_name.strip())
            actual = sha256_file(candidate) if candidate.is_file() else None
            if actual != expected:
                raise RuntimeError(f"FROZEN_SOURCE_HASH_MISMATCH:{candidate}")
            observed[str(candidate)] = actual
    return observed


def verify_parent_freezes() -> dict[str, dict[str, str]]:
    """Verify every parent manifest and all source-hash lists before execution."""
    for directory in FREEZES:
        _verify_manifest(directory)
    cpu_implementation = _verify_hash_list(CPU_FREEZE / "IMPLEMENTATION_SOURCE_SHA256.txt")
    cpu_legacy = _verify_hash_list(CPU_FREEZE / "LEGACY_SOURCE_SHA256.txt")
    smoke_source = _verify_hash_list(SMOKE_FREEZE / "SMOKE_SOURCE_SHA256.txt")
    smoke_ucab = _verify_hash_list(SMOKE_FREEZE / "FROZEN_UCAB_SOURCE_SHA256.txt")
    smoke_legacy = _verify_hash_list(SMOKE_FREEZE / "LEGACY_SOURCE_SHA256.txt")
    if sha256_file(ucab_protocol.LEGACY_SEMANTIC_CORE) != ucab_protocol.LEGACY_SEMANTIC_CORE_SHA256:
        raise RuntimeError("LEGACY_SEMANTIC_CORE_HASH_MISMATCH")
    ucab_protocol.verify_preregistered_protocol()
    c0_protocol.verify_native_hashes()
    c0_protocol.verify_parent_freeze_hashes()
    return {
        "cpu_implementation": cpu_implementation,
        "cpu_legacy": cpu_legacy,
        "smoke_source": smoke_source,
        "smoke_ucab": smoke_ucab,
        "smoke_legacy": smoke_legacy,
    }


def validate_dataset_seed(dataset: str, training_seed: int) -> None:
    if dataset not in ucab_protocol.DATASETS:
        raise ValueError("UNREGISTERED_UCAB_DATASET")
    if training_seed not in (*ucab_protocol.HISTORICAL_PAIRED_SEEDS, *ucab_protocol.CONFIRMATORY_SEEDS):
        raise ValueError("UNREGISTERED_UCAB_TRAINING_SEED")


def seed_role(training_seed: int) -> str:
    validate_dataset_seed("ALOI", training_seed)
    return "historical" if training_seed in ucab_protocol.HISTORICAL_PAIRED_SEEDS else "confirmatory"


def arms_for_seed(training_seed: int) -> tuple[str, ...]:
    return HISTORICAL_ARMS if seed_role(training_seed) == "historical" else CONFIRMATORY_ARMS


def semantic_config(dataset: str) -> dict[str, object]:
    validate_dataset_seed(dataset, 20)
    config = ucab_protocol.DATASET_CONFIG[dataset]
    if config["optimizer"] != "Adam":
        raise RuntimeError("FROZEN_SEMANTIC_OPTIMIZER_DRIFT")
    return dict(config)


def run_root(output_root: str | Path, dataset: str, training_seed: int) -> Path:
    validate_dataset_seed(dataset, training_seed)
    return Path(output_root) / dataset.lower() / f"seed{training_seed}"


def _root_relative(root: Path, candidate: Path) -> str:
    resolved_root, resolved_candidate = root.resolve(), candidate.resolve()
    if resolved_root not in resolved_candidate.parents:
        raise RuntimeError(f"SEALED_ARTIFACT_PATH_ESCAPE:{candidate}")
    return str(resolved_candidate.relative_to(resolved_root))


def hash_relative_files(root: Path, relatives: list[str] | tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in relatives:
        path = root / relative
        if _root_relative(root, path) != relative or not path.is_file():
            raise FileNotFoundError(f"SEALED_ARTIFACT_MISSING:{relative}")
        result[relative] = sha256_file(path)
    return result


def validation_source_hashes() -> dict[str, str]:
    """Hash both new validation sources and every frozen source it dispatches to."""
    relative_paths = (
        "experiments/utility_relation/p3_c3_a0_ucab/protocol.py",
        "experiments/utility_relation/p3_c3_a0_ucab/budget.py",
        "experiments/utility_relation/p3_c3_a0_ucab/semantic_epoch.py",
        "experiments/utility_relation/p3_c3_a0_ucab/audit.py",
        "experiments/utility_relation/p3_c3_a0_ucab/run_confirmatory.py",
        "experiments/utility_relation/p3_c3_a0_ucab/evaluate_confirmatory.py",
        "experiments/utility_relation/p3_c3_a0_ucab/validation_io.py",
        "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py",
        "experiments/utility_relation/bone_p3_c0_cross_dataset_formal_protocol.py",
        "experiments/utility_relation/bone_p3_b0_generic_carrier.py",
        "method/bone.py",
        "method/amv.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relative_paths if (ROOT / relative).is_file()}


def required_artifacts(training_seed: int) -> list[str]:
    artifacts = [
        "RUN_CONFIG.json",
        "carrier/carrier_model_state.pt",
        "carrier/carrier_pre_gt.npz",
        "carrier/carrier_audit.json",
    ]
    for arm in arms_for_seed(training_seed):
        artifacts.extend((f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"))
    return artifacts + ["BRANCHES_COMPLETE.json", "seal/pre_gt_audit.json"]


def verify_pre_gt_seal(root: str | Path) -> dict:
    """Verify sealed pre-GT output without reading data or full labels."""
    root = Path(root)
    seal_path, complete_path = root / "seal/pre_gt_seal.json", root / "PRE_GT_COMPLETE.json"
    if not seal_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError("PRE_GT_SEAL_OR_COMPLETE_MISSING")
    seal, complete = read_json(seal_path, "PRE_GT_SEAL"), read_json(complete_path, "PRE_GT_COMPLETE")
    if seal.get("status") != "PRE_GT_SEALED" or complete.get("status") != "COMPLETE":
        raise RuntimeError("PRE_GT_COMPLETE_STATUS_INVALID")
    dataset, training_seed = seal.get("dataset"), seal.get("training_seed")
    if not isinstance(dataset, str) or isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise RuntimeError("PRE_GT_SEAL_IDENTITY_INVALID")
    validate_dataset_seed(dataset, training_seed)
    if complete.get("dataset") != dataset or complete.get("training_seed") != training_seed:
        raise RuntimeError("PRE_GT_COMPLETE_IDENTITY_INVALID")
    if complete.get("pre_gt_seal_sha256") != sha256_file(seal_path):
        raise RuntimeError("PRE_GT_COMPLETE_SEAL_HASH_INVALID")
    expected = seal.get("artifact_sha256")
    if not isinstance(expected, dict) or expected != hash_relative_files(root, required_artifacts(training_seed)):
        raise RuntimeError("PRE_GT_ARTIFACT_SEAL_INVALID")
    if seal.get("source_sha256") != validation_source_hashes():
        raise RuntimeError("PRE_GT_SOURCE_PROVENANCE_INVALID")
    required_false = ("GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_selection", "formal_metrics_computed")
    if any(seal.get(name) is not False for name in required_false):
        raise RuntimeError("PRE_GT_FIREWALL_SEAL_INVALID")
    audit = read_json(root / "seal/pre_gt_audit.json", "PRE_GT_AUDIT")
    if any(audit.get(name) is not False for name in required_false):
        raise RuntimeError("PRE_GT_FIREWALL_AUDIT_INVALID")
    if tuple(audit.get("arms", ())) != arms_for_seed(training_seed):
        raise RuntimeError("PRE_GT_ARM_SET_INVALID")
    return {"seal": seal, "audit": audit, "complete": complete}


def verify_all_pre_gt_runs_sealed(output_root: str | Path) -> bool:
    for dataset in ucab_protocol.DATASETS:
        for seed in (*ucab_protocol.HISTORICAL_PAIRED_SEEDS, *ucab_protocol.CONFIRMATORY_SEEDS):
            verify_pre_gt_seal(run_root(output_root, dataset, seed))
    return True
