"""Read-only provenance and frozen-parent validation for B4."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import protocol as b3_protocol

from . import protocol


def _manifest_rows(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows[name.strip()] = digest
    return rows


def _verify_manifest(path: Path) -> None:
    for relative, expected in _manifest_rows(path).items():
        target = protocol.REPO_ROOT / relative
        if not target.is_file() or protocol.sha256_file(target) != expected:
            raise RuntimeError(f"B4_PARENT_MANIFEST_MISMATCH:{relative}")


def verify_parent_integrity() -> dict:
    observed = {}
    for name, (path, expected) in protocol.PARENT_FREEZE_SHA256.items():
        observed[name] = protocol.sha256_file(path)
        if observed[name] != expected:
            raise RuntimeError(f"B4_PARENT_FREEZE_HASH_MISMATCH:{name}")
    for manifest in (
        protocol.P2_PREFORMAL / "FREEZE_MANIFEST.sha256",
        protocol.P2_PASS / "FREEZE_MANIFEST.sha256",
        protocol.P3_A3 / "FREEZE_MANIFEST.sha256",
        protocol.P3_A4 / "FREEZE_MANIFEST.sha256",
        protocol.P3_B0 / "FREEZE_MANIFEST.sha256",
    ):
        _verify_manifest(manifest)
    for relative, expected in protocol.FROZEN_METHOD_SHA256.items():
        if protocol.sha256_file(protocol.REPO_ROOT / relative) != expected:
            raise RuntimeError(f"B4_FROZEN_METHOD_HASH_MISMATCH:{relative}")
    preformal = (protocol.P2_PREFORMAL / "PREREGISTERED_PROTOCOL.txt").read_text(encoding="utf-8")
    required_historical = (
        "PRE-FORMAL FROZEN", "DATASET:\nMSRC", "20 refinement epochs",
        "Semantic Adam lr = 1e-4", "one semantic optimizer.step per epoch",
        "NO:\nhyperparameter tuning",
    )
    if any(value not in preformal for value in required_historical):
        raise RuntimeError("B4_HISTORICAL_DEFAULT_PROVENANCE_INVALID")
    pass_status = (protocol.P2_PASS / "STATUS.txt").read_text(encoding="utf-8")
    if "P2_A1B_MULTI_SEED_PASS" not in pass_status:
        raise RuntimeError("B4_P2_A1B_PASS_PARENT_INVALID")
    b3_protocol.verify_parent_integrity()
    b3_status = (protocol.P3_B3 / "STATUS.txt").read_text(encoding="utf-8")
    if "RUN_COUNT=9" not in b3_status or "ALL_RUNS_COMPLETE=true" not in b3_status:
        raise RuntimeError("B4_B3_PARENT_STATUS_INVALID")
    for parent in (protocol.P3_B2, protocol.P3_B3):
        for relative, expected in _manifest_rows(parent / "source_sha256.txt").items():
            target = protocol.REPO_ROOT / relative
            if not target.is_file() or protocol.sha256_file(target) != expected:
                raise RuntimeError(f"B4_P3_PARENT_SOURCE_MISMATCH:{relative}")
    split_current = protocol.REPO_ROOT / "outputs/bone_p3/sparse_splits/MSRC_2perclass.npz"
    split_frozen = protocol.P3_A3 / "split_artifacts/MSRC_2perclass.npz"
    mask_current = protocol.REPO_ROOT / "outputs/bone_p3/corruptions/MSRC_mask.npz"
    mask_frozen = protocol.P3_A4 / "corruption_artifacts/MSRC_mask.npz"
    input_hashes = {
        "sparse_split_sha256": protocol.sha256_file(split_current),
        "corruption_artifact_sha256": protocol.sha256_file(mask_current),
    }
    if input_hashes["sparse_split_sha256"] != protocol.sha256_file(split_frozen):
        raise RuntimeError("B4_SPARSE_SPLIT_FREEZE_MISMATCH")
    if input_hashes["corruption_artifact_sha256"] != protocol.sha256_file(mask_frozen):
        raise RuntimeError("B4_CORRUPTION_FREEZE_MISMATCH")
    return {
        "verified": True,
        "historical_default_source": "P2_A1B_FROZEN_PRE_B1_B2_B3",
        "historical_parent_date": "20261001",
        "b1_b2_b3_date": "20261004",
        "historical_precedes_b1_b2_b3": True,
        "parent_source_modification": False,
        "freeze_sha256": observed,
        **input_hashes,
    }


def preregister(freeze_dir: str | Path = protocol.PREREGISTERED_FREEZE) -> Path:
    target = Path(freeze_dir)
    if target.exists():
        raise FileExistsError("B4_PREREGISTRATION_ALREADY_EXISTS")
    parent = verify_parent_integrity()
    target.mkdir(parents=True)
    status = """STATUS=P3_C1_B4_MSRC_HISTORICAL_DEFAULT_CONFIRMATORY_PREREGISTERED
DATASET=MSRC
SEEDS=100,110,130
ARMS=BASE,OURS_HISTORICAL_DEFAULT
HISTORICAL_DEFAULT_SOURCE=P2_A1B_FROZEN_PRE_B1_B2_B3
HISTORICAL_DEFAULT_SEMANTIC_LR=0.0001
HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE=20
REFINEMENT_EPOCHS=20
SEMANTIC_OPTIMIZER=Adam
SEMANTIC_STEPS_PER_ACTIVE_EPOCH=1
PARAMETER_SEARCH=false
PARAMETER_RETUNING_ALLOWED=false
OPTIMIZER_TUNING_ALLOWED=false
METHOD_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
RELATION_PRIMITIVE_CHANGED=false
SEMANTIC_LOSS_CHANGED=false
FULL_GT_USED_FOR_TRAINING=false
FULL_GT_USED_FOR_SELECTION=false
REAL_CONFIRMATORY_RUN=false
"""
    (target / "STATUS.txt").write_text(status, encoding="utf-8")
    protocol_text = """P3-C1-B4 MSRC Historical-Default New-Seed Confirmatory

Historical source: P2-A1B pre-formal frozen protocol dated 20261001, preceding B1/B2/B3.
Dataset: MSRC only. Seeds: 100,110,130. Arms: BASE, OURS_HISTORICAL_DEFAULT.
Historical-default: semantic Adam lr=1e-4; 20 active semantic epochs; exactly one semantic optimizer.step per epoch.
Current P3 carrier: 100 native epochs. Refinement: 20 epochs. BASE is native-only; historical arm is semantic then native each epoch.
No parameter search, optimizer search, retuning, B1/B2 result selection, GT training access, or post-hoc gate change.
Primary gate: CONFIRMED iff mean Delta_ACC > 0 and mean Delta_NMI > 0 and mean Delta_ARI > 0; otherwise NOT_CONFIRMED.
REAL_CONFIRMATORY_RUN=false
"""
    (target / "PROTOCOL.txt").write_text(protocol_text, encoding="utf-8")
    sources = protocol.source_hashes()
    (target / "source_sha256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in sources.items()), encoding="utf-8")
    (target / "parent_freeze_sha256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in parent["freeze_sha256"].items()), encoding="utf-8")
    protocol.write_json(target / "PARENT_INTEGRITY.txt", parent)
    return target


def verify_preregistration(freeze_dir: str | Path = protocol.PREREGISTERED_FREEZE) -> dict:
    root = Path(freeze_dir)
    if any(not (root / name).is_file() for name in ("STATUS.txt", "PROTOCOL.txt", "source_sha256.txt", "parent_freeze_sha256.txt", "PARENT_INTEGRITY.txt")):
        raise FileNotFoundError("B4_PREREGISTRATION_MISSING")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    required = ("STATUS=P3_C1_B4_MSRC_HISTORICAL_DEFAULT_CONFIRMATORY_PREREGISTERED", "SEEDS=100,110,130", "HISTORICAL_DEFAULT_SEMANTIC_LR=0.0001", "HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE=20", "PARAMETER_SEARCH=false", "REAL_CONFIRMATORY_RUN=false")
    if any(value not in status for value in required):
        raise RuntimeError("B4_PREREGISTRATION_STATUS_INVALID")
    sources = protocol.source_hashes()
    if _manifest_rows(root / "source_sha256.txt") != sources:
        raise RuntimeError("B4_PREREGISTRATION_SOURCE_HASH_MISMATCH")
    parent = verify_parent_integrity()
    if _manifest_rows(root / "parent_freeze_sha256.txt") != parent["freeze_sha256"]:
        raise RuntimeError("B4_PREREGISTRATION_PARENT_HASH_MISMATCH")
    return {"verified": True, "status_sha256": protocol.sha256_file(root / "STATUS.txt"), "source_sha256": sources, "parent": parent}
