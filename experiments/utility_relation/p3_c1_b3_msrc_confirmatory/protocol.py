"""Fail-closed constants, parent integrity, and freeze helpers for P3-C1-B3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.utility_relation.p3_c1_b1_final_formal import (
    final_formal_protocol as final_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = "MSRC"
CONFIRMATORY_SEEDS = (60, 70, 90)
FORBIDDEN_SELECTION_SEEDS = (20, 30, 50)
ARMS = ("BASE", "OURS_B1", "OURS_B2")
ARM_PARAMETERS = {
    "BASE": {"semantic_lr": None, "semantic_exposure": 0},
    "OURS_B1": {"semantic_lr": 1e-3, "semantic_exposure": 1},
    "OURS_B2": {"semantic_lr": 2e-3, "semantic_exposure": 1},
}
CARRIER_EPOCHS = 100
REFINEMENT_EPOCHS = 20
QUERY_CHUNK_SIZE = 256
OUTPUT_ROOT = REPO_ROOT / "outputs/bone_p3/p3_c1_b3_msrc_b2_confirmatory"
PREREGISTERED_FREEZE = (
    REPO_ROOT
    / "experiment_freeze/bone_p3_c1_b3_msrc_b2_confirmatory_preregistered_20261004"
)
RESULT_FREEZE = (
    REPO_ROOT
    / "experiment_freeze/bone_p3_c1_b3_msrc_b2_confirmatory_results_20261004"
)
B2_PARENT_FREEZE = (
    REPO_ROOT
    / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004"
)

PARENT_FINAL_FORMAL_SHA256 = {
    "experiments/utility_relation/p3_c1_b1_final_formal/__init__.py": "e3ccf091c93db36d65951581164f97498ff201ac85050c34a7a24cd1f97a00d4",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_protocol.py": "4e3d443bc9e5f21ddf92cb763832a6b3ed5bf5520a07b7113067288fd8847ffb",
    "experiments/utility_relation/p3_c1_b1_final_formal/final_formal_context.py": "77840c28319df9e4d07a548523903a859dfaf09a544e9a74cc17fc7411cf8e56",
    "experiments/utility_relation/p3_c1_b1_final_formal/run_final_formal.py": "b6abc513f81ae08d00141ea897d2c8c209840677503715fd79ab3f237284a6bd",
    "experiments/utility_relation/p3_c1_b1_final_formal/evaluate_final_formal.py": "3cfc841b74589a96f7c237e7751cc200a53a3087590adb194e95979a3ad3cc71",
}
PARENT_B2_FREEZE_SHA256 = {
    "STATUS.txt": "a16db7bf30182097eb97169e7a63df972ad9d5b60dda0009a2867487fdb42a0b",
    "B2_FINAL_SELECTION.json": "ed33b56d870065cc76688037dd893f7e66ca00796ef029ae7fbd9fe845263bca",
    "source_sha256.txt": "0628244515a36a927851cafb5fc6c3893b00cb942d51c63cc7d154d85002040f",
    "parent_source_sha256.txt": "06d78b81c3a3787d16527fb206fe450c0f354f86b8e073cfd46d6805a2b54dc5",
    "parent_freeze_sha256.txt": "729fd46300ca90df96fa166d343f8a72438706ff9fa2e5445999e6a5ea4a6281",
    "artifact_sha256.txt": "8433153bea0fd43778f4d5a35e45138d86861c9c9e24a10ea487060768648952",
    "PARENT_INTEGRITY.txt": "49deb7318e6afffa9abf1b2ceca02a15028d25e3d6702e4605265438a3d81790",
}
SOURCE_RELATIVE_PATHS = (
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/__init__.py",
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/protocol.py",
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/context.py",
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/run_confirmatory.py",
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/evaluate_confirmatory.py",
    "experiments/utility_relation/p3_c1_b3_msrc_confirmatory/summarize_confirmatory.py",
    "tests/test_p3_c1_b3_msrc_confirmatory.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"B3_SOURCE_MISSING:{relative}")
        result[relative] = sha256_file(path)
    return result


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        rows[name.strip()] = digest
    return rows


def verify_parent_integrity() -> dict:
    """Verify B2 selection, its chain, and the reused final-formal implementation."""
    observed_final = {}
    for relative, expected in PARENT_FINAL_FORMAL_SHA256.items():
        path = REPO_ROOT / relative
        observed_final[relative] = sha256_file(path)
        if observed_final[relative] != expected:
            raise RuntimeError(f"B3_FINAL_FORMAL_PARENT_HASH_MISMATCH:{relative}")
    observed_b2 = {}
    for name, expected in PARENT_B2_FREEZE_SHA256.items():
        path = B2_PARENT_FREEZE / name
        observed_b2[name] = sha256_file(path)
        if observed_b2[name] != expected:
            raise RuntimeError(f"B3_B2_PARENT_FREEZE_HASH_MISMATCH:{name}")
    for manifest_name in ("source_sha256.txt", "parent_source_sha256.txt"):
        for relative, expected in _parse_sha256_manifest(
            B2_PARENT_FREEZE / manifest_name
        ).items():
            path = REPO_ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"B3_B2_PARENT_CHAIN_MISMATCH:{relative}")
    status = (B2_PARENT_FREEZE / "STATUS.txt").read_text(encoding="utf-8")
    required = (
        "STATUS=P3_C1_B2_TARGETED_LOCAL_REFINEMENT_RESULTS_FROZEN",
        "FULL_GT_USED=false",
        "FORMAL_METRICS_READ=false",
        "FINAL_B2_PARAMETERS_FROZEN=true",
        "CONFIRMATORY_SEEDS_RUN=false",
    )
    if any(item not in status for item in required):
        raise RuntimeError("B3_B2_PARENT_STATUS_INVALID")
    selection = json.loads(
        (B2_PARENT_FREEZE / "B2_FINAL_SELECTION.json").read_text(encoding="utf-8")
    )
    rows = [row for row in selection.get("selections", []) if row.get("dataset") == DATASET]
    if len(rows) != 1:
        raise RuntimeError("B3_MSRC_B2_SELECTION_MISSING")
    row = rows[0]
    expected_values = {
        "b1_selected_semantic_lr": 1e-3,
        "b1_selected_semantic_exposure": 1,
        "b2_selected_semantic_lr": 2e-3,
        "b2_selected_semantic_exposure": 1,
        "winner_positive_count": 4,
        "parameter_changed_from_B1": True,
        "full_gt_used": False,
        "formal_metrics_read": False,
    }
    if any(row.get(key) != value for key, value in expected_values.items()):
        raise RuntimeError("B3_MSRC_B2_FROZEN_SELECTION_MISMATCH")
    if (
        float(row.get("winner_mean_delta_J")) != 0.07743207886209852
        or float(row.get("winner_worst_delta_J")) != 0.048171067321766126
    ):
        raise RuntimeError("B3_MSRC_B2_FROZEN_EVIDENCE_MISMATCH")
    final_protocol.verify_parent_sources()
    return {
        "verified": True,
        "parent_source_modification": False,
        "b2_parent_freeze_sha256": observed_b2,
        "final_formal_parent_sha256": observed_final,
    }


def run_root(output_root: str | Path, training_seed: int) -> Path:
    if training_seed not in CONFIRMATORY_SEEDS:
        raise ValueError("B3_UNREGISTERED_CONFIRMATORY_SEED")
    return Path(output_root) / f"seed{training_seed}"


def preregister(freeze_dir: str | Path = PREREGISTERED_FREEZE) -> Path:
    target = Path(freeze_dir)
    if target.exists():
        raise FileExistsError("B3_PREREGISTRATION_ALREADY_EXISTS")
    parent = verify_parent_integrity()
    target.mkdir(parents=True)
    status = """STATUS=P3_C1_B3_MSRC_B2_CONFIRMATORY_PREREGISTERED
DATASET=MSRC
SEEDS=60,70,90
ARMS=BASE,OURS_B1,OURS_B2
B1_LR=0.001
B1_ES=1
B2_LR=0.002
B2_ES=1
METHOD_CHANGED=false
OPTIMIZER_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
SEMANTIC_LOSS_CHANGED=false
FULL_GT_USED_FOR_TRAINING=false
FULL_GT_USED_FOR_SELECTION=false
B2_PARAMETER_ALREADY_FROZEN=true
PARAMETER_RETUNING_ALLOWED=false
REAL_CONFIRMATORY_RUN=false
"""
    (target / "STATUS.txt").write_text(status, encoding="utf-8")
    write_json(
        target / "PROTOCOL.json",
        {
            "stage": "P3-C1-B3",
            "dataset": DATASET,
            "seeds": list(CONFIRMATORY_SEEDS),
            "forbidden_selection_seeds": list(FORBIDDEN_SELECTION_SEEDS),
            "arms": list(ARMS),
            "arm_parameters": ARM_PARAMETERS,
            "carrier_epochs": CARRIER_EPOCHS,
            "refinement_epochs": REFINEMENT_EPOCHS,
            "full_gt_used": False,
            "real_confirmatory_run": False,
        },
    )
    hashes = source_hashes()
    (target / "source_sha256.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    (target / "parent_freeze_sha256.txt").write_text(
        "".join(
            f"{digest}  {B2_PARENT_FREEZE.relative_to(REPO_ROOT) / name}\n"
            for name, digest in parent["b2_parent_freeze_sha256"].items()
        ),
        encoding="utf-8",
    )
    write_json(target / "PARENT_INTEGRITY.txt", parent)
    return target


def verify_preregistration(freeze_dir: str | Path = PREREGISTERED_FREEZE) -> dict:
    root = Path(freeze_dir)
    required_files = (
        "STATUS.txt",
        "PROTOCOL.json",
        "source_sha256.txt",
        "parent_freeze_sha256.txt",
        "PARENT_INTEGRITY.txt",
    )
    if any(not (root / name).is_file() for name in required_files):
        raise FileNotFoundError("B3_PREREGISTRATION_MISSING")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    required_status = (
        "STATUS=P3_C1_B3_MSRC_B2_CONFIRMATORY_PREREGISTERED",
        "SEEDS=60,70,90",
        "ARMS=BASE,OURS_B1,OURS_B2",
        "B2_PARAMETER_ALREADY_FROZEN=true",
        "PARAMETER_RETUNING_ALLOWED=false",
        "REAL_CONFIRMATORY_RUN=false",
    )
    if any(item not in status for item in required_status):
        raise RuntimeError("B3_PREREGISTRATION_STATUS_INVALID")
    if _parse_sha256_manifest(root / "source_sha256.txt") != source_hashes():
        raise RuntimeError("B3_PREREGISTRATION_SOURCE_HASH_MISMATCH")
    protocol_payload = json.loads((root / "PROTOCOL.json").read_text(encoding="utf-8"))
    if (
        protocol_payload.get("dataset") != DATASET
        or tuple(protocol_payload.get("seeds", ())) != CONFIRMATORY_SEEDS
        or tuple(protocol_payload.get("arms", ())) != ARMS
        or protocol_payload.get("arm_parameters") != ARM_PARAMETERS
        or protocol_payload.get("real_confirmatory_run") is not False
    ):
        raise RuntimeError("B3_PREREGISTRATION_PROTOCOL_INVALID")
    parent = verify_parent_integrity()
    expected_parent_manifest = {
        str(B2_PARENT_FREEZE.relative_to(REPO_ROOT) / name): digest
        for name, digest in parent["b2_parent_freeze_sha256"].items()
    }
    if _parse_sha256_manifest(root / "parent_freeze_sha256.txt") != expected_parent_manifest:
        raise RuntimeError("B3_PREREGISTRATION_PARENT_HASH_MISMATCH")
    return {
        "verified": True,
        "preregistration_status_sha256": sha256_file(root / "STATUS.txt"),
        "preregistration_protocol_sha256": sha256_file(root / "PROTOCOL.json"),
        "source_sha256": source_hashes(),
        "parent": parent,
    }
