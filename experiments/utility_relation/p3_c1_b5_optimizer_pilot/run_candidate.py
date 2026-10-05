"""Explicit future B5 candidate entrypoint; never called by preregistration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import protocol
from .audit_optimizer_pilot import verify_parent_integrity
from .candidate_reuse import preflight

AUTHORIZATION_FREEZE = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004"
TRAINING_WIRING_FREEZE = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a2_real_candidate_training_wiring_20261004"
API_WIRING_REPAIR_FREEZE = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a3_frozen_evaluator_api_wiring_repair_20261004"
OLD_RUN_SOURCE_SHA256 = "5d0814b8295729759f636b4e3894a80946bcd14fd541afaadec9238f0cfe808c"
A1_RUN_SOURCE_SHA256 = "4aeaa73aaad762696d86b49b15d2e39f903f0f7bfc8b84edfe2f49b5aaa55a97"
A1_FINALIZER_SOURCE_SHA256 = "99d1d33a8385f0b388cc037516037df2016329c4bd91dde42cea40074a5c45e2"
A2_DRIFT_BASELINES = {
    "experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py": A1_RUN_SOURCE_SHA256,
    "experiments/utility_relation/p3_c1_b5_optimizer_pilot/finalize_optimizer_pilot.py": A1_FINALIZER_SOURCE_SHA256,
}
A3_DRIFT_BASELINES = {
    "experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py": "dbe93d394ff4a3e331663c016796b6d0e7a2b1f0d2d9699331394a3c2a01199c",
    "experiments/utility_relation/p3_c1_b5_optimizer_pilot/training_wiring.py": "4d40c7e1ec48c1ecc5287daa7da64611dd72d0632d1b6b8ba1eb0da8ef26d9d1",
    "tests/test_p3_c1_b5_a2_training_wiring.py": "85dc1551a6aa0cc2276a56d14b9de82dd32a2c72cf878c12f83294acd27f83c1",
}


def _manifest(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows[name.strip()] = digest
    return rows


def _verify_parent_preregistration_for_amendment() -> None:
    """Validate the immutable preregistration while allowing this one runner amendment."""
    verify_parent_integrity()
    root = protocol.PREREGISTERED_FREEZE
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    if "STATUS=P3_C1_B5_OPTIMIZER_PILOT_PREREGISTERED" not in status or "TOTAL_CANDIDATE_COUNT=18" not in status:
        raise RuntimeError("B5_PARENT_PREREGISTRATION_INVALID")
    expected = _manifest(root / "source_sha256.txt")
    for relative, digest in expected.items():
        observed = protocol.sha256_file(protocol.REPO_ROOT / relative)
        if relative in A2_DRIFT_BASELINES:
            if relative.endswith("/run_candidate.py") and digest != OLD_RUN_SOURCE_SHA256:
                raise RuntimeError("B5_OLD_RUNNER_SOURCE_BASELINE_INVALID")
            if relative.endswith("/finalize_optimizer_pilot.py") and digest != A1_FINALIZER_SOURCE_SHA256:
                raise RuntimeError("B5_OLD_RUNNER_SOURCE_BASELINE_INVALID")
        elif observed != digest:
            raise RuntimeError(f"B5_PARENT_PREREGISTRATION_SOURCE_DRIFT:{relative}")
    if protocol.sha256_file(root / "STATUS.txt") != "efb153610c24e3d82d4ba1f9f5824b1698964946c2d887ec2df804b4d851955d":
        raise RuntimeError("B5_PARENT_PREREGISTRATION_STATUS_DRIFT")


def _verify_authorization_amendment() -> dict:
    _verify_parent_preregistration_for_amendment()
    root = AUTHORIZATION_FREEZE
    required = ("STATUS.txt", "AUTHORIZATION.txt", "PARENT_INTEGRITY.txt", "source_before_after_sha256.txt", "source_sha256.txt", "parent_freeze_sha256.txt")
    if any(not (root / name).is_file() for name in required):
        raise RuntimeError("B5_A1_AUTHORIZATION_FREEZE_MISSING")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    required_status = ("STATUS=P3_C1_B5_A1_REAL_CANDIDATE_EXECUTION_AUTHORIZED", "PARENT_B5_PREREGISTRATION_PASS=true", "AUTHORIZED_NEW_CANDIDATE_COUNT=12", "ADAM_REUSE_COUNT=6", "REAL_CANDIDATE_EXECUTION_AUTHORIZED=true", "REAL_CANDIDATES_ALREADY_RUN=false", "FAILED_ATTEMPT_PRODUCED_ARTIFACT=false")
    if any(value not in status for value in required_status):
        raise RuntimeError("B5_A1_AUTHORIZATION_STATUS_INVALID")
    current = _manifest(root / "source_sha256.txt")
    parent_manifest = _manifest(AUTHORIZATION_FREEZE / "source_sha256.txt")
    expected_current = protocol.source_hashes()
    if set(current) != set(expected_current):
        raise RuntimeError("B5_A1_SOURCE_HASH_MISMATCH")
    for relative, digest in expected_current.items():
        if relative in A2_DRIFT_BASELINES:
            if parent_manifest.get(relative) != A2_DRIFT_BASELINES[relative]:
                raise RuntimeError("B5_A1_DRIFT_BASELINE_INVALID")
        elif current.get(relative) != digest or parent_manifest.get(relative) != digest:
            raise RuntimeError("B5_A1_SOURCE_HASH_MISMATCH")
    before_after = _manifest(root / "source_before_after_sha256.txt")
    if before_after.get("experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py.before") != OLD_RUN_SOURCE_SHA256:
        raise RuntimeError("B5_A1_BEFORE_HASH_INVALID")
    if before_after.get("experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py.after") != parent_manifest["experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py"]:
        raise RuntimeError("B5_A1_AFTER_HASH_INVALID")
    integrity = json.loads((root / "PARENT_INTEGRITY.txt").read_text(encoding="utf-8"))
    if integrity.get("parent_b5_preregistration_pass") is not True or integrity.get("scientific_protocol_changed") is not False:
        raise RuntimeError("B5_A1_PARENT_INTEGRITY_INVALID")
    parent_hashes = _manifest(root / "parent_freeze_sha256.txt")
    parent_root = protocol.PREREGISTERED_FREEZE
    if parent_hashes.get("experiment_freeze/bone_p3_c1_b5_optimizer_pilot_preregistered_20261004/STATUS.txt") != protocol.sha256_file(parent_root / "STATUS.txt"):
        raise RuntimeError("B5_A1_PARENT_STATUS_HASH_INVALID")
    if parent_hashes.get("experiment_freeze/bone_p3_c1_b5_optimizer_pilot_preregistered_20261004/source_sha256.txt") != protocol.sha256_file(parent_root / "source_sha256.txt"):
        raise RuntimeError("B5_A1_PARENT_SOURCE_MANIFEST_HASH_INVALID")
    return {"status": "PASS", "authorization_freeze": str(root), "parent_b5_preregistration_pass": True}


def _verify_training_wiring_amendment() -> dict:
    root = TRAINING_WIRING_FREEZE
    required = (
        "STATUS.txt", "IMPLEMENTATION_AUDIT.txt", "PARENT_INTEGRITY.txt",
        "source_sha256.txt", "source_before_after_sha256.txt", "parent_freeze_sha256.txt",
    )
    if any(not (root / name).is_file() for name in required):
        raise RuntimeError("B5_A2_TRAINING_WIRING_FREEZE_MISSING")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    required_status = (
        "STATUS=P3_C1_B5_A2_REAL_CANDIDATE_TRAINING_WIRING_READY",
        "SCIENTIFIC_PROTOCOL_CHANGED=false", "SEARCH_SPACE_CHANGED=false",
        "SELECTION_RULE_CHANGED=false", "J_DEFINITION_CHANGED=false",
        "REAL_TRAINING_PATH_CONNECTED=true", "REAL_CANDIDATES_ALREADY_RUN=false",
        "PREFLIGHT_ONLY_ATTEMPT_COUNT=12", "PREFLIGHT_ONLY_ARTIFACT_COUNT=0",
        "AUTHORIZED_REAL_NEW_CANDIDATE_COUNT=12", "ADAM_REUSE_COUNT=6",
        "FULL_GT_USED=false", "FORMAL_METRICS_READ=false",
        "READY_FOR_REAL_B5_CANDIDATE_EXECUTION=true",
    )
    if any(value not in status for value in required_status):
        raise RuntimeError("B5_A2_TRAINING_WIRING_STATUS_INVALID")
    manifest = _manifest(root / "source_sha256.txt")
    for relative, digest in manifest.items():
        if relative in A3_DRIFT_BASELINES:
            if digest != A3_DRIFT_BASELINES[relative]:
                raise RuntimeError(f"B5_A2_SOURCE_BASELINE_INVALID:{relative}")
        elif protocol.sha256_file(protocol.REPO_ROOT / relative) != digest:
            raise RuntimeError(f"B5_A2_SOURCE_HASH_MISMATCH:{relative}")
    parent = _manifest(root / "parent_freeze_sha256.txt")
    if parent.get("experiment_freeze/bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004/STATUS.txt") != protocol.sha256_file(AUTHORIZATION_FREEZE / "STATUS.txt"):
        raise RuntimeError("B5_A2_PARENT_STATUS_HASH_INVALID")
    if parent.get("experiment_freeze/bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004/source_sha256.txt") != protocol.sha256_file(AUTHORIZATION_FREEZE / "source_sha256.txt"):
        raise RuntimeError("B5_A2_PARENT_SOURCE_HASH_INVALID")
    frozen_parent_hashes = {
        "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/STATUS.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/STATUS.txt",
        "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/SOURCE_SHA256.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/SOURCE_SHA256.txt",
        "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004/STATUS.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004/STATUS.txt",
        "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004/source_sha256.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004/source_sha256.txt",
        "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004/STATUS.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004/STATUS.txt",
        "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004/source_sha256.txt": protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004/source_sha256.txt",
    }
    if any(parent.get(relative) != protocol.sha256_file(path) for relative, path in frozen_parent_hashes.items()):
        raise RuntimeError("B5_A2_B1_B2_PARENT_HASH_INVALID")
    integrity = json.loads((root / "PARENT_INTEGRITY.txt").read_text(encoding="utf-8"))
    if integrity.get("parent_a1_authorization_pass") is not True or integrity.get("scientific_protocol_changed") is not False:
        raise RuntimeError("B5_A2_PARENT_INTEGRITY_INVALID")
    return {"status": "PASS", "training_wiring_freeze": str(root)}


def _verify_evaluator_api_wiring_repair() -> dict:
    """Validate A3 and its immutable A2 parent before any real candidate."""
    _verify_training_wiring_amendment()
    root = API_WIRING_REPAIR_FREEZE
    required = (
        "STATUS.txt", "API_AUDIT.txt", "MODEL_LOAD_SMOKE.json",
        "PARENT_INTEGRITY.txt", "source_before_after_sha256.txt",
        "source_sha256.txt", "parent_freeze_sha256.txt",
    )
    if any(not (root / name).is_file() for name in required):
        raise RuntimeError("B5_A3_API_WIRING_REPAIR_FREEZE_MISSING")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    required_status = (
        "STATUS=P3_C1_B5_A3_FROZEN_EVALUATOR_API_WIRING_REPAIRED",
        "ROOT_CAUSE=INVALID_PRIVATE_HELPER_REFERENCE_ON_R2_WRAPPER",
        "SCIENTIFIC_PROTOCOL_CHANGED=false", "SEARCH_SPACE_CHANGED=false",
        "SELECTION_RULE_CHANGED=false", "J_DEFINITION_CHANGED=false",
        "FROZEN_B1_B2_SOURCE_MODIFIED=false", "MODEL_LOADING_PATH_PARITY=true",
        "MODEL_LOAD_SMOKE=PASS", "FAILED_REAL_INVOCATION_COUNT=1",
        "FAILED_REAL_INVOCATION_ARTIFACT_COUNT=0", "VALID_REAL_CANDIDATE_COUNT=0",
        "READY_FOR_REAL_B5_CANDIDATE_EXECUTION=true", "FULL_GT_USED=false",
        "FORMAL_METRICS_READ=false",
    )
    if any(value not in status for value in required_status):
        raise RuntimeError("B5_A3_API_WIRING_REPAIR_STATUS_INVALID")
    a4_local_amendment_paths = {
        "experiments/utility_relation/p3_c1_b5_optimizer_pilot/run_candidate.py",
        "experiments/utility_relation/p3_c1_b5_optimizer_pilot/training_wiring.py",
        "tests/test_p3_c1_b5_a3_frozen_evaluator_api_wiring.py",
    }
    for relative, digest in _manifest(root / "source_sha256.txt").items():
        # The A3 manifest itself remains immutable.  A4 may amend only these
        # B5-local compatibility surfaces; frozen B1/B2 and every other A3
        # source must remain byte-identical to the A3 manifest.
        if relative not in a4_local_amendment_paths and protocol.sha256_file(protocol.REPO_ROOT / relative) != digest:
            raise RuntimeError(f"B5_A3_SOURCE_HASH_MISMATCH:{relative}")
    parent = _manifest(root / "parent_freeze_sha256.txt")
    for name in ("STATUS.txt", "source_sha256.txt"):
        relative = (
            "experiment_freeze/"
            "bone_p3_c1_b5_a2_real_candidate_training_wiring_20261004/"
            f"{name}"
        )
        if parent.get(relative) != protocol.sha256_file(TRAINING_WIRING_FREEZE / name):
            raise RuntimeError("B5_A3_A2_PARENT_HASH_INVALID")
    smoke = json.loads((root / "MODEL_LOAD_SMOKE.json").read_text(encoding="utf-8"))
    if (
        smoke.get("MODEL_LOAD_SMOKE") != "PASS"
        or smoke.get("model_state_exact_parity") is not True
        or smoke.get("carrier_model_state_sha256")
        != smoke.get("loaded_initial_model_state_sha256")
        or smoke.get("optimizer_step_executed") is not False
        or smoke.get("full_gt_used") is not False
    ):
        raise RuntimeError("B5_A3_MODEL_LOAD_SMOKE_INVALID")
    return {"status": "PASS", "api_wiring_repair_freeze": str(root)}


def authorization_preflight(candidate: protocol.Candidate) -> dict:
    amendment = _verify_authorization_amendment()
    allowed = {item.key() for item in protocol.candidates() if item.optimizer_name != "Adam"}
    if candidate.key() not in allowed:
        if candidate.optimizer_name == "Adam":
            raise RuntimeError("B5_ADAM_CANDIDATE_REUSE_ONLY")
        raise ValueError("B5_CANDIDATE_NOT_IN_AUTHORIZED_NEW_SET")
    return {"authorization_preflight": "PASS", "candidate": candidate.__dict__, **amendment}

def candidate_plan(candidate: protocol.Candidate) -> dict:
    if candidate not in protocol.candidates(candidate.dataset): raise ValueError("B5_CANDIDATE_OUTSIDE_PREREGISTERED_GRID")
    reuse=next(item for item in preflight()["entries"] if item["candidate"]==candidate.__dict__)
    return {"candidate":candidate.__dict__,"reuse":reuse,"full_gt_used":False,"formal_metrics_read":False,"semantic_tensor_detach":{"U_cycle":True,"T_rel":True,"anchor_posterior":True},"semantic_to_native_grad_leak":False,"native_to_next_semantic_grad_leak":False,"real_training_started":False}

def run_candidate(candidate: protocol.Candidate, output_root: str|Path=protocol.OUTPUT_ROOT, *, execute: bool=False, authorize_real_candidate: bool=False) -> dict:
    """Require both switches, then execute exactly one authorized pre-GT candidate."""
    if not authorize_real_candidate:
        raise RuntimeError("B5_REAL_CANDIDATE_EXECUTION_REQUIRES_SEPARATE_AUTHORIZATION")
    if not execute:
        raise RuntimeError("B5_REAL_CANDIDATE_EXECUTION_REQUIRES_EXECUTE")
    plan = candidate_plan(candidate)
    authorization = authorization_preflight(candidate)
    wiring = _verify_evaluator_api_wiring_repair()
    from . import training_wiring
    result = training_wiring.run_real_candidate(candidate, output_root=output_root)
    plan.update(
        {**authorization, **wiring, **result},
        real_training_started=True,
        real_training_completed=True,
        branch_count=4,
        full_gt_used=False,
        formal_metrics_read=False,
    )
    return plan

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    parser.add_argument("--optimizer", choices=protocol.OPTIMIZERS, required=True)
    parser.add_argument("--semantic-lr", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-real-candidate", action="store_true", help="explicit B5-A1 authorization; required with --execute")
    return parser


def main() -> None:
    parser = build_parser()
    args=parser.parse_args(); candidate=next((c for c in protocol.candidates(args.dataset) if c.optimizer_name==args.optimizer and c.semantic_lr==args.semantic_lr),None)
    if candidate is None: raise SystemExit("B5_CANDIDATE_NOT_IN_FROZEN_GRID")
    print(run_candidate(candidate, execute=args.execute, authorize_real_candidate=args.authorize_real_candidate))
if __name__=="__main__": main()
