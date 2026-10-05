"""Phase-A B2 preregistration writer; it performs only source/provenance reads."""
from __future__ import annotations

import json
from pathlib import Path

from . import candidate_reuse
from .protocol import B1_SELECTED, DATASETS, LOCAL_GRIDS, candidates, validate_protocol

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004"
NEW_FILES = (
    "experiments/utility_relation/p3_c1_b2_local_refinement/__init__.py",
    "experiments/utility_relation/p3_c1_b2_local_refinement/protocol.py",
    "experiments/utility_relation/p3_c1_b2_local_refinement/candidate_reuse.py",
    "experiments/utility_relation/p3_c1_b2_local_refinement/run_local_refinement.py",
    "experiments/utility_relation/p3_c1_b2_local_refinement/audit_local_refinement.py",
    "experiments/utility_relation/p3_c1_b2_local_refinement/finalize_local_refinement.py",
    "tests/test_p3_c1_b2_local_refinement.py",
)


def _hash(path: Path) -> str:
    return candidate_reuse.sha256_file(path)


def preflight() -> dict:
    validate_protocol()
    dependencies = candidate_reuse.verify_frozen_dependencies()
    by_dataset = {dataset: candidate_reuse.preflight(dataset, candidates(dataset)) for dataset in DATASETS}
    return {"developmental_local_refinement": True, "target_datasets": list(DATASETS), "dev_seeds": [11, 37],
            "branches": ["11A", "11B", "37A", "37B"], "grids": LOCAL_GRIDS, "b1_selected": {key: {"semantic_lr": val[0], "semantic_exposure": val[1]} for key, val in B1_SELECTED.items()},
            "selection_rule": ["maximum round(mean_delta_J,12)", "maximum round(worst_delta_J,12)", "maximum positive_count", "smaller semantic_exposure", "smaller semantic_lr"],
            "candidate_reuse_requires": ["exact dataset/config", "carrier state and pre-GT hashes", "corruption and sparse realization through frozen carrier provenance", "A2 context/protocol fingerprint", "A2 objective", "frozen semantic source dependency manifest", "four branch schedule and state hashes"],
            "datasets": by_dataset, "total_candidates": sum(item["candidate_count"] for item in by_dataset.values()),
            "reusable_candidates": sum(item["reusable_count"] for item in by_dataset.values()), "new_candidates": sum(item["new_count"] for item in by_dataset.values()),
            "full_gt_used": False, "formal_metrics_read": False, "boundary_extension_allowed": False, "post_grid_extension_allowed": False,
            "frozen_dependencies": dependencies}


def freeze_phase_a(test_results: str) -> Path:
    if FREEZE.exists():
        raise FileExistsError("B2_PHASE_A_FREEZE_ALREADY_EXISTS")
    result = preflight()
    FREEZE.mkdir(parents=True)
    status = """STATUS=P3_C1_B2_TARGETED_LOCAL_REFINEMENT_PREREGISTERED
STAGE=P3-C1-B2

DEVELOPMENTAL_LOCAL_REFINEMENT=true

TARGET_DATASETS=CCV,MSRC,ALOI
TARGET_DATASET_COUNT=3

DEV_SEEDS=11,37
DEV_BRANCHES=11A,11B,37A,37B

METHOD_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
SEMANTIC_LOSS_CHANGED=false
ACTION_BUDGET_CHANGED=false

HYPERPARAMETERS_SEARCHED=semantic_lr,semantic_exposure

FULL_GT_USED=false
FORMAL_METRICS_READ=false

SELECTION_RULE_CHANGED=false
B1_SELECTION_RULE_REUSED=true

BOUNDARY_EXTENSION_ALLOWED=false
POST_GRID_EXTENSION_ALLOWED=false

REAL_B2_TRIALS_RUN=false

NEXT_STAGE=B2_LOCAL_REFINEMENT_EXECUTION
"""
    (FREEZE / "STATUS.txt").write_text(status)
    (FREEZE / "SEARCH_SPACE.json").write_text(json.dumps({"grids": LOCAL_GRIDS, "counts": {d: len(candidates(d)) for d in DATASETS}, "total": 73}, indent=2, sort_keys=True) + "\n")
    (FREEZE / "SELECTION_RULE.json").write_text(json.dumps({"round_decimals": 12, "rule": result["selection_rule"], "selection_inputs": ["mean_delta_J", "worst_delta_J", "positive_count"]}, indent=2) + "\n")
    (FREEZE / "CANDIDATE_REUSE_PREFLIGHT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    protocol_text = "# P3-C1-B2 local refinement\n\nOnly CCV, MSRC, and ALOI are included. The 25/18/30 local grids total 73 candidates. Candidates use frozen A2 arrays, sparse folds, relation primitive, semantic/native order, 100 carrier epochs, 20 refinement epochs, and the four original dev branches. Selection is the frozen B1 rounded lexicographic pre-GT DeltaJ rule. No boundary extension is permitted.\n"
    (FREEZE / "PROTOCOL.md").write_text(protocol_text)
    new_hashes = {name: _hash(ROOT / name) for name in NEW_FILES}
    (FREEZE / "source_sha256.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in new_hashes.items()) + "\n")
    (FREEZE / "parent_source_sha256.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in result["frozen_dependencies"].items()) + "\n")
    parent_freezes = (ROOT / "experiment_freeze/bone_p3_c1_b1_a2_final_hyperparameters_20261004/FREEZE_SHA256.txt", ROOT / "experiment_freeze/bone_p3_c2_b0_utility_mass_action_budget_audit_20261004/STATUS.txt")
    (FREEZE / "parent_freeze_sha256.txt").write_text("\n".join(f"{_hash(path)}  {path.relative_to(ROOT)}" for path in parent_freezes) + "\n")
    (FREEZE / "PARENT_INTEGRITY.txt").write_text(json.dumps({"verified": True, "method_source_modified": False, "frozen_dependency_manifest": str(candidate_reuse.B1_DEPENDENCY_MANIFEST.relative_to(ROOT))}, indent=2) + "\n")
    (FREEZE / "test_results.txt").write_text(test_results.rstrip() + "\n")
    return FREEZE
