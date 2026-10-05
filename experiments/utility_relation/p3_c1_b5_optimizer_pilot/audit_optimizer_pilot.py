"""Read-only B5 parent/preregistration integrity; no candidate training or GT."""
from __future__ import annotations
import json
from pathlib import Path
from experiments.utility_relation.p3_c1_b1_final_formal import final_formal_protocol
from . import protocol

def _manifest(path: Path):
    rows={}
    for line in path.read_text().splitlines():
        if line.strip():
            digest,name=line.split(maxsplit=1); rows[name.strip()]=digest
    return rows

def verify_parent_integrity() -> dict:
    status=(protocol.B4_RESULT_FREEZE/"STATUS.txt").read_text()
    required=("STATUS=P3_C1_B4_MSRC_HISTORICAL_DEFAULT_CONFIRMATORY_RESULTS_FROZEN","HISTORICAL_DEFAULT_CONFIRMATION=NOT_CONFIRMED","B4_CLOSED=true")
    if any(v not in status for v in required): raise RuntimeError("B5_B4_RESULT_PARENT_INVALID")
    for root in (protocol.B4_RESULT_FREEZE, protocol.B2_RESULT_FREEZE):
        for relative,expected in _manifest(root/"FREEZE_MANIFEST.sha256" if root==protocol.B4_RESULT_FREEZE else root/"source_sha256.txt").items():
            path=protocol.REPO_ROOT/relative
            if not path.is_file() or protocol.sha256_file(path)!=expected: raise RuntimeError(f"B5_PARENT_HASH_MISMATCH:{relative}")
    frozen=json.loads((protocol.B1_FREEZE/"FINAL_HYPERPARAMETERS.json").read_text())
    values={row["dataset"]:(row["semantic_lr"],row["semantic_exposure"]) for row in frozen["datasets"]}
    if values.get("CCV")!=(1e-5,3) or values.get("ALOI")!=(1e-4,1): raise RuntimeError("B5_B1_EXPOSURE_PARENT_INVALID")
    final_formal_protocol.verify_parent_sources()
    return {"verified":True,"b4_result_freeze":"PASS","b4_conclusion":"NOT_CONFIRMED","b1_exposures":{"CCV":3,"ALOI":1},"method_bone_sha256":protocol.sha256_file(protocol.REPO_ROOT/"method/bone.py"),"method_amv_sha256":protocol.sha256_file(protocol.REPO_ROOT/"method/amv.py")}

def preregister(freeze_dir: str|Path=protocol.PREREGISTERED_FREEZE) -> Path:
    target=Path(freeze_dir)
    if target.exists(): raise FileExistsError("B5_PREREGISTRATION_ALREADY_EXISTS")
    parent=verify_parent_integrity(); target.mkdir(parents=True)
    status="""STATUS=P3_C1_B5_OPTIMIZER_PILOT_PREREGISTERED
DATASETS=CCV,ALOI
DEV_SEEDS=11,37
FOLDS=A,B
OPTIMIZERS=Adam,AdamW,SGD_MOMENTUM
CCV_SEMANTIC_EXPOSURE=3
ALOI_SEMANTIC_EXPOSURE=1
CCV_CANDIDATE_COUNT=9
ALOI_CANDIDATE_COUNT=9
TOTAL_CANDIDATE_COUNT=18
PARAMETER_SEARCH=OPTIMIZER_FAMILY_AND_SEMANTIC_LR_ONLY
EXPOSURE_SEARCH=false
WEIGHT_DECAY_SEARCH=false
MOMENTUM_SEARCH=false
NATIVE_OPTIMIZER_SEARCH=false
FULL_GT_USED=false
FORMAL_METRICS_READ=false
METHOD_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
SEMANTIC_LOSS_CHANGED=false
ABSOLUTE_UTILITY_MASS_FORMULA_CHANGED=false
BOUNDARY_EXTENSION_ALLOWED=false
REAL_OPTIMIZER_CANDIDATES_RUN=false
CONFIRMATORY_SEEDS_RUN=false
"""
    (target/"STATUS.txt").write_text(status)
    (target/"PROTOCOL.txt").write_text("""B5 freezes U_cycle/T_rel/Pi, relation primitive and semantic relation loss. Sparse labels determine WHAT; frozen U_cycle determines WHETHER; semantic optimizer determines only HOW valid actions are absorbed. Absolute utility mass cancellation remains frozen and unresolved. No action-budget coefficient, GT, formal metrics, boundary extension, exposure search, or native optimizer change is permitted. Primary score is the frozen coverage-aware J surrogate, not clustering improvement.\n""")
    write={"datasets":list(protocol.DATASETS),"dev_seeds":list(protocol.DEV_SEEDS),"folds":list(protocol.FOLDS),"candidates":[c.__dict__ for c in protocol.candidates()],"absolute_utility_mass_cancellation":"frozen_and_unresolved"}; protocol.write_json(target/"SEARCH_SPACE.json",write)
    protocol.write_json(target/"SELECTION_RULE.json",{"rule":["maximize round(mean_delta_J,12)","maximize round(worst_delta_J,12)","maximize positive_count","tie: Adam then AdamW then SGD_MOMENTUM","same-family tie: smaller semantic_lr"],"full_gt_used":False})
    from .candidate_reuse import preflight
    protocol.write_json(target/"CANDIDATE_REUSE_PREFLIGHT.json",preflight())
    protocol.write_json(target/"PARENT_INTEGRITY.txt",parent)
    (target/"source_sha256.txt").write_text("".join(f"{d}  {n}\n" for n,d in protocol.source_hashes().items()))
    (target/"parent_freeze_sha256.txt").write_text(f"{protocol.sha256_file(protocol.B4_RESULT_FREEZE/'STATUS.txt')}  experiment_freeze/bone_p3_c1_b4_msrc_historical_default_confirmatory_results_20261004/STATUS.txt\n")
    return target

def verify_preregistration(freeze_dir: str|Path=protocol.PREREGISTERED_FREEZE) -> dict:
    root=Path(freeze_dir); status=(root/"STATUS.txt").read_text()
    if "TOTAL_CANDIDATE_COUNT=18" not in status or "REAL_OPTIMIZER_CANDIDATES_RUN=false" not in status: raise RuntimeError("B5_PREREG_STATUS_INVALID")
    if _manifest(root/"source_sha256.txt")!=protocol.source_hashes(): raise RuntimeError("B5_PREREG_SOURCE_HASH_MISMATCH")
    return {"verified":True,"parent":verify_parent_integrity(),"source_sha256":protocol.source_hashes()}
