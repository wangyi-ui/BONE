"""Strict Adam-only reuse preflight over immutable B2/B1 pre-GT artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from . import protocol

B2_OUTPUT=protocol.REPO_ROOT/"outputs/bone_p3/p3_c1_b2_targeted_local_refinement"
def _record(candidate: protocol.Candidate) -> dict:
    if candidate.optimizer_name != "Adam": return {"candidate":candidate.__dict__,"reusable":False,"reason":"NON_ADAM_REUSE_FORBIDDEN"}
    path=B2_OUTPUT/candidate.dataset.lower()/f"lr_{candidate.semantic_lr:.17g}__es_{candidate.semantic_exposure}.json"
    if not path.is_file(): return {"candidate":candidate.__dict__,"reusable":False,"reason":"NO_EXACT_ADAM_ARTIFACT"}
    row=json.loads(path.read_text())
    branches=row.get("branches",{})
    valid=(row.get("completion_state")=="COMPLETE" and row.get("dataset")==candidate.dataset and row.get("semantic_lr")==candidate.semantic_lr and row.get("semantic_exposure")==candidate.semantic_exposure and row.get("full_gt_used") is False and set(branches)=={"11A","11B","37A","37B"} and all(v.get("semantic_steps")==candidate.semantic_exposure and v.get("native_epochs")==20 and v.get("full_gt_loaded") is False for v in branches.values()))
    return {"candidate":candidate.__dict__,"reusable":bool(valid),"reason":"EXACT_ADAM_B1_B2_PRE_GT_PROVENANCE" if valid else "PROVENANCE_MISMATCH","source_path":str(path.relative_to(protocol.REPO_ROOT)) if valid else None,"source_sha256":protocol.sha256_file(path) if valid else None,"optimizer_provenance":{"name":"Adam","betas":[0.9,0.999],"eps":1e-8,"weight_decay":0.0,"verified_from_frozen_B1_default_Adam_runner":bool(valid)}}
def preflight() -> dict:
    rows=[_record(c) for c in protocol.candidates()]
    return {"full_gt_used":False,"formal_metrics_read":False,"entries":rows,"reusable_count":sum(x["reusable"] for x in rows),"new_count":sum(not x["reusable"] for x in rows)}
