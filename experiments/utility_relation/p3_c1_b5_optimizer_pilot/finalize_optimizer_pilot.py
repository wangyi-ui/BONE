"""Pure pre-GT B5 selector; formal metrics and GT are forbidden."""
from __future__ import annotations
import json
from pathlib import Path
from . import protocol
from .candidate_reuse import preflight
from .training_wiring import candidate_artifact_path, verify_candidate_artifact

def select_best(records: list[dict]) -> dict:
    if len(records)!=9 or any(record.get("full_gt_used") is not False for record in records): raise ValueError("B5_INCOMPLETE_OR_GT_CONTAMINATED_CANDIDATES")
    return max(records,key=protocol.selection_key)

def score_record(candidate: protocol.Candidate, deltas: dict[str,float]) -> dict:
    if set(deltas)!={"11A","11B","37A","37B"}: raise ValueError("B5_BRANCH_SET_INVALID")
    values=list(deltas.values())
    return {"candidate":candidate,"DeltaJ":deltas,"mean_delta_J":sum(values)/4,"worst_delta_J":min(values),"positive_count":sum(value>0 for value in values),"full_gt_used":False,"formal_metrics_read":False}


def _record_from_reused_adam(entry: dict) -> dict:
    if not entry.get("reusable") or entry["candidate"]["optimizer_name"] != "Adam":
        raise RuntimeError("B5_ADAM_REUSE_INCOMPLETE")
    path = protocol.REPO_ROOT / entry["source_path"]
    if not path.is_file() or protocol.sha256_file(path) != entry.get("source_sha256"):
        raise RuntimeError("B5_ADAM_REUSE_ARTIFACT_HASH_INVALID")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("completion_state") != "COMPLETE" or payload.get("full_gt_used") is not False:
        raise RuntimeError("B5_ADAM_REUSE_ARTIFACT_INVALID")
    candidate = protocol.Candidate(**entry["candidate"])
    branches = payload.get("branches", {})
    if set(branches) != set(("11A", "11B", "37A", "37B")):
        raise RuntimeError("B5_ADAM_REUSE_BRANCH_SET_INVALID")
    deltas = {key: float(branches[key]["DeltaJ"]) for key in branches}
    return score_record(candidate, deltas)


def load_complete_new_candidate_records(output_root: str | Path = protocol.OUTPUT_ROOT) -> list[dict]:
    """Load all twelve new artifacts; missing or partial results fail closed."""
    records = []
    for candidate in protocol.candidates():
        if candidate.optimizer_name == "Adam":
            continue
        payload = verify_candidate_artifact(candidate_artifact_path(output_root, candidate), candidate)
        deltas = {key: float(payload["branches"][key]["DeltaJ"]) for key in ("11A", "11B", "37A", "37B")}
        record = score_record(candidate, deltas)
        if payload.get("real_training_completed") is not True:
            raise RuntimeError("B5_NEW_CANDIDATE_NOT_COMPLETE")
        records.append(record)
    if len(records) != 12:
        raise RuntimeError("B5_REAL_NEW_CANDIDATE_COUNT_INCOMPLETE")
    return records


def load_complete_candidate_records(output_root: str | Path = protocol.OUTPUT_ROOT) -> list[dict]:
    """Require exactly six reusable Adam plus twelve complete real candidates."""
    reuse = preflight()["entries"]
    adam_entries = [entry for entry in reuse if entry["candidate"]["optimizer_name"] == "Adam"]
    if len(adam_entries) != 6 or any(not entry.get("reusable") for entry in adam_entries):
        raise RuntimeError("B5_ADAM_REUSE_COUNT_INCOMPLETE")
    records = [_record_from_reused_adam(entry) for entry in adam_entries]
    records.extend(load_complete_new_candidate_records(output_root))
    if len(records) != 18:
        raise RuntimeError("B5_EXACT_18_CANDIDATES_REQUIRED")
    counts = {dataset: sum(record["candidate"].dataset == dataset for record in records) for dataset in protocol.DATASETS}
    if counts != {"CCV": 9, "ALOI": 9}:
        raise RuntimeError("B5_DATASET_CANDIDATE_COUNT_INVALID")
    if any(record.get("full_gt_used") is not False or record.get("formal_metrics_read") is not False for record in records):
        raise RuntimeError("B5_FINALIZER_GT_CONTAMINATION")
    return records


def select_best_complete(output_root: str | Path = protocol.OUTPUT_ROOT) -> dict:
    records = load_complete_candidate_records(output_root)
    return max(records, key=protocol.selection_key)
