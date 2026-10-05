"""B2 result aggregation/freeze; only reads B2 candidate DeltaJ records."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from . import candidate_reuse
from .protocol import B1_SELECTED, DATASETS, Candidate, DEV_SEEDS, FOLDS, candidates, select_best
from .run_local_refinement import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[3]
RESULT_FREEZE = ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_results_20261004"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=fields); out.writeheader(); out.writerows(rows)


def _records(dataset: str, output_root: Path) -> list[dict]:
    result = []
    for candidate in candidates(dataset):
        path = output_root / dataset.lower() / f"lr_{candidate.semantic_lr:.17g}__es_{candidate.semantic_exposure}.json"
        if not path.is_file():
            raise RuntimeError(f"B2_CANDIDATE_MISSING:{dataset}:{candidate.key()}")
        item = json.loads(path.read_text())
        if item.get("completion_state") != "COMPLETE" or item.get("dataset") != dataset:
            raise RuntimeError("B2_CANDIDATE_INCOMPLETE")
        result.append({"candidate": candidate, **item})
    return result


def finalize(output_root: Path = OUTPUT_ROOT) -> dict:
    all_records = {dataset: _records(dataset, output_root) for dataset in DATASETS}
    winners = {dataset: select_best(records) for dataset, records in all_records.items()}
    selection, summary, flat = [], [], []
    for dataset, winner in winners.items():
        b1_lr, b1_es = B1_SELECTED[dataset]
        baseline = next(item for item in all_records[dataset] if item["candidate"] == Candidate(b1_lr, b1_es))
        selected = {"dataset": dataset, "b1_selected_semantic_lr": b1_lr, "b1_selected_semantic_exposure": b1_es,
                    "b2_selected_semantic_lr": winner["semantic_lr"], "b2_selected_semantic_exposure": winner["semantic_exposure"], "winner_origin": winner["candidate_origin"],
                    "winner_11A": winner["branches"]["11A"]["DeltaJ"], "winner_11B": winner["branches"]["11B"]["DeltaJ"], "winner_37A": winner["branches"]["37A"]["DeltaJ"], "winner_37B": winner["branches"]["37B"]["DeltaJ"],
                    "winner_mean_delta_J": winner["mean_delta_J"], "winner_worst_delta_J": winner["worst_delta_J"], "winner_positive_count": winner["positive_count"],
                    "parameter_changed_from_B1": (winner["semantic_lr"], winner["semantic_exposure"]) != (b1_lr, b1_es), "full_gt_used": False, "formal_metrics_read": False}
        selection.append(selected)
        summary.append({"DATASET": dataset, "B1_LR": b1_lr, "B1_ES": b1_es, "B2_LR": winner["semantic_lr"], "B2_ES": winner["semantic_exposure"],
                        "B2_MEAN_DELTA_J": winner["mean_delta_J"], "B2_WORST_DELTA_J": winner["worst_delta_J"], "B2_POSITIVE_COUNT": winner["positive_count"],
                        "DELTA_MEAN_J_B2_MINUS_B1": winner["mean_delta_J"] - baseline["mean_delta_J"], "DELTA_WORST_J_B2_MINUS_B1": winner["worst_delta_J"] - baseline["worst_delta_J"],
                        "BRANCH_WINS_B2_OVER_B1": sum(winner["branches"][f"{seed}{fold}"]["DeltaJ"] > baseline["branches"][f"{seed}{fold}"]["DeltaJ"] for seed in DEV_SEEDS for fold in FOLDS),
                        "PARAMETER_CHANGED": selected["parameter_changed_from_B1"]})
        for item in all_records[dataset]:
            flat.append({"DATASET": dataset, "SEMANTIC_LR": item["semantic_lr"], "SEMANTIC_EXPOSURE": item["semantic_exposure"], "ORIGIN": item["candidate_origin"],
                         "11A": item["branches"]["11A"]["DeltaJ"], "11B": item["branches"]["11B"]["DeltaJ"], "37A": item["branches"]["37A"]["DeltaJ"], "37B": item["branches"]["37B"]["DeltaJ"],
                         "MEAN_DELTA_J": item["mean_delta_J"], "WORST_DELTA_J": item["worst_delta_J"], "POSITIVE_COUNT": item["positive_count"], "COMPLETION_STATE": item["completion_state"]})
    (output_root / "B2_FINAL_SELECTION.json").write_text(json.dumps({"selections": selection, "full_gt_used": False, "formal_metrics_read": False}, indent=2, sort_keys=True) + "\n")
    _write_csv(output_root / "ALL_CANDIDATES.csv", flat, list(flat[0]))
    _write_csv(output_root / "THREE_DATASET_SUMMARY.csv", summary, list(summary[0]))
    reuse = {"records": [{"dataset": item["dataset"], "semantic_lr": item["semantic_lr"], "semantic_exposure": item["semantic_exposure"], "candidate_origin": item["candidate_origin"], "candidate_path": item["candidate_path"]} for values in all_records.values() for item in values], "reused_count": sum(item["candidate_origin"] != "NEW_B2_RUN" for values in all_records.values() for item in values), "new_count": sum(item["candidate_origin"] == "NEW_B2_RUN" for values in all_records.values() for item in values)}
    (output_root / "CANDIDATE_REUSE_AUDIT.json").write_text(json.dumps(reuse, indent=2, sort_keys=True) + "\n")
    return {"selection": selection, "summary": summary, "reuse": reuse}


def freeze_results(output_root: Path, test_results: str) -> Path:
    if RESULT_FREEZE.exists():
        raise FileExistsError("B2_RESULTS_FREEZE_ALREADY_EXISTS")
    finalized = finalize(output_root)
    RESULT_FREEZE.mkdir(parents=True)
    status = """STATUS=P3_C1_B2_TARGETED_LOCAL_REFINEMENT_RESULTS_FROZEN
STAGE=P3-C1-B2

DEVELOPMENTAL_LOCAL_REFINEMENT=true

DATASET_COUNT=3
DATASETS=CCV,MSRC,ALOI

FULL_GT_USED=false
FORMAL_METRICS_READ=false

METHOD_CHANGED=false
U_CYCLE_CHANGED=false
T_REL_CHANGED=false
SEMANTIC_LOSS_CHANGED=false
ACTION_BUDGET_CHANGED=false

SELECTION_RULE_CHANGED=false
B1_SELECTION_RULE_REUSED=true

BOUNDARY_EXTENSION_USED=false

FINAL_B2_PARAMETERS_FROZEN=true

CONFIRMATORY_SEEDS_RUN=false

NEXT_STAGE=NEW_SEED_CONFIRMATORY_EVALUATION
"""
    (RESULT_FREEZE / "STATUS.txt").write_text(status)
    for source, destination in (("B2_FINAL_SELECTION.json", "B2_FINAL_SELECTION.json"), ("ALL_CANDIDATES.csv", "ALL_CANDIDATES.csv"), ("THREE_DATASET_SUMMARY.csv", "THREE_DATASET_SUMMARY.csv"), ("CANDIDATE_REUSE_AUDIT.json", "CANDIDATE_REUSE_AUDIT.json")):
        shutil.copy2(output_root / source, RESULT_FREEZE / destination)
    source_files = tuple(sorted(str(path.relative_to(ROOT)) for path in (ROOT / "experiments/utility_relation/p3_c1_b2_local_refinement").glob("*.py"))) + ("tests/test_p3_c1_b2_local_refinement.py",)
    (RESULT_FREEZE / "source_sha256.txt").write_text("\n".join(f"{candidate_reuse.sha256_file(ROOT / name)}  {name}" for name in source_files) + "\n")
    dependencies = candidate_reuse.verify_frozen_dependencies()
    (RESULT_FREEZE / "parent_source_sha256.txt").write_text("\n".join(f"{digest}  {name}" for name, digest in dependencies.items()) + "\n")
    parent = ROOT / "experiment_freeze/bone_p3_c1_b2_targeted_local_refinement_preregistered_20261004/STATUS.txt"
    (RESULT_FREEZE / "parent_freeze_sha256.txt").write_text(f"{candidate_reuse.sha256_file(parent)}  {parent.relative_to(ROOT)}\n")
    artifacts = sorted(path for path in output_root.rglob("*.json") if path.name != "B2_FINAL_SELECTION.json")
    (RESULT_FREEZE / "artifact_sha256.txt").write_text("\n".join(f"{candidate_reuse.sha256_file(path)}  {path.relative_to(ROOT)}" for path in artifacts) + "\n")
    (RESULT_FREEZE / "PARENT_INTEGRITY.txt").write_text(json.dumps({"verified": True, "parent_source_modification": False, "candidate_count": 73, "reused": finalized["reuse"]["reused_count"], "new": finalized["reuse"]["new_count"]}, indent=2) + "\n")
    return RESULT_FREEZE
