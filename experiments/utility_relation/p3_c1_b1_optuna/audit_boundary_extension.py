"""Read-only preflight audit for the eight-dataset boundary extension."""

from __future__ import annotations

import json
from pathlib import Path

import optuna

from . import protocol
from .boundary_extension_protocol import verify_boundary_extension_preconditions
from .formal_base_reference import load_verified_base_reference
from .formal_candidate_evaluator_r2 import load_verified_candidate
from .formal_tuning_context_a2 import build_formal_tuning_context_a2
from .optuna_driver import expected_main_candidates
from .run_formal_optuna_tuning import storage_url


def audit_boundary_extension_preflight(
    formal_tuning_root: str | Path,
    repository_root: str | Path,
) -> dict:
    root = Path(formal_tuning_root)
    repository_root = Path(repository_root)
    trigger = verify_boundary_extension_preconditions(repository_root)
    selection_path = (
        repository_root
        / "experiment_freeze/bone_p3_c1_b1_a2_eight_dataset_main_grid_provisional_selection_20261004/PROVISIONAL_SELECTION.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_by_dataset = {item["dataset"]: item for item in selection["datasets"]}
    records = []
    for dataset in protocol.DATASETS:
        context = build_formal_tuning_context_a2(dataset, root)
        carriers = {
            str(seed): context.carrier(seed).directory.is_dir()
            for seed in protocol.DEV_SEEDS
        }
        bases = {}
        for seed in protocol.DEV_SEEDS:
            load_verified_base_reference(context, seed)
            bases[str(seed)] = True
        complete = 0
        for candidate in expected_main_candidates():
            load_verified_candidate(context, candidate)
            complete += 1
        frozen = selection_by_dataset[dataset]
        db_path = repository_root / frozen["study_db"]
        study = optuna.load_study(
            study_name=frozen["study_name"], storage=storage_url(db_path)
        )
        failed = sum(
            trial.state == optuna.trial.TrialState.FAIL for trial in study.trials
        )
        pruned = sum(
            trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
        )
        study_complete = sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
        ready = (
            complete == 25
            and study_complete == 25
            and failed == 0
            and pruned == 0
            and all(carriers.values())
            and all(bases.values())
        )
        records.append({
            "dataset": dataset,
            "main_candidate_count": complete,
            "study_complete_count": study_complete,
            "study_failed_count": failed,
            "study_pruned_count": pruned,
            "base11_verified": bases["11"],
            "base37_verified": bases["37"],
            "carrier11_complete": carriers["11"],
            "carrier37_complete": carriers["37"],
            "extension_ready": ready,
            "full_gt_used": False,
        })
    return {
        "stage": "P3-C1-B1-A2-UPPER-LR-BOUNDARY-EXTENSION",
        "boundary_trigger_verified": True,
        "a2_feasibility_active": trigger["a2_feasibility_active"],
        "datasets": records,
        "ALL_EIGHT_EXTENSION_READY": all(item["extension_ready"] for item in records),
        "real_extension_run": False,
        "full_gt_used": False,
    }

