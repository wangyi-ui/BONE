"""Exports from completed Optuna trials only; no formal metrics are represented."""

import csv
import json
from pathlib import Path

import optuna

from . import protocol
from .audit import study_audit
from .optuna_driver import final_selected_candidate


def _complete_trials(study):
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def export_trials_csv(study, output_path: str | Path) -> Path:
    path = Path(output_path)
    rows = _complete_trials(study)
    fields = ["number", "semantic_lr", "semantic_exposure", "mean_delta_j", "worst_delta_j", "positive_count", "candidate_key"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in rows:
            writer.writerow({
                "number": trial.number, "semantic_lr": trial.params["semantic_lr"],
                "semantic_exposure": trial.params["semantic_exposure"], "mean_delta_j": trial.value,
                "worst_delta_j": trial.user_attrs["worst_delta_j"],
                "positive_count": trial.user_attrs["positive_count"],
                "candidate_key": trial.user_attrs["candidate_key"],
            })
    return path


def export_best_params_json(study, dataset: str, output_path: str | Path) -> Path:
    selected = final_selected_candidate(study)
    selected_key = f"semantic_lr={selected.semantic_lr:.17g};semantic_exposure={selected.semantic_exposure}"
    trial = next(item for item in _complete_trials(study) if item.user_attrs["candidate_key"] == selected_key)
    payload = {
        "dataset": dataset, "semantic_lr": selected.semantic_lr, "semantic_exposure": selected.semantic_exposure,
        "mean_delta_j": trial.value, "worst_delta_j": trial.user_attrs["worst_delta_j"],
        "positive_count": trial.user_attrs["positive_count"], "selection_rule": "preregistered_lexicographic",
        "dev_seeds": list(protocol.DEV_SEEDS), "formal_seeds": list(protocol.FORMAL_SEEDS),
        "full_gt_used_for_selection": False,
    }
    path = Path(output_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def export_selection_audit_json(study, storage_path: str, dataset: str, output_path: str | Path, best_lrs_by_dataset=None) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(study_audit(study, storage_path, dataset, best_lrs_by_dataset), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
