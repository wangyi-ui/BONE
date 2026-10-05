"""Finalize one complete 25-candidate dataset search using frozen selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import optuna

try:
    from . import objective, protocol
    from .formal_candidate_evaluator import candidate_directory, load_verified_candidate
    from .formal_tuning_context import build_formal_tuning_context
    from .run_formal_optuna_tuning import audit_study_coverage, create_or_load_study, dataset_slug, protocol_fingerprint
    from .study_schema import CandidateConfig, candidate_key
except ImportError:  # pragma: no cover
    from experiments.utility_relation.p3_c1_b1_optuna import objective, protocol
    from experiments.utility_relation.p3_c1_b1_optuna.formal_candidate_evaluator import candidate_directory, load_verified_candidate
    from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context import build_formal_tuning_context
    from experiments.utility_relation.p3_c1_b1_optuna.run_formal_optuna_tuning import audit_study_coverage, create_or_load_study, dataset_slug, protocol_fingerprint
    from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateConfig, candidate_key


def _complete_trials(study):
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def _verified_candidate_keys(context, study):
    records = {}
    for trial in _complete_trials(study):
        candidate = CandidateConfig(trial.params["semantic_lr"], trial.params["semantic_exposure"])
        evaluation = load_verified_candidate(context, candidate)
        if evaluation.mean_delta_j != trial.value:
            raise RuntimeError("CANDIDATE_STUDY_VALUE_MISMATCH")
        records[candidate] = {
            "mean_delta_J": evaluation.mean_delta_j,
            "worst_delta_J": evaluation.worst_delta_j,
            "positive_count": evaluation.positive_count,
        }
    return records


def finalize_formal_tuning(
    dataset: str,
    formal_tuning_root: str | Path,
    output_root: str | Path | None = None,
    best_lrs_by_dataset: dict[str, float] | None = None,
) -> dict:
    context = build_formal_tuning_context(dataset, formal_tuning_root)
    db_path = Path(formal_tuning_root) / dataset_slug(dataset) / "study.db"
    study = create_or_load_study(dataset, db_path)
    coverage = audit_study_coverage(study)
    if not coverage["main_grid_complete"]:
        raise RuntimeError("CANNOT_FINALIZE_BEFORE_25_OF_25")
    records = _verified_candidate_keys(context, study)
    if len(records) != 25:
        raise RuntimeError("CANDIDATE_ARTIFACT_COVERAGE_MISMATCH")
    selected = objective.select_best_candidate(records)
    selected_key = candidate_key(selected)
    selected_trial = next(t for t in _complete_trials(study) if t.user_attrs.get("candidate_key") == selected_key)
    out = Path(output_root) if output_root is not None else Path(formal_tuning_root) / dataset_slug(dataset)
    out.mkdir(parents=True, exist_ok=True)
    best = {
        "dataset": dataset, "semantic_lr": selected.semantic_lr,
        "semantic_exposure": selected.semantic_exposure,
        "mean_delta_j": records[selected]["mean_delta_J"],
        "worst_delta_j": records[selected]["worst_delta_J"],
        "positive_count": records[selected]["positive_count"],
        "delta_j_seed11_foldA": selected_trial.user_attrs["delta_j_seed11_foldA"],
        "delta_j_seed11_foldB": selected_trial.user_attrs["delta_j_seed11_foldB"],
        "delta_j_seed37_foldA": selected_trial.user_attrs["delta_j_seed37_foldA"],
        "delta_j_seed37_foldB": selected_trial.user_attrs["delta_j_seed37_foldB"],
        "selection_rule": "preregistered_lexicographic",
        "dev_seeds": list(protocol.DEV_SEEDS),
        "full_gt_used_for_selection": False,
        "protocol_fingerprint": protocol_fingerprint(),
    }
    boundaries = {
        "lower_lr_boundary_hit": selected.semantic_lr == protocol.SEMANTIC_LR_GRID[0],
        "upper_lr_boundary_hit": selected.semantic_lr == protocol.SEMANTIC_LR_GRID[-1],
        "semantic_exposure_upper_bound_hit": protocol.exposure_upper_bound_hit(selected.semantic_exposure),
    }
    best["boundary_flags"] = boundaries
    (out / "best_params.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["number", "semantic_lr", "semantic_exposure", "mean_delta_j", "worst_delta_j", "positive_count", "candidate_key"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for trial in _complete_trials(study):
            writer.writerow({
                "number": trial.number, "semantic_lr": trial.params["semantic_lr"],
                "semantic_exposure": trial.params["semantic_exposure"], "mean_delta_j": trial.value,
                "worst_delta_j": trial.user_attrs["worst_delta_j"],
                "positive_count": trial.user_attrs["positive_count"],
                "candidate_key": trial.user_attrs["candidate_key"],
            })
    selection_audit = {
        "dataset": dataset, "study_name": study.study_name, "storage_path": str(db_path),
        "protocol_fingerprint": protocol_fingerprint(), **coverage,
        "verified_candidate_count": len(records), "selected_candidate": best,
        "boundary_flags": boundaries, "full_gt_used": False,
    }
    (out / "selection_audit.json").write_text(json.dumps(selection_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(protocol.DATASETS), required=True)
    parser.add_argument("--formal-tuning-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(finalize_formal_tuning(args.dataset, args.formal_tuning_root, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


finalize = finalize_formal_tuning
