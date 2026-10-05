"""One-dataset formal Optuna GridSampler runner (execution is opt-in)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import optuna

try:  # Supports both ``python -m`` and the documented file-path CLI.
    from . import amendment_a1, protocol
    from .formal_candidate_evaluator import evaluate_candidate
    from .formal_tuning_context import FormalTuningContext, build_formal_tuning_context
    from .optuna_driver import expected_main_candidates
    from .study_schema import CandidateConfig, CandidateEvaluation, candidate_key
except ImportError:  # pragma: no cover - exercised by CLI smoke, not package tests
    from experiments.utility_relation.p3_c1_b1_optuna import amendment_a1, protocol
    from experiments.utility_relation.p3_c1_b1_optuna.formal_candidate_evaluator import evaluate_candidate
    from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context import FormalTuningContext, build_formal_tuning_context
    from experiments.utility_relation.p3_c1_b1_optuna.optuna_driver import expected_main_candidates
    from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateConfig, CandidateEvaluation, candidate_key


def dataset_slug(dataset: str) -> str:
    from .formal_carrier_materializer import DATASET_SLUGS
    if dataset not in DATASET_SLUGS:
        raise ValueError("UNREGISTERED_DATASET")
    return DATASET_SLUGS[dataset]


def study_name_for_dataset(dataset: str) -> str:
    return f"p3_c1_b1_a1__{dataset_slug(dataset)}__main"


def protocol_fingerprint() -> str:
    payload = {
        "amendment_id": amendment_a1.AMENDMENT_ID,
        "semantic_lr_grid": list(protocol.SEMANTIC_LR_GRID),
        "semantic_exposure_grid": list(protocol.SEMANTIC_EXPOSURE_GRID),
        "dev_seeds": list(protocol.DEV_SEEDS),
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def storage_url(storage_path: str | Path) -> str:
    value = str(storage_path)
    return value if value.startswith("sqlite:///") else f"sqlite:///{Path(value).resolve()}"


def _sampler() -> optuna.samplers.GridSampler:
    return optuna.samplers.GridSampler({
        "semantic_lr": list(protocol.SEMANTIC_LR_GRID),
        "semantic_exposure": list(protocol.SEMANTIC_EXPOSURE_GRID),
    }, seed=0)


def create_or_load_study(dataset: str, storage_path: str | Path) -> optuna.study.Study:
    study = optuna.create_study(
        study_name=study_name_for_dataset(dataset), storage=storage_url(storage_path),
        sampler=_sampler(), direction="maximize", load_if_exists=True,
    )
    fingerprint = protocol_fingerprint()
    existing = study.user_attrs.get("protocol_fingerprint")
    if existing is not None and existing != fingerprint:
        raise RuntimeError("PROTOCOL_FINGERPRINT_MISMATCH")
    if existing is None:
        study.set_user_attr("protocol_fingerprint", fingerprint)
        study.set_user_attr("dataset", dataset)
        study.set_user_attr("full_gt_used", False)
    elif study.user_attrs.get("dataset") != dataset:
        raise RuntimeError("STUDY_DATASET_MISMATCH")
    return study


def _completed_keys(study: optuna.study.Study) -> set[str]:
    return {trial.user_attrs.get("candidate_key", "") for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE}


def audit_study_coverage(study: optuna.study.Study) -> dict:
    expected = {candidate_key(item) for item in expected_main_candidates()}
    keys = [trial.user_attrs.get("candidate_key", "") for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("candidate_key", "")]
    actual = set(keys)
    return {
        "expected_candidate_count": len(expected),
        "complete_candidate_count": len(keys),
        "unique_complete_candidate_count": len(actual),
        "duplicate_candidate_count": len(keys) - len(actual),
        "missing_candidates": sorted(expected - actual),
        "unregistered_candidates": sorted(actual - expected),
        "failed_trial_count": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "pruned_trial_count": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "main_grid_complete": len(keys) == 25 and len(actual) == 25 and actual == expected,
    }


def _trial_objective(dataset: str, evaluator: Callable[[CandidateConfig], CandidateEvaluation]):
    def objective(trial: optuna.trial.Trial) -> float:
        candidate = CandidateConfig(
            trial.suggest_categorical("semantic_lr", list(protocol.SEMANTIC_LR_GRID)),
            trial.suggest_categorical("semantic_exposure", list(protocol.SEMANTIC_EXPOSURE_GRID)),
        )
        evaluation = evaluator(candidate)
        if not isinstance(evaluation, CandidateEvaluation) or evaluation.dataset != dataset:
            raise ValueError("INVALID_EVALUATOR_RESULT")
        if evaluation.candidate_config() != candidate:
            raise ValueError("EVALUATOR_CANDIDATE_MISMATCH")
        attrs = {
            "worst_delta_j": evaluation.worst_delta_j,
            "positive_count": evaluation.positive_count,
            "delta_j_seed11_foldA": evaluation.delta_j_seed11_foldA,
            "delta_j_seed11_foldB": evaluation.delta_j_seed11_foldB,
            "delta_j_seed37_foldA": evaluation.delta_j_seed37_foldA,
            "delta_j_seed37_foldB": evaluation.delta_j_seed37_foldB,
            "candidate_key": candidate_key(candidate),
            "protocol_fingerprint": protocol_fingerprint(),
            "full_gt_used": False,
        }
        for name, value in attrs.items():
            trial.set_user_attr(name, value)
        return evaluation.mean_delta_j
    return objective


def run_formal_optuna_tuning(
    dataset: str,
    device: str,
    formal_tuning_root: str | Path,
    max_new_trials: int | None = None,
    evaluator: Callable[[CandidateConfig], CandidateEvaluation] | None = None,
    context: FormalTuningContext | None = None,
) -> optuna.study.Study:
    """Run/resume one dataset's 25-point grid; never auto-runs other datasets."""
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    if max_new_trials is not None and (isinstance(max_new_trials, bool) or max_new_trials < 0):
        raise ValueError("INVALID_MAX_NEW_TRIALS")
    context = context or build_formal_tuning_context(dataset, formal_tuning_root)
    if evaluator is None:
        evaluator = lambda candidate: evaluate_candidate(context, candidate, device=device)
    db_path = Path(formal_tuning_root) / dataset_slug(dataset) / "study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = create_or_load_study(dataset, db_path)
    remaining = 25 - audit_study_coverage(study)["complete_candidate_count"]
    if max_new_trials is not None:
        remaining = min(remaining, max_new_trials)
    if remaining > 0:
        study.optimize(_trial_objective(dataset, evaluator), n_trials=remaining, n_jobs=1)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(protocol.DATASETS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--formal-tuning-root", type=Path, required=True)
    parser.add_argument("--max-new-trials", type=int, default=None)
    args = parser.parse_args()
    study = run_formal_optuna_tuning(args.dataset, args.device, args.formal_tuning_root, args.max_new_trials)
    print(json.dumps(audit_study_coverage(study), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


run_formal_tuning = run_formal_optuna_tuning
