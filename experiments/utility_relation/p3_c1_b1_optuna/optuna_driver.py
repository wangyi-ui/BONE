"""CPU-only GridSampler/SQLite orchestration with an injectable evaluator."""

import hashlib
import json
from pathlib import Path
from typing import Callable

import optuna

from . import objective, protocol
from .study_schema import CandidateConfig, CandidateEvaluation, candidate_key


def protocol_fingerprint() -> str:
    """Hash the frozen values that make a persisted grid scientifically valid."""
    payload = {
        "semantic_lr_grid": protocol.SEMANTIC_LR_GRID,
        "semantic_exposure_grid": protocol.SEMANTIC_EXPOSURE_GRID,
        "dev_seeds": protocol.DEV_SEEDS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def storage_url(storage_path: str | Path) -> str:
    value = str(storage_path)
    if value.startswith("sqlite:///"):
        return value
    return f"sqlite:///{Path(value).resolve()}"


def study_name_for_dataset(dataset: str) -> str:
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in dataset).strip("-")
    return f"p3_c1_b1__{slug}__main"


def expected_main_candidates() -> tuple[CandidateConfig, ...]:
    """Build the 25-point main grid exclusively from frozen protocol data."""
    candidates = tuple(
        CandidateConfig(item.semantic_lr, item.semantic_exposure)
        for item in protocol.main_search_candidates()
    )
    if len(candidates) != 25 or len({candidate_key(item) for item in candidates}) != 25:
        raise RuntimeError("INVALID_FROZEN_MAIN_GRID")
    return candidates


def _sampler() -> optuna.samplers.GridSampler:
    return optuna.samplers.GridSampler(
        {
            "semantic_lr": list(protocol.SEMANTIC_LR_GRID),
            "semantic_exposure": list(protocol.SEMANTIC_EXPOSURE_GRID),
        },
        seed=0,
    )


def create_or_load_study(dataset: str, storage_path: str | Path) -> optuna.study.Study:
    """Create or reopen the deterministic SQLite study after fingerprint check."""
    study = optuna.create_study(
        study_name=study_name_for_dataset(dataset),
        storage=storage_url(storage_path),
        sampler=_sampler(),
        direction="maximize",
        load_if_exists=True,
    )
    expected = protocol_fingerprint()
    existing = study.user_attrs.get("protocol_fingerprint")
    if existing is not None and existing != expected:
        raise RuntimeError("PROTOCOL_FINGERPRINT_MISMATCH")
    if existing is None:
        study.set_user_attr("protocol_fingerprint", expected)
        study.set_user_attr("dataset", dataset)
    elif study.user_attrs.get("dataset") != dataset:
        raise RuntimeError("STUDY_DATASET_MISMATCH")
    return study


def _objective(dataset: str, evaluator: Callable[[CandidateConfig], CandidateEvaluation]):
    def run_trial(trial: optuna.trial.Trial) -> float:
        candidate = CandidateConfig(
            semantic_lr=trial.suggest_categorical("semantic_lr", list(protocol.SEMANTIC_LR_GRID)),
            semantic_exposure=trial.suggest_categorical("semantic_exposure", list(protocol.SEMANTIC_EXPOSURE_GRID)),
        )
        evaluation = evaluator(candidate)
        if not isinstance(evaluation, CandidateEvaluation) or evaluation.dataset != dataset:
            raise ValueError("INVALID_EVALUATOR_RESULT")
        if evaluation.candidate_config() != candidate:
            raise ValueError("EVALUATOR_CANDIDATE_MISMATCH")
        attributes = {
            "worst_delta_j": evaluation.worst_delta_j,
            "positive_count": evaluation.positive_count,
            "delta_j_seed11_foldA": evaluation.delta_j_seed11_foldA,
            "delta_j_seed11_foldB": evaluation.delta_j_seed11_foldB,
            "delta_j_seed37_foldA": evaluation.delta_j_seed37_foldA,
            "delta_j_seed37_foldB": evaluation.delta_j_seed37_foldB,
            "candidate_key": candidate_key(candidate),
        }
        for name, value in attributes.items():
            trial.set_user_attr(name, value)
        return evaluation.mean_delta_j
    return run_trial


def run_remaining_trials(
    dataset: str,
    storage_path: str | Path,
    evaluator: Callable[[CandidateConfig], CandidateEvaluation],
    max_new_trials: int | None = None,
) -> optuna.study.Study:
    """Create/resume one SQLite grid, running at most the requested new trials."""
    if max_new_trials is not None and max_new_trials < 0:
        raise ValueError("INVALID_MAX_NEW_TRIALS")
    study = create_or_load_study(dataset, storage_path)
    remaining = len(expected_main_candidates()) - len(_completed_candidate_keys(study))
    if max_new_trials is not None:
        remaining = min(remaining, max_new_trials)
    if remaining > 0:
        study.optimize(_objective(dataset, evaluator), n_trials=remaining, n_jobs=1)
    return study


def _completed_candidate_keys(study: optuna.study.Study) -> list[str]:
    complete = optuna.trial.TrialState.COMPLETE
    return [
        trial.user_attrs.get("candidate_key", "")
        for trial in study.trials if trial.state == complete
    ]


def audit_study_coverage(study: optuna.study.Study) -> dict:
    """Audit complete-grid coverage without treating failed/pruned trials as valid."""
    expected = {candidate_key(item) for item in expected_main_candidates()}
    keys = _completed_candidate_keys(study)
    actual = set(keys)
    duplicates = len(keys) - len(actual)
    complete = optuna.trial.TrialState.COMPLETE
    failed = optuna.trial.TrialState.FAIL
    pruned = optuna.trial.TrialState.PRUNED
    trial_unregistered = [key for key in actual if key not in expected]
    missing = sorted(expected - actual)
    return {
        "expected_candidate_count": len(expected),
        "complete_candidate_count": len(keys),
        "unique_complete_candidate_count": len(actual),
        "duplicate_candidate_count": duplicates,
        "missing_candidates": missing,
        "unregistered_candidates": sorted(trial_unregistered),
        "failed_trial_count": sum(trial.state == failed for trial in study.trials),
        "pruned_trial_count": sum(trial.state == pruned for trial in study.trials),
        "main_grid_complete": len(keys) == len(expected) and not duplicates and not missing and not trial_unregistered,
    }


def final_selected_candidate(study: optuna.study.Study) -> CandidateConfig:
    """Use the preregistered objective tie-break; never use ``study.best_trial``."""
    records = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        candidate = CandidateConfig(trial.params["semantic_lr"], trial.params["semantic_exposure"])
        key = candidate_key(candidate)
        if key in {candidate_key(item) for item in records}:
            raise RuntimeError("DUPLICATE_COMPLETED_CANDIDATE")
        records[candidate] = {
            "mean_delta_J": trial.value,
            "worst_delta_J": trial.user_attrs["worst_delta_j"],
            "positive_count": trial.user_attrs["positive_count"],
        }
    selected = objective.select_best_candidate(records)
    if candidate_key(selected) not in _completed_candidate_keys(study):
        raise RuntimeError("SELECTED_CANDIDATE_NOT_COMPLETE")
    return selected
