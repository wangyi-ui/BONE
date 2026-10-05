"""Deterministic isolated runner for the preregistered upper-LR extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import optuna

from . import protocol
from .boundary_extension_context import (
    BoundaryExtensionContext,
    build_boundary_extension_context,
    evaluate_extension_candidate,
)
from .boundary_extension_protocol import (
    STUDY_DB_NAME,
    ExtensionCandidateConfig,
    ExtensionCandidateEvaluation,
    extension_candidate_key,
    extension_candidates,
    extension_protocol_fingerprint,
    verify_boundary_extension_preconditions,
)
from .formal_tuning_bootstrap_r1 import bootstrap_base_references
from .run_formal_optuna_tuning import dataset_slug, storage_url


def study_name_for_dataset(dataset: str) -> str:
    return f"p3_c1_b1_a2__{dataset_slug(dataset)}__upper_lr_1e3_extension"


def study_path(context: BoundaryExtensionContext) -> Path:
    return context.extension_root / STUDY_DB_NAME


def _sampler() -> optuna.samplers.GridSampler:
    return optuna.samplers.GridSampler({
        "semantic_lr": [1e-3],
        "semantic_exposure": [candidate.semantic_exposure for candidate in extension_candidates()],
    }, seed=0)


def create_or_load_extension_study(
    dataset: str,
    storage_path: str | Path,
) -> optuna.study.Study:
    study = optuna.create_study(
        study_name=study_name_for_dataset(dataset),
        storage=storage_url(storage_path),
        sampler=_sampler(),
        direction="maximize",
        load_if_exists=True,
    )
    fingerprint = extension_protocol_fingerprint()
    existing = study.user_attrs.get("extension_protocol_fingerprint")
    if existing is not None and existing != fingerprint:
        raise RuntimeError("EXTENSION_PROTOCOL_FINGERPRINT_MISMATCH")
    if study.user_attrs.get("dataset") not in (None, dataset):
        raise RuntimeError("EXTENSION_STUDY_DATASET_MISMATCH")
    if existing is None:
        study.set_user_attr("extension_protocol_fingerprint", fingerprint)
        study.set_user_attr("dataset", dataset)
        study.set_user_attr("a2_feasibility_active", True)
        study.set_user_attr("second_boundary_extension_allowed", False)
        study.set_user_attr("full_gt_used", False)
    return study


def _objective(
    dataset: str,
    evaluator: Callable[[ExtensionCandidateConfig], ExtensionCandidateEvaluation],
):
    def run(trial: optuna.trial.Trial) -> float:
        candidate = ExtensionCandidateConfig(
            trial.suggest_categorical("semantic_lr", [1e-3]),
            trial.suggest_categorical(
                "semantic_exposure",
                [candidate.semantic_exposure for candidate in extension_candidates()],
            ),
        )
        evaluation = evaluator(candidate)
        if (
            not isinstance(evaluation, ExtensionCandidateEvaluation)
            or evaluation.dataset != dataset
            or evaluation.candidate_config() != candidate
        ):
            raise ValueError("INVALID_EXTENSION_EVALUATOR_RESULT")
        attrs = {
            "worst_delta_j": evaluation.worst_delta_j,
            "positive_count": evaluation.positive_count,
            "delta_j_seed11_foldA": evaluation.delta_j_seed11_foldA,
            "delta_j_seed11_foldB": evaluation.delta_j_seed11_foldB,
            "delta_j_seed37_foldA": evaluation.delta_j_seed37_foldA,
            "delta_j_seed37_foldB": evaluation.delta_j_seed37_foldB,
            "candidate_key": extension_candidate_key(candidate),
            "full_gt_used": False,
        }
        for name, value in attrs.items():
            trial.set_user_attr(name, value)
        return evaluation.mean_delta_j
    return run


def audit_extension_study(study: optuna.study.Study) -> dict:
    expected = {extension_candidate_key(candidate) for candidate in extension_candidates()}
    complete_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    keys = [trial.user_attrs.get("candidate_key", "") for trial in complete_trials]
    actual = set(keys)
    return {
        "expected_candidate_count": 5,
        "complete_candidate_count": len(keys),
        "unique_complete_candidate_count": len(actual),
        "duplicate_candidate_count": len(keys) - len(actual),
        "missing_candidates": sorted(expected - actual),
        "unregistered_candidates": sorted(actual - expected),
        "failed_trial_count": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
        "pruned_trial_count": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "extension_grid_complete": len(keys) == 5 and actual == expected,
    }


def run_boundary_extension(
    dataset: str,
    device: str,
    formal_tuning_root: str | Path,
    repository_root: str | Path,
    max_new_trials: int | None = None,
    evaluator: Callable[[ExtensionCandidateConfig], ExtensionCandidateEvaluation] | None = None,
    context: BoundaryExtensionContext | None = None,
    bootstrap: Callable[..., object] = bootstrap_base_references,
    precondition: Callable[[str | Path], dict] = verify_boundary_extension_preconditions,
) -> optuna.study.Study:
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    if max_new_trials is not None and (
        isinstance(max_new_trials, bool) or max_new_trials < 0
    ):
        raise ValueError("INVALID_MAX_NEW_TRIALS")
    precondition(repository_root)
    context = context or build_boundary_extension_context(
        dataset, formal_tuning_root, repository_root
    )
    references = bootstrap(context.base_context, device=device)
    if set(references) != set(protocol.DEV_SEEDS):
        raise RuntimeError("NO_VERIFIED_BASES")
    for seed in protocol.DEV_SEEDS:
        if references[seed].native_epochs != protocol.REFINEMENT_HORIZON:
            raise RuntimeError("NO_VERIFIED_BASES")
    if evaluator is None:
        evaluator = lambda candidate: evaluate_extension_candidate(
            context, candidate, device=device
        )
    db_path = study_path(context)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = create_or_load_extension_study(dataset, db_path)
    remaining = 5 - audit_extension_study(study)["complete_candidate_count"]
    if max_new_trials is not None:
        remaining = min(remaining, max_new_trials)
    if remaining > 0:
        study.optimize(_objective(dataset, evaluator), n_trials=remaining, n_jobs=1)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(protocol.DATASETS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--formal-tuning-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--max-new-trials", type=int, default=None)
    args = parser.parse_args()
    study = run_boundary_extension(
        args.dataset, args.device, args.formal_tuning_root,
        args.repository_root, args.max_new_trials,
    )
    print(json.dumps(audit_extension_study(study), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

