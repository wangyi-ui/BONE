"""Formal Optuna runner for the A2 feasibility protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import optuna

from . import amendment_a2, protocol
from . import run_formal_optuna_tuning as frozen_runner
from .formal_candidate_evaluator_r2 import evaluate_candidate
from .formal_tuning_bootstrap_r1 import bootstrap_base_references
from .formal_tuning_context import FormalTuningContext
from .formal_tuning_context_a2 import build_formal_tuning_context_a2
from .study_schema import CandidateConfig, CandidateEvaluation


A2_STUDY_DB_NAME = "study_a2.db"
A2_STUDY_NAMESPACE_SUFFIX = "__main"


def dataset_slug(dataset: str) -> str:
    return frozen_runner.dataset_slug(dataset)


def study_name_for_dataset_a2(dataset: str) -> str:
    return f"p3_c1_b1_a2__{dataset_slug(dataset)}{A2_STUDY_NAMESPACE_SUFFIX}"


def study_a2_path(formal_tuning_root: str | Path, dataset: str) -> Path:
    return Path(formal_tuning_root) / dataset_slug(dataset) / A2_STUDY_DB_NAME


def create_or_load_study_a2(
    dataset: str,
    storage_path: str | Path,
) -> optuna.study.Study:
    """Create/reopen only the versioned A2 study namespace."""
    study = optuna.create_study(
        study_name=study_name_for_dataset_a2(dataset),
        storage=frozen_runner.storage_url(storage_path),
        sampler=frozen_runner._sampler(),
        direction="maximize",
        load_if_exists=True,
    )
    candidate_fingerprint = frozen_runner.protocol_fingerprint()
    existing = study.user_attrs.get("candidate_protocol_fingerprint")
    if existing is not None and existing != candidate_fingerprint:
        raise RuntimeError("A2_CANDIDATE_PROTOCOL_FINGERPRINT_MISMATCH")
    existing_dataset = study.user_attrs.get("dataset")
    if existing_dataset is not None and existing_dataset != dataset:
        raise RuntimeError("A2_STUDY_DATASET_MISMATCH")
    if existing is None:
        study.set_user_attr(
            "candidate_protocol_fingerprint", candidate_fingerprint
        )
        study.set_user_attr("feasibility_protocol", amendment_a2.AMENDMENT_ID)
        study.set_user_attr("scientific_protocol_changed", True)
        study.set_user_attr("feasibility_protocol_changed", True)
        study.set_user_attr("dataset", dataset)
        study.set_user_attr("full_gt_used", False)
        study.set_user_attr("namespace_version", "A2")
    return study


def run_formal_optuna_tuning_a2(
    dataset: str,
    device: str,
    formal_tuning_root: str | Path,
    max_new_trials: int | None = None,
    evaluator: Callable[[CandidateConfig], CandidateEvaluation] | None = None,
    context: FormalTuningContext | None = None,
    bootstrap: Callable[..., object] = bootstrap_base_references,
) -> optuna.study.Study:
    """Apply A2 feasibility, verify BASE, then run/resume only A2."""
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    if max_new_trials is not None and (
        isinstance(max_new_trials, bool) or max_new_trials < 0
    ):
        raise ValueError("INVALID_MAX_NEW_TRIALS")

    context = context or build_formal_tuning_context_a2(
        dataset, formal_tuning_root
    )
    references = bootstrap(context, device=device)
    if set(references) != set(protocol.DEV_SEEDS):
        raise RuntimeError("NO_VERIFIED_BASES")
    for dev_seed in protocol.DEV_SEEDS:
        if references[dev_seed].native_epochs != protocol.REFINEMENT_HORIZON:
            raise RuntimeError("NO_VERIFIED_BASES")

    if evaluator is None:
        evaluator = lambda candidate: evaluate_candidate(
            context, candidate, device=device
        )
    db_path = study_a2_path(formal_tuning_root, dataset)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = create_or_load_study_a2(dataset, db_path)
    remaining = 25 - frozen_runner.audit_study_coverage(study)[
        "complete_candidate_count"
    ]
    if max_new_trials is not None:
        remaining = min(remaining, max_new_trials)
    if remaining > 0:
        study.optimize(
            frozen_runner._trial_objective(dataset, evaluator),
            n_trials=remaining,
            n_jobs=1,
        )
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(protocol.DATASETS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--formal-tuning-root", type=Path, required=True)
    parser.add_argument("--max-new-trials", type=int, default=None)
    args = parser.parse_args()
    study = run_formal_optuna_tuning_a2(
        args.dataset,
        args.device,
        args.formal_tuning_root,
        args.max_new_trials,
    )
    print(
        json.dumps(
            frozen_runner.audit_study_coverage(study), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()


run_formal_tuning = run_formal_optuna_tuning_a2

