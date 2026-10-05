from pathlib import Path

import optuna
import pytest

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning import audit_study_coverage
from .run_formal_optuna_tuning_r2 import (
    R2_STUDY_DB_NAME,
    run_formal_optuna_tuning_r2,
    study_name_for_dataset_r2,
)
from .study_schema import CandidateEvaluation


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(
            dataset,
            seed,
            Path(f"base{seed}.json"),
            f"s{seed}",
            f"p{seed}",
            protocol.REFINEMENT_HORIZON,
            0.0,
            0.0,
            {},
            tuple("h" for _ in range(protocol.REFINEMENT_HORIZON)),
            "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def _evaluator(dataset, calls=None):
    def evaluate(candidate):
        if calls is not None:
            calls.append((candidate.semantic_lr, candidate.semantic_exposure))
        value = candidate.semantic_exposure * 0.01 - candidate.semantic_lr
        return CandidateEvaluation(
            dataset,
            candidate.semantic_lr,
            candidate.semantic_exposure,
            value,
            value,
            value,
            value,
        )

    return evaluate


def test_base_bootstrap_precedes_r2_study_and_both_old_dbs_are_preserved(tmp_path):
    directory = tmp_path / "bdgp"
    directory.mkdir(parents=True)
    old = directory / "study.db"
    r1 = directory / "study_r1.db"
    old.write_bytes(b"old failed study")
    r1.write_bytes(b"r1 failed study")
    before = {old: old.read_bytes(), r1: r1.read_bytes()}
    order = []

    def bootstrap(context, **kwargs):
        order.append("bootstrap")
        assert not (directory / R2_STUDY_DB_NAME).exists()
        return _references()

    def evaluator(candidate):
        order.append("candidate")
        return _evaluator("BDGP")(candidate)

    study = run_formal_optuna_tuning_r2(
        "BDGP",
        "cpu",
        tmp_path,
        max_new_trials=1,
        evaluator=evaluator,
        context=object(),
        bootstrap=bootstrap,
    )
    assert order == ["bootstrap", "candidate"]
    assert study.study_name == study_name_for_dataset_r2("BDGP")
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.user_attrs["namespace_version"] == "R2"
    assert len(study.trials) == 1
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("missing_seed", protocol.DEV_SEEDS)
def test_each_missing_base_prevents_r2_study_creation(tmp_path, missing_seed):
    def bootstrap(context, **kwargs):
        return {
            seed: reference
            for seed, reference in _references().items()
            if seed != missing_seed
        }

    with pytest.raises(RuntimeError, match="NO_VERIFIED_BASES"):
        run_formal_optuna_tuning_r2(
            "BDGP", "cpu", tmp_path, context=object(), bootstrap=bootstrap
        )
    assert not (tmp_path / "bdgp" / R2_STUDY_DB_NAME).exists()


def test_r2_grid_has_exactly_25_canonical_candidate_identities(tmp_path):
    study = run_formal_optuna_tuning_r2(
        "BDGP",
        "cpu",
        tmp_path,
        evaluator=_evaluator("BDGP"),
        context=object(),
        bootstrap=lambda context, **kwargs: _references(),
    )
    coverage = audit_study_coverage(study)
    assert coverage == {
        "expected_candidate_count": 25,
        "complete_candidate_count": 25,
        "unique_complete_candidate_count": 25,
        "duplicate_candidate_count": 0,
        "missing_candidates": [],
        "unregistered_candidates": [],
        "failed_trial_count": 0,
        "pruned_trial_count": 0,
        "main_grid_complete": True,
    }
    assert all(set(trial.params) == {"semantic_lr", "semantic_exposure"}
               for trial in study.trials)
    assert all(trial.user_attrs["full_gt_used"] is False for trial in study.trials)


def test_r2_sqlite_resume_from_7_to_25_without_duplicate_candidates(tmp_path):
    calls = []
    bootstrap_calls = []

    def bootstrap(context, **kwargs):
        bootstrap_calls.append(1)
        return _references("CCV")

    first = run_formal_optuna_tuning_r2(
        "CCV", "cpu", tmp_path, max_new_trials=7,
        evaluator=_evaluator("CCV", calls), context=object(), bootstrap=bootstrap,
    )
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    resumed = run_formal_optuna_tuning_r2(
        "CCV", "cpu", tmp_path,
        evaluator=_evaluator("CCV", calls), context=object(), bootstrap=bootstrap,
    )
    coverage = audit_study_coverage(resumed)
    assert coverage["main_grid_complete"] is True
    assert len(bootstrap_calls) == 2
    assert len(calls) == len(set(calls)) == 25

