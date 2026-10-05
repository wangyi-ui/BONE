from pathlib import Path

import optuna
import pytest

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning import audit_study_coverage
from .run_formal_optuna_tuning_a2 import (
    A2_STUDY_DB_NAME,
    run_formal_optuna_tuning_a2,
    study_a2_path,
    study_name_for_dataset_a2,
)
from .study_schema import CandidateEvaluation


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(
            dataset, seed, Path(f"base{seed}.json"), f"s{seed}", f"p{seed}",
            20, 0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp",
        )
        for seed in protocol.DEV_SEEDS
    }


def _evaluator(dataset, calls=None):
    def evaluate(candidate):
        if calls is not None:
            calls.append((candidate.semantic_lr, candidate.semantic_exposure))
        value = candidate.semantic_exposure * 0.01 - candidate.semantic_lr
        return CandidateEvaluation(
            dataset, candidate.semantic_lr, candidate.semantic_exposure,
            value, value, value, value,
        )
    return evaluate


def test_a2_bootstrap_precedes_new_study_and_old_studies_are_unchanged(tmp_path):
    directory = tmp_path / "bdgp"
    directory.mkdir(parents=True)
    old_paths = [directory / name for name in (
        "study.db", "study_r1.db", "study_r2.db", "study_r3.db"
    )]
    for path in old_paths:
        path.write_bytes(f"old-{path.name}".encode())
    before = {path: path.read_bytes() for path in old_paths}
    order = []

    def bootstrap(context, **kwargs):
        order.append("bootstrap")
        assert not (directory / A2_STUDY_DB_NAME).exists()
        return _references()

    def evaluator(candidate):
        order.append("candidate")
        return _evaluator("BDGP")(candidate)

    study = run_formal_optuna_tuning_a2(
        "BDGP", "cpu", tmp_path, max_new_trials=1,
        evaluator=evaluator, context=object(), bootstrap=bootstrap,
    )
    assert order == ["bootstrap", "candidate"]
    assert study.study_name == study_name_for_dataset_a2("BDGP")
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.user_attrs["feasibility_protocol"] == "P3-C1-B1-A2"
    assert study.user_attrs["scientific_protocol_changed"] is True
    assert all(path.read_bytes() == content for path, content in before.items())


def test_all_datasets_use_uniform_a2_namespace(tmp_path):
    assert len({study_a2_path(tmp_path, dataset) for dataset in protocol.DATASETS}) == 8
    for dataset in protocol.DATASETS:
        assert study_a2_path(tmp_path, dataset).name == A2_STUDY_DB_NAME
        assert study_name_for_dataset_a2(dataset).startswith("p3_c1_b1_a2__")
        assert study_name_for_dataset_a2(dataset).endswith("__main")


def test_missing_base_prevents_a2_study_creation(tmp_path):
    with pytest.raises(RuntimeError, match="NO_VERIFIED_BASES"):
        run_formal_optuna_tuning_a2(
            "BDGP", "cpu", tmp_path, context=object(),
            bootstrap=lambda context, **kwargs: {11: _references()[11]},
        )
    assert not (tmp_path / "bdgp" / A2_STUDY_DB_NAME).exists()


def test_a2_exact_25_grid_and_no_failed_trials(tmp_path):
    study = run_formal_optuna_tuning_a2(
        "BDGP", "cpu", tmp_path, evaluator=_evaluator("BDGP"),
        context=object(), bootstrap=lambda context, **kwargs: _references(),
    )
    coverage = audit_study_coverage(study)
    assert coverage["main_grid_complete"] is True
    assert coverage["complete_candidate_count"] == 25
    assert coverage["unique_complete_candidate_count"] == 25
    assert coverage["failed_trial_count"] == 0


def test_a2_sqlite_resume_7_to_25(tmp_path):
    calls = []
    first = run_formal_optuna_tuning_a2(
        "CCV", "cpu", tmp_path, max_new_trials=7,
        evaluator=_evaluator("CCV", calls), context=object(),
        bootstrap=lambda context, **kwargs: _references("CCV"),
    )
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    resumed = run_formal_optuna_tuning_a2(
        "CCV", "cpu", tmp_path, evaluator=_evaluator("CCV", calls),
        context=object(), bootstrap=lambda context, **kwargs: _references("CCV"),
    )
    assert audit_study_coverage(resumed)["main_grid_complete"] is True
    assert len(calls) == len(set(calls)) == 25

