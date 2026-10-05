from pathlib import Path

import optuna
import pytest

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning import audit_study_coverage
from .run_formal_optuna_tuning_r3 import (
    R3_STUDY_DB_NAME,
    run_formal_optuna_tuning_r3,
    study_name_for_dataset_r3,
    study_r3_path,
)
from .study_schema import CandidateEvaluation


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(
            dataset, seed, Path(f"base{seed}.json"), f"s{seed}", f"p{seed}",
            protocol.REFINEMENT_HORIZON, 0.0, 0.0, {},
            tuple("h" for _ in range(protocol.REFINEMENT_HORIZON)), "fp",
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


def test_canonical_context_and_base_bootstrap_precede_r3_study(tmp_path):
    directory = tmp_path / "bdgp"
    directory.mkdir(parents=True)
    historical = {
        name: directory / name for name in ("study.db", "study_r1.db", "study_r2.db")
    }
    for name, path in historical.items():
        path.write_bytes(f"historical-{name}".encode())
    before = {path: path.read_bytes() for path in historical.values()}
    order = []
    context = object()

    def bootstrap(received, **kwargs):
        order.append("bootstrap")
        assert received is context
        assert not (directory / R3_STUDY_DB_NAME).exists()
        return _references()

    def evaluator(candidate):
        order.append("candidate")
        return _evaluator("BDGP")(candidate)

    study = run_formal_optuna_tuning_r3(
        "BDGP", "cpu", tmp_path, max_new_trials=1,
        evaluator=evaluator, context=context, bootstrap=bootstrap,
    )
    assert order == ["bootstrap", "candidate"]
    assert study.study_name == study_name_for_dataset_r3("BDGP")
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.user_attrs["namespace_version"] == "R3"
    assert all(path.read_bytes() == content for path, content in before.items())


def test_all_eight_datasets_use_uniform_r3_namespace(tmp_path):
    assert len(protocol.DATASETS) == 8
    assert len({study_r3_path(tmp_path, dataset) for dataset in protocol.DATASETS}) == 8
    for dataset in protocol.DATASETS:
        assert study_r3_path(tmp_path, dataset).name == R3_STUDY_DB_NAME
        assert study_name_for_dataset_r3(dataset).endswith("__main_r3")


def test_missing_verified_base_prevents_r3_study_creation(tmp_path):
    def bootstrap(context, **kwargs):
        return {11: _references()[11]}

    with pytest.raises(RuntimeError, match="NO_VERIFIED_BASES"):
        run_formal_optuna_tuning_r3(
            "BDGP", "cpu", tmp_path, context=object(), bootstrap=bootstrap
        )
    assert not (tmp_path / "bdgp" / R3_STUDY_DB_NAME).exists()


def test_r3_clean_study_completes_exact_25_grid(tmp_path):
    study = run_formal_optuna_tuning_r3(
        "BDGP", "cpu", tmp_path, evaluator=_evaluator("BDGP"),
        context=object(), bootstrap=lambda context, **kwargs: _references(),
    )
    coverage = audit_study_coverage(study)
    assert coverage["main_grid_complete"] is True
    assert coverage["complete_candidate_count"] == 25
    assert coverage["unique_complete_candidate_count"] == 25
    assert coverage["failed_trial_count"] == 0
    assert coverage["duplicate_candidate_count"] == 0


def test_r3_sqlite_resume_7_to_25(tmp_path):
    calls = []
    first = run_formal_optuna_tuning_r3(
        "CCV", "cpu", tmp_path, max_new_trials=7,
        evaluator=_evaluator("CCV", calls), context=object(),
        bootstrap=lambda context, **kwargs: _references("CCV"),
    )
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    resumed = run_formal_optuna_tuning_r3(
        "CCV", "cpu", tmp_path, evaluator=_evaluator("CCV", calls),
        context=object(), bootstrap=lambda context, **kwargs: _references("CCV"),
    )
    assert audit_study_coverage(resumed)["main_grid_complete"] is True
    assert len(calls) == len(set(calls)) == 25


def test_r3_sources_are_no_gt_and_protocol_neutral():
    root = Path(__file__).parent
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "canonical_fold_order_r3.py",
            "formal_tuning_context_r3.py",
            "run_formal_optuna_tuning_r3.py",
        )
    )
    assert not any(
        token in text
        for token in ("ACC", "NMI", "Purity", "ARI", "FORMAL_SEEDS", "seed20", "seed30", "seed50")
    )
    assert "trial_runner_a1.py" not in text
    assert protocol.SEMANTIC_LR_GRID == (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
    assert protocol.SEMANTIC_EXPOSURE_GRID == (1, 3, 5, 10, 20)
    assert protocol.REFINEMENT_HORIZON == 20

