from pathlib import Path

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning_r1 import (
    R1_STUDY_DB_NAME, create_or_load_study_r1, run_formal_optuna_tuning_r1,
    study_name_for_dataset_r1,
)
from .study_schema import CandidateEvaluation


def _references(dataset="BDGP"):
    return {
        seed: BaseReference(dataset, seed, Path("base.json"), f"s{seed}", f"p{seed}", 20,
                            0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp")
        for seed in protocol.DEV_SEEDS
    }


def test_bootstrap_runs_before_r1_study_creation_and_uses_versioned_namespace(tmp_path):
    old = tmp_path / "bdgp" / "study.db"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"historical-failed-study")
    before = old.read_bytes()
    order = []

    def bootstrap(context, **kwargs):
        order.append("bootstrap")
        assert not (tmp_path / "bdgp" / R1_STUDY_DB_NAME).exists()
        return _references()

    def evaluator(candidate):
        order.append("candidate")
        return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure, 0.0, 0.0, 0.0, 0.0)

    study = run_formal_optuna_tuning_r1("BDGP", "cpu", tmp_path, max_new_trials=1,
                                        evaluator=evaluator, context=object(), bootstrap=bootstrap)
    assert order == ["bootstrap", "candidate"]
    assert study.study_name == study_name_for_dataset_r1("BDGP")
    assert (tmp_path / "bdgp" / R1_STUDY_DB_NAME).is_file()
    assert old.read_bytes() == before
    assert len(study.trials) == 1
    assert all(trial.state.name == "COMPLETE" for trial in study.trials)


def test_missing_base_prevents_r1_study_creation(tmp_path):
    def failing_bootstrap(context, **kwargs):
        raise RuntimeError("NO_VERIFIED_BASES")

    try:
        run_formal_optuna_tuning_r1("BDGP", "cpu", tmp_path, context=object(), bootstrap=failing_bootstrap)
    except RuntimeError as error:
        assert str(error) == "NO_VERIFIED_BASES"
    else:
        raise AssertionError("missing BASE must fail closed")
    assert not (tmp_path / "bdgp" / R1_STUDY_DB_NAME).exists()


def test_each_missing_base_seed_prevents_r1_study_creation(tmp_path):
    for missing_seed in protocol.DEV_SEEDS:
        root = tmp_path / f"missing_{missing_seed}"

        def incomplete_bootstrap(context, *, missing_seed=missing_seed, **kwargs):
            return {seed: ref for seed, ref in _references().items() if seed != missing_seed}

        try:
            run_formal_optuna_tuning_r1("BDGP", "cpu", root, context=object(), bootstrap=incomplete_bootstrap)
        except RuntimeError as error:
            assert str(error) == "NO_VERIFIED_BASES"
        else:
            raise AssertionError("a missing BASE seed must fail closed")
        assert not (root / "bdgp" / R1_STUDY_DB_NAME).exists()
