import optuna
import pytest

from experiments.utility_relation.p3_c1_b1_optuna.optuna_driver import (
    audit_study_coverage, expected_main_candidates, run_remaining_trials,
)
from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateEvaluation, candidate_key


def _evaluator(candidate):
    # Deterministic mock only: no model, data, GT, or randomness.
    base = candidate.semantic_exposure * .01 - candidate.semantic_lr
    return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure,
                               base, base + .001, base + .002, base + .003)


def test_grid_driver_records_only_preregistered_trial_information(tmp_path):
    study = run_remaining_trials("BDGP", tmp_path / "grid.db", _evaluator)
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.direction.name == "MAXIMIZE"
    assert len(expected_main_candidates()) == 25
    coverage = audit_study_coverage(study)
    assert coverage["main_grid_complete"] is True
    assert coverage["complete_candidate_count"] == 25
    assert coverage["duplicate_candidate_count"] == 0
    keys = []
    allowed = {"semantic_lr", "semantic_exposure"}
    required_attrs = {"worst_delta_j", "positive_count", "delta_j_seed11_foldA", "delta_j_seed11_foldB", "delta_j_seed37_foldA", "delta_j_seed37_foldB", "candidate_key"}
    for trial in study.trials:
        assert set(trial.params) == allowed
        assert required_attrs <= set(trial.user_attrs)
        assert not {"ACC", "NMI", "Purity", "ARI"} & set(trial.user_attrs)
        assert trial.value == pytest.approx(trial.user_attrs["delta_j_seed11_foldA"] + .0015)
        keys.append(trial.user_attrs["candidate_key"])
    assert len(keys) == len(set(keys)) == 25


def test_partial_grid_is_not_complete(tmp_path):
    study = run_remaining_trials("BDGP", tmp_path / "partial.db", _evaluator, max_new_trials=7)
    coverage = audit_study_coverage(study)
    assert coverage["complete_candidate_count"] == 7
    assert coverage["main_grid_complete"] is False
