import optuna

from . import protocol
from .run_formal_optuna_tuning import (
    audit_study_coverage, run_formal_optuna_tuning, study_name_for_dataset,
)
from .study_schema import CandidateEvaluation


def _evaluator(candidate):
    value = candidate.semantic_exposure * 0.01 - candidate.semantic_lr
    return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure, value, value, value, value)


def test_grid_sampler_registers_exactly_25_and_only_two_params(tmp_path):
    study = run_formal_optuna_tuning("BDGP", "cpu", tmp_path, evaluator=_evaluator, context=object())
    coverage = audit_study_coverage(study)
    assert isinstance(study.sampler, optuna.samplers.GridSampler)
    assert study.study_name == study_name_for_dataset("BDGP")
    assert coverage["complete_candidate_count"] == 25
    assert coverage["main_grid_complete"] is True
    for trial in study.trials:
        assert set(trial.params) == {"semantic_lr", "semantic_exposure"}
        assert trial.user_attrs["full_gt_used"] is False
        assert trial.user_attrs["protocol_fingerprint"]


def test_runner_partial_grid_is_resumable(tmp_path):
    evaluator = lambda candidate: CandidateEvaluation("CCV", candidate.semantic_lr, candidate.semantic_exposure,
                                                       candidate.semantic_exposure * 0.01 - candidate.semantic_lr,
                                                       0.0, 0.0, 0.0)
    first = run_formal_optuna_tuning("CCV", "cpu", tmp_path, max_new_trials=7, evaluator=evaluator, context=object())
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    resumed = run_formal_optuna_tuning("CCV", "cpu", tmp_path, evaluator=evaluator, context=object())
    assert audit_study_coverage(resumed)["complete_candidate_count"] == 25
