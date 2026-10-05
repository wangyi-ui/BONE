from experiments.utility_relation.p3_c1_b1_optuna.optuna_driver import (
    audit_study_coverage, final_selected_candidate, run_remaining_trials,
)
from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateEvaluation


def _evaluator(candidate):
    base = candidate.semantic_exposure * .1 - candidate.semantic_lr * 10
    return CandidateEvaluation("CCV", candidate.semantic_lr, candidate.semantic_exposure,
                               base, base - .01, base + .01, base)


def test_sqlite_resume_from_seven_to_complete_grid_matches_uninterrupted(tmp_path):
    resumed_path = tmp_path / "resumed.db"
    first = run_remaining_trials("CCV", resumed_path, _evaluator, max_new_trials=7)
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    resumed = run_remaining_trials("CCV", resumed_path, _evaluator)
    resumed_audit = audit_study_coverage(resumed)
    assert resumed_audit["complete_candidate_count"] == 25
    assert resumed_audit["unique_complete_candidate_count"] == 25
    assert resumed_audit["duplicate_candidate_count"] == 0
    uninterrupted = run_remaining_trials("CCV", tmp_path / "uninterrupted.db", _evaluator)
    assert final_selected_candidate(resumed) == final_selected_candidate(uninterrupted)
