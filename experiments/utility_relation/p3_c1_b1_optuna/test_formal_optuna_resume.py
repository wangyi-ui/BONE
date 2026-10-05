from . import formal_candidate_evaluator as candidate_module
from . import run_formal_optuna_tuning as runner_module
from .run_formal_optuna_tuning import audit_study_coverage, run_formal_optuna_tuning
from .study_schema import CandidateEvaluation


def test_cached_candidate_artifact_is_reused_without_training(monkeypatch, tmp_path):
    calls = []

    def cached(candidate):
        calls.append(candidate)
        return CandidateEvaluation("BDGP", candidate.semantic_lr, candidate.semantic_exposure, 0.1, 0.1, 0.1, 0.1)

    # The runner's evaluator is the cache boundary: SQLite resume requests only
    # uncompleted grid points, so the already-complete seven are never retrained.
    first = run_formal_optuna_tuning("BDGP", "cpu", tmp_path, max_new_trials=7, evaluator=cached, context=object())
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    before = len(calls)
    resumed = run_formal_optuna_tuning("BDGP", "cpu", tmp_path, evaluator=cached, context=object())
    assert audit_study_coverage(resumed)["complete_candidate_count"] == 25
    assert len(calls) == 25
    assert len(set((item.semantic_lr, item.semantic_exposure) for item in calls)) == 25


def test_candidate_cache_entrypoint_is_explicit():
    assert callable(candidate_module.load_verified_candidate)
    assert "COMPLETE" in open(candidate_module.__file__, encoding="utf-8").read()

