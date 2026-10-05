from pathlib import Path

from . import protocol
from .formal_base_reference import BaseReference
from .run_formal_optuna_tuning_r1 import run_formal_optuna_tuning_r1
from .run_formal_optuna_tuning import audit_study_coverage
from .study_schema import CandidateEvaluation


def test_r1_partial_resume_reaches_25_without_rebootstrapping_training(tmp_path):
    references = {
        seed: BaseReference("CCV", seed, Path("base.json"), f"s{seed}", f"p{seed}", 20,
                            0.0, 0.0, {}, tuple("h" for _ in range(20)), "fp")
        for seed in protocol.DEV_SEEDS
    }
    bootstrap_calls = []
    evaluator_calls = []

    def bootstrap(context, **kwargs):
        bootstrap_calls.append(1)
        return references

    def evaluator(candidate):
        evaluator_calls.append((candidate.semantic_lr, candidate.semantic_exposure))
        return CandidateEvaluation("CCV", candidate.semantic_lr, candidate.semantic_exposure, 0.0, 0.0, 0.0, 0.0)

    first = run_formal_optuna_tuning_r1("CCV", "cpu", tmp_path, max_new_trials=7,
                                        evaluator=evaluator, context=object(), bootstrap=bootstrap)
    assert audit_study_coverage(first)["complete_candidate_count"] == 7
    second = run_formal_optuna_tuning_r1("CCV", "cpu", tmp_path,
                                         evaluator=evaluator, context=object(), bootstrap=bootstrap)
    assert audit_study_coverage(second)["complete_candidate_count"] == 25
    assert len(bootstrap_calls) == 2
    assert len(evaluator_calls) == 25
    assert len(set(evaluator_calls)) == 25

