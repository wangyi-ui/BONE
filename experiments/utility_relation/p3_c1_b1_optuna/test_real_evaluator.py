import math

import pytest

from experiments.utility_relation.p3_c1_b1_optuna import objective
from experiments.utility_relation.p3_c1_b1_optuna.real_evaluator import build_candidate_evaluation
from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateEvaluation


def _result(seed, a=.1, b=-.2, **overrides):
    values = {
        "dev_seed": seed, "dataset": "Caltech-5V", "semantic_lr": 1e-5,
        "semantic_exposure": 5, "delta_j_foldA": a, "delta_j_foldB": b,
        "full_gt_loaded": False, "carrier_epochs": 100, "refinement_horizon": 20,
        "base_native_epochs": 20, "ours_foldA_semantic_steps": 5,
        "ours_foldA_native_epochs": 20, "ours_foldB_semantic_steps": 5,
        "ours_foldB_native_epochs": 20, "initial_state_parity": True,
        "zero_utility_foldA": False, "zero_utility_foldB": False,
    }
    values.update(overrides)
    return values


def test_exact_four_score_mapping_and_frozen_aggregation():
    evaluation = build_candidate_evaluation("Caltech-5V", 1e-5, 5, _result(11, .1, -.2), _result(37, .3, .4))
    assert isinstance(evaluation, CandidateEvaluation)
    assert evaluation.delta_j_seed11_foldA == .1
    assert evaluation.delta_j_seed11_foldB == -.2
    assert evaluation.delta_j_seed37_foldA == .3
    assert evaluation.delta_j_seed37_foldB == .4
    frozen = objective.aggregate_candidate_scores(evaluation.delta_scores())
    assert evaluation.mean_delta_j == frozen["mean_delta_J"]
    assert evaluation.worst_delta_j == frozen["worst_delta_J"]
    assert evaluation.positive_count == frozen["positive_count"] == 3


@pytest.mark.parametrize("seed11,seed37", [(_result(37), _result(37)), (_result(11), _result(11))])
def test_wrong_seed_fails(seed11, seed37):
    with pytest.raises(ValueError, match="SEED"):
        build_candidate_evaluation("Caltech-5V", 1e-5, 5, seed11, seed37)


def test_wrong_candidate_or_dataset_fails():
    with pytest.raises(ValueError):
        build_candidate_evaluation("BDGP", 1e-5, 5, _result(11), _result(37))
    with pytest.raises(ValueError):
        build_candidate_evaluation("Caltech-5V", 3e-5, 5, _result(11), _result(37))
    with pytest.raises(ValueError, match="CANDIDATE"):
        build_candidate_evaluation("Caltech-5V", 1e-5, 5, _result(11), _result(37, semantic_exposure=10))


def test_nonfinite_full_gt_zero_utility_and_state_parity_fail_closed():
    invalid = (
        _result(37, a=math.nan),
        _result(37, full_gt_loaded=True),
        _result(37, zero_utility_foldA=True),
        _result(37, initial_state_parity=False),
    )
    for seed37 in invalid:
        with pytest.raises(ValueError):
            build_candidate_evaluation("Caltech-5V", 1e-5, 5, _result(11), seed37)


@pytest.mark.parametrize("field,value", [
    ("carrier_epochs", 99), ("refinement_horizon", 19), ("base_native_epochs", 19),
    ("ours_foldA_semantic_steps", 4), ("ours_foldA_native_epochs", 19),
    ("ours_foldB_semantic_steps", 4), ("ours_foldB_native_epochs", 19),
])
def test_wrong_epoch_counts_fail(field, value):
    with pytest.raises(ValueError, match="EPOCH"):
        build_candidate_evaluation("Caltech-5V", 1e-5, 5, _result(11), _result(37, **{field: value}))
