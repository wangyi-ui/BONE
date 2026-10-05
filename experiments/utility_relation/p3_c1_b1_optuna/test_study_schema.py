import pytest

from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateConfig, CandidateEvaluation, candidate_key


def _evaluation(**overrides):
    values = dict(dataset="BDGP", semantic_lr=1e-5, semantic_exposure=5,
                  delta_j_seed11_foldA=.1, delta_j_seed11_foldB=.2,
                  delta_j_seed37_foldA=.3, delta_j_seed37_foldB=.4)
    values.update(overrides)
    return CandidateEvaluation(**values)


def test_registered_config_and_deterministic_key():
    config = CandidateConfig(1e-5, 5)
    assert candidate_key(config) == candidate_key(CandidateConfig(1e-5, 5))
    assert "semantic_lr=" in candidate_key(config)


def test_unregistered_config_is_rejected():
    with pytest.raises(ValueError, match="LR"):
        CandidateConfig(2e-5, 5)
    with pytest.raises(ValueError):
        CandidateConfig(1e-5, 2)


def test_evaluation_uses_frozen_four_way_aggregation():
    item = _evaluation()
    assert item.mean_delta_j == pytest.approx(.25)
    assert item.worst_delta_j == .1
    assert item.positive_count == 4
    assert set(item.delta_scores()) == {(11, "A"), (11, "B"), (37, "A"), (37, "B")}


def test_nonfinite_scores_and_invalid_dataset_fail_closed():
    with pytest.raises(ValueError):
        _evaluation(delta_j_seed11_foldA=float("nan"))
    with pytest.raises(ValueError, match="DATASET"):
        _evaluation(dataset="not-a-dataset")
