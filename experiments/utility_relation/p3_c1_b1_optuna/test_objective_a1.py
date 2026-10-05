import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b1_optuna import objective as original
from experiments.utility_relation.p3_c1_b1_optuna import objective_a1


def test_all_supported_exact_parity_with_original_objective():
    utility = np.array([[1.0, 3.0], [2.0, 2.0]])
    pi = np.array([[[.5, .5], [.25, .75]], [[.75, .25], [.5, .5]]])
    targets = np.array([1, 0])
    assert objective_a1.validation_j(utility, pi, targets) == original.validation_j(utility, pi, targets)


def test_zero_mass_anchor_is_exact_zero_and_kept_in_k_denominator():
    utility = np.array([[0.0, 0.0], [1.0, 1.0]])
    pi = np.array([[[.5, .5], [.5, .5]], [[.75, .25], [.75, .25]]])
    score, audit = objective_a1.validation_j_with_audit(utility, pi, np.array([0, 0]))
    supported_score = np.log(2 * .75)
    assert audit["anchor_scores"][0] == 0.0
    assert score == pytest.approx(supported_score / 2)
    assert audit["fold_denominator"] == 2
    assert audit["supported_anchor_count"] == 1
    assert audit["abstained_anchor_count"] == 1
    assert audit["coverage"] == .5


def test_zero_mass_score_is_independent_of_any_valid_pi():
    left = np.array([[[.99, .01], [.01, .99]]])
    right = np.array([[[.20, .80], [.80, .20]]])
    assert objective_a1.validation_j([[0.0, 0.0]], left, [0]) == 0.0
    assert objective_a1.validation_j([[0.0, 0.0]], right, [0]) == 0.0


def test_zero_u_action_with_zero_probability_does_not_create_nan():
    pi = np.array([[[0.0, 1.0], [.5, .5]]])
    assert objective_a1.validation_j([[0.0, 2.0]], pi, [0]) == 0.0


def test_positive_u_action_with_nonpositive_true_probability_fails_closed():
    pi = np.array([[[0.0, 1.0], [.5, .5]]])
    with pytest.raises(ValueError, match="NONPOSITIVE_TRUE_RELATION_PROBABILITY"):
        objective_a1.validation_j([[1.0, 0.0]], pi, [0])


def test_positive_scaling_invariance_for_supported_samples():
    utility = np.array([[1.0, 2.0], [3.0, 4.0]])
    pi = np.array([[[.8, .2], [.6, .4]], [[.7, .3], [.5, .5]]])
    targets = np.array([0, 1])
    assert objective_a1.validation_j(utility, pi, targets) == pytest.approx(
        objective_a1.validation_j(utility * 11, pi, targets)
    )


@pytest.mark.parametrize("utility,pi", [
    (np.array([[np.nan, 0.0]]), np.full((1, 2, 2), .5)),
    (np.array([[1.0, 0.0]]), np.array([[[np.inf, 0.0], [.5, .5]]])),
])
def test_nonfinite_fails_closed(utility, pi):
    with pytest.raises(ValueError, match="NONFINITE"):
        objective_a1.validation_j(utility, pi, [0])


def test_paired_base_and_ours_abstain_identically():
    utility = np.array([[0.0, 0.0], [1.0, 1.0]])
    base_pi = np.array([[[.9, .1], [.2, .8]], [[.6, .4], [.6, .4]]])
    ours_pi = np.array([[[.1, .9], [.8, .2]], [[.7, .3], [.7, .3]]])
    base, base_audit = objective_a1.validation_j_with_audit(utility, base_pi, [0, 0])
    ours, ours_audit = objective_a1.validation_j_with_audit(utility, ours_pi, [0, 0])
    assert base_audit["anchor_scores"][0] == ours_audit["anchor_scores"][0] == 0.0
    assert objective_a1.paired_delta_j(base, ours) == pytest.approx(ours - base)
