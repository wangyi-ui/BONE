import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b1_optuna.objective import (
    aggregate_candidate_scores, paired_delta_j, select_best_candidate, validation_j,
)
from experiments.utility_relation.p3_c1_b1_optuna.protocol import Candidate


def test_hand_computed_score_and_fold_mean():
    utility = np.array([[1., 3.], [2., 2.]])
    probabilities = np.array([[[.5, .5], [.25, .75]], [[.75, .25], [.5, .5]]])
    # targets (1, 0): (log(1)+3*log(1.5))/4 and (2*log(1.5)+0)/4
    expected = ((3 * np.log(1.5) / 4) + (np.log(1.5) / 2)) / 2
    assert validation_j(utility, probabilities, np.array([1, 0])) == pytest.approx(expected)


def test_chance_weighting_and_positive_scale_invariance():
    utility = np.array([[1., 2.], [3., 4.]])
    chance = np.full((2, 2, 2), .5)
    assert validation_j(utility, chance, np.array([0, 1])) == 0.0
    probabilities = np.array([[[.8, .2], [.6, .4]], [[.7, .3], [.5, .5]]])
    assert validation_j(utility, probabilities, np.array([0, 1])) == pytest.approx(
        validation_j(utility * 11, probabilities, np.array([0, 1]))
    )


def test_zero_mass_nonfinite_and_positive_u_nonpositive_probability_fail_closed():
    pi = np.full((1, 2, 2), .5)
    with pytest.raises(ValueError, match="ZERO_UTILITY_MASS"):
        validation_j([[0., 0.]], pi, [0])
    with pytest.raises(ValueError, match="NONFINITE"):
        validation_j([[np.nan, 1.]], pi, [0])
    bad = pi.copy(); bad[0, 0, 0] = 0
    with pytest.raises(ValueError, match="NONPOSITIVE"):
        validation_j([[1., 0.]], bad, [0])


def test_zero_u_zero_probability_does_not_create_nan():
    pi = np.array([[[0., 1.], [.5, .5]]])
    assert validation_j([[0., 2.]], pi, [0]) == 0.0


@pytest.mark.parametrize("utility, pi, targets", [
    (np.ones((1, 2)), np.ones((1, 2, 2)), np.array([2])),
    (np.ones((1, 2, 1)), np.ones((1, 2, 2)), np.array([0])),
    (np.ones((1, 2)), np.ones((2, 2, 2)), np.array([0])),
])
def test_input_shape_or_target_validation(utility, pi, targets):
    with pytest.raises(ValueError):
        validation_j(utility, pi, targets)


def _deltas(a, b, c, d):
    return {(11, "A"): a, (11, "B"): b, (37, "A"): c, (37, "B"): d}


def test_paired_delta_and_exact_four_way_aggregation():
    assert paired_delta_j(.2, .7) == pytest.approx(.5)
    aggregate = aggregate_candidate_scores(_deltas(.1, -.2, .3, .4))
    assert aggregate["mean_delta_J"] == pytest.approx(.15)
    assert aggregate["worst_delta_J"] == -.2
    assert aggregate["positive_count"] == 3
    assert len(aggregate["raw_fold_seed_scores"]) == 4
    with pytest.raises(ValueError):
        aggregate_candidate_scores({(11, "A"): .1})


def _score(mean, worst, positives):
    return {"mean_delta_J": mean, "worst_delta_J": worst, "positive_count": positives}


def test_tie_break_layer_one_mean():
    low, high = Candidate(1e-5, 1), Candidate(3e-6, 20)
    assert select_best_candidate({low: _score(.1, .9, 4), high: _score(.2, -.9, 0)}) == high


def test_tie_break_layer_two_worst_after_rounded_mean():
    left, right = Candidate(1e-5, 1), Candidate(3e-6, 20)
    assert select_best_candidate({left: _score(.1, .1, 1), right: _score(.1 + 1e-14, .2, 0)}) == right


def test_tie_break_layer_three_positive_count():
    left, right = Candidate(1e-5, 1), Candidate(3e-6, 20)
    assert select_best_candidate({left: _score(.1, .1, 2), right: _score(.1, .1, 3)}) == right


def test_tie_break_layer_four_smaller_exposure():
    low, high = Candidate(1e-5, 1), Candidate(3e-6, 20)
    assert select_best_candidate({high: _score(.1, .1, 3), low: _score(.1, .1, 3)}) == low


def test_tie_break_layer_five_smaller_lr():
    low, high = Candidate(3e-6, 5), Candidate(1e-5, 5)
    assert select_best_candidate({high: _score(.1, .1, 3), low: _score(.1, .1, 3)}) == low
