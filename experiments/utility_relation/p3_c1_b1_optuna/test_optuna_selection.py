from experiments.utility_relation.p3_c1_b1_optuna.audit import boundary_information
from experiments.utility_relation.p3_c1_b1_optuna.objective import select_best_candidate
from experiments.utility_relation.p3_c1_b1_optuna.protocol import Candidate, DATASETS


def _score(mean, worst, positive):
    return {"mean_delta_J": mean, "worst_delta_J": worst, "positive_count": positive}


def test_preregistered_selection_ties_are_not_primary_only():
    a, b = Candidate(1e-5, 20), Candidate(3e-6, 1)
    assert select_best_candidate({a: _score(.1, .2, 1), b: _score(.1, .3, 0)}) == b
    assert select_best_candidate({a: _score(.1, .2, 1), b: _score(.1, .2, 2)}) == b
    assert select_best_candidate({a: _score(.1, .2, 2), b: _score(.1, .2, 2)}) == b
    c, d = Candidate(1e-5, 5), Candidate(3e-6, 5)
    assert select_best_candidate({c: _score(.1, .2, 2), d: _score(.1, .2, 2)}) == d


def test_boundary_detection_delegates_to_frozen_protocol():
    base = {dataset: 1e-5 for dataset in DATASETS}
    assert boundary_information(base, 5)["lr_extension"] == []
    assert boundary_information(dict(base, HandWrite=3e-6), 5)["lr_extension"] == [1e-6]
    assert boundary_information(dict(base, HandWrite=3e-4), 5)["lr_extension"] == [1e-3]
    assert boundary_information(dict(base, HandWrite=3e-6, BDGP=3e-4), 20) == {
        "lr_extension": [1e-6, 1e-3], "SEMANTIC_EXPOSURE_UPPER_BOUND_HIT": True,
    }
