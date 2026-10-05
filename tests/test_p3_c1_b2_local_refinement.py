from __future__ import annotations

from experiments.utility_relation.p3_c1_b2_local_refinement import protocol


def test_exact_datasets_grids_and_total():
    protocol.validate_protocol()
    assert protocol.DATASETS == ("CCV", "MSRC", "ALOI")
    assert [len(protocol.candidates(dataset)) for dataset in protocol.DATASETS] == [25, 18, 30]
    assert sum(len(protocol.candidates(dataset)) for dataset in protocol.DATASETS) == 73


def test_b1_points_and_aloi_reference_are_in_local_spaces():
    for dataset, config in protocol.B1_SELECTED.items():
        assert protocol.Candidate(*config) in protocol.candidates(dataset)
    assert protocol.Candidate(1e-4, 20) in protocol.candidates("ALOI")


def test_exact_rounded_lexicographic_selection():
    early_small_lr = {"candidate": protocol.Candidate(1e-5, 1), "mean_delta_J": 0.1, "worst_delta_J": 0.01, "positive_count": 2}
    later_large_lr = {"candidate": protocol.Candidate(2e-5, 1), "mean_delta_J": 0.1000000000004, "worst_delta_J": 0.01, "positive_count": 2}
    assert protocol.select_best([later_large_lr, early_small_lr]) is early_small_lr
    stronger_worst = {"candidate": protocol.Candidate(3e-5, 5), "mean_delta_J": 0.1, "worst_delta_J": 0.02, "positive_count": 1}
    assert protocol.select_best([early_small_lr, stronger_worst]) is stronger_worst
