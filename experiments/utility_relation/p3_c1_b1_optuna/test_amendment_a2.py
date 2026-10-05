import numpy as np
import pytest

from . import amendment_a2


def _audit(targets, masses):
    return amendment_a2.validate_coverage_aware_feasibility(
        targets,
        np.arange(len(targets)),
        masses,
        expected_class_ids=range(max(targets) + 1),
    )


def test_positive_zero_pair_for_one_class_passes():
    audit = _audit([0, 0], [1.0, 0.0])
    assert audit["supported_anchor_count"] == 1
    assert audit["abstained_class_count"] == 0
    assert audit["A2_FEASIBILITY"] == "PASS"


def test_zero_zero_class_passes_when_another_class_has_support():
    audit = _audit([0, 0, 1, 1], [0.0, 0.0, 0.5, 0.0])
    assert audit["supported_anchor_count"] == 1
    assert audit["abstained_class_count"] == 1
    assert audit["per_class"][0]["utility_abstained_class"] is True
    assert audit["coverage_threshold_added"] is False


def test_many_zero_evidence_classes_pass_with_one_global_supported_anchor():
    K = 50
    targets = np.repeat(np.arange(K), 2)
    masses = np.zeros(2 * K)
    masses[-1] = 0.01
    audit = _audit(targets, masses)
    assert audit["supported_anchor_count"] == 1
    assert audit["abstained_class_count"] == K - 1
    assert audit["supported_class_coverage"] == pytest.approx(1 / K)


def test_all_zero_masses_fail_closed_with_exact_error():
    with pytest.raises(
        ValueError, match="^NO_UTILITY_SUPPORTED_VALIDATION_EVIDENCE$"
    ):
        _audit([0, 0, 1, 1], [0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "masses,error",
    [
        ([1.0, -1.0], "NEGATIVE_UTILITY_MASS"),
        ([1.0, np.nan], "NONFINITE_UTILITY_MASS"),
        ([1.0, np.inf], "NONFINITE_UTILITY_MASS"),
    ],
)
def test_invalid_utility_masses_fail(masses, error):
    with pytest.raises(ValueError, match=error):
        _audit([0, 0], masses)


@pytest.mark.parametrize(
    "targets,expected,error",
    [
        ([0, 0, 0], range(1), "CLASS_MUST_HAVE_EXACTLY_TWO_SPARSE_ANCHORS"),
        ([0, 0], range(2), "MISSING_OR_UNEXPECTED_CLASS"),
        ([0, 0, 2, 2], range(2), "MISSING_OR_UNEXPECTED_CLASS"),
    ],
)
def test_invalid_sparse_class_structure_fails(targets, expected, error):
    with pytest.raises(ValueError, match=error):
        amendment_a2.validate_coverage_aware_feasibility(
            targets, np.arange(len(targets)), np.ones(len(targets)), expected
        )


def test_cross_seed_audit_records_persistent_union_and_seed_specific_classes():
    seed11 = _audit([0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 1, 0])
    seed37 = _audit([0, 0, 1, 1, 2, 2], [0, 0, 1, 0, 0, 0])
    summary = amendment_a2.aggregate_dev_seed_coverage_audits(
        {11: seed11, 37: seed37}
    )
    assert summary["persistent_abstained_classes"] == [0]
    assert summary["union_abstained_classes"] == [0, 1, 2]
    assert summary["seed_specific_abstained_classes"] == {
        "11": [1], "37": [2]
    }
    assert summary["coverage_audit_only"] is True

