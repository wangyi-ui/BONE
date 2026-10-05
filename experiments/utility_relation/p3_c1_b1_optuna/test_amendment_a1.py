import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b1_optuna import amendment_a1 as a1
from experiments.utility_relation.p3_c1_b1_optuna import protocol


def test_metadata_inherits_exactly_all_eight_frozen_datasets():
    assert a1.AMENDMENT_ID == "P3-C1-B1-A1"
    assert a1.APPLIES_TO_ALL_DATASETS is True
    assert a1.AMENDED_DATASETS is protocol.DATASETS
    assert a1.AMENDED_DATASETS == (
        "HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS", "ALOI", "CCV",
    )


def test_msrc_has_no_special_search_space_or_policy_parameter():
    assert len(protocol.main_search_candidates()) == 25
    assert a1.ZERO_UTILITY_VALIDATION_POLICY == "ABSTAIN_ZERO_CONTRIBUTION"
    assert a1.CLASS_FEASIBILITY_POLICY == "AT_LEAST_ONE_SUPPORTED_ANCHOR_PER_CLASS_PER_DEV_SEED"
    public_names = set(vars(a1))
    assert not any("EPSILON" in name or "THRESHOLD" in name for name in public_names)


@pytest.mark.parametrize("masses", [[2.0, 1.0], [2.0, 0.0], [0.0, 1.0]])
def test_one_class_feasible_supported_patterns(masses):
    audit = a1.validate_class_feasibility([0, 0], masses, expected_class_ids=[0])
    assert audit["all_classes_feasible"] is True


def test_one_class_zero_zero_fails_closed():
    with pytest.raises(ValueError, match="CLASS_ZERO_UTILITY_EVIDENCE"):
        a1.validate_class_feasibility([0, 0], [0.0, 0.0], expected_class_ids=[0])


def test_any_infeasible_class_fails_multiclass_gate():
    with pytest.raises(ValueError, match="CLASS_ZERO_UTILITY_EVIDENCE"):
        a1.validate_class_feasibility(
            [0, 0, 1, 1, 2, 2], [1.0, 0.0, 0.0, 0.0, 2.0, 3.0], expected_class_ids=range(3)
        )


def test_feasibility_rejects_nonfinite_negative_and_wrong_anchor_count():
    with pytest.raises(ValueError, match="NONFINITE"):
        a1.validate_class_feasibility([0, 0], [1.0, np.nan])
    with pytest.raises(ValueError, match="NEGATIVE"):
        a1.validate_class_feasibility([0, 0], [1.0, -1.0])
    with pytest.raises(ValueError, match="EXACTLY_TWO"):
        a1.validate_class_feasibility([0], [1.0])
