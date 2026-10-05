from pathlib import Path

from experiments.utility_relation.p3_c1_b1_optuna.diagnose_msrc_zero_utility import diagnose


CARRIER = Path(
    "outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke/"
    "msrc/dev_seed11/carrier/carrier_pre_gt.npz"
)


def test_real_carrier_zero_utility_diagnosis_without_gt():
    result = diagnose(CARRIER)
    assert result["u_formula_parity"] is True
    assert result["u_max_abs_diff"] <= 1e-7
    assert result["sample_id_unique_pass"] is True
    assert result["action_complement_pass"] is True
    assert result["zero_mass_sample_count"] == 4
    assert result["zero_mass_raw_ids"] == [15, 47, 114, 175]
    assert result["zero_mass_labeled_count"] == 1
    assert result["zero_mass_labeled_records"] == [
        {"raw_id": 175, "class": 5, "array_position": 175}
    ]
    assert result["raw175"]["utility_mass"] == 0.0
    assert result["raw175"]["closure_count"] == 0
    assert result["class_feasibility"]["all_classes_feasible"] is True
    assert result["full_gt_loaded"] is False
