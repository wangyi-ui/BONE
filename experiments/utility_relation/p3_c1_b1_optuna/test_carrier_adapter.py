from pathlib import Path

import pytest

from experiments.utility_relation.p3_c1_b1_optuna import protocol
from experiments.utility_relation.p3_c1_b1_optuna.carrier_adapter import (
    expected_carrier_dir, validate_carrier_request,
)
from experiments.utility_relation.p3_c1_b1_optuna.parent_integrity import sha256_file


ROOT = Path(__file__).resolve().parents[3]


def test_carrier_contract_uses_frozen_epoch_count_and_namespace():
    assert protocol.CARRIER_EPOCHS == 100
    assert expected_carrier_dir(
        "outputs/bone_p3/p3_c1_b1_optuna/engineering_smoke", "Caltech-5V", 11
    ) == Path("outputs/bone_p3/p3_c1_b1_optuna/engineering_smoke/caltech_5v/dev_seed11/carrier")


def test_dev_seed_is_validated_without_formal_substitution():
    validate_carrier_request("Caltech-5V", 11, "cuda:0")
    with pytest.raises(ValueError):
        validate_carrier_request("Caltech-5V", 20, "cuda:0")
    with pytest.raises(ValueError):
        validate_carrier_request("Caltech-5V", 11, "cpu")


def test_contract_validation_never_mutates_parent_or_p3_c0_output():
    parent = ROOT / "method/bone.py"
    before = sha256_file(parent)
    validate_carrier_request("Caltech-5V", 11, "cuda:0")
    assert sha256_file(parent) == before
    assert "p3_c0_formal" not in str(expected_carrier_dir(
        "outputs/bone_p3/p3_c1_b1_optuna/engineering_smoke", "Caltech-5V", 11
    ))


def test_adapter_source_has_no_full_gt_loader():
    source = (Path(__file__).with_name("carrier_adapter.py")).read_text(encoding="utf-8")
    assert "loadmat" not in source
    assert "variable_names" not in source
    assert "FULL_GT" not in source
