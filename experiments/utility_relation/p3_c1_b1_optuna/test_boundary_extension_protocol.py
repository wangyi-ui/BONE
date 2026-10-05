from pathlib import Path

import pytest

from . import protocol
from .boundary_extension_protocol import (
    A2_FREEZE_RELATIVE,
    EXTENSION_LR,
    REQUIRED_A2_STATUS,
    REQUIRED_TRIGGER_STATUS,
    SECOND_BOUNDARY_EXTENSION_ALLOWED,
    SEMANTIC_EXPOSURES,
    TRIGGER_FREEZE_RELATIVE,
    ExtensionCandidateConfig,
    extension_candidates,
    verify_boundary_extension_preconditions,
)


def _write_status(root: Path, relative: Path, values: dict):
    path = root / relative / "STATUS.txt"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _valid_preconditions(root):
    _write_status(root, TRIGGER_FREEZE_RELATIVE, REQUIRED_TRIGGER_STATUS)
    _write_status(root, A2_FREEZE_RELATIVE, REQUIRED_A2_STATUS)


def test_extension_grid_is_exactly_the_five_preregistered_configs():
    candidates = extension_candidates()
    assert {(item.semantic_lr, item.semantic_exposure) for item in candidates} == {
        (1e-3, 1), (1e-3, 3), (1e-3, 5), (1e-3, 10), (1e-3, 20),
    }
    assert len(candidates) == 5


def test_all_eight_datasets_share_one_identical_grid():
    grids = {
        dataset: tuple(extension_candidates()) for dataset in protocol.DATASETS
    }
    assert len(grids) == 8
    assert len(set(grids.values())) == 1


def test_no_lower_lr_no_lr_above_upper_extension_and_no_second_stage():
    assert EXTENSION_LR == (1e-3,)
    assert 1e-6 not in EXTENSION_LR
    assert max(EXTENSION_LR) == 1e-3
    assert SEMANTIC_EXPOSURES == (1, 3, 5, 10, 20)
    assert SECOND_BOUNDARY_EXTENSION_ALLOWED is False


@pytest.mark.parametrize("lr", [1e-6, 3e-4, 3e-3, 1e-2])
def test_unregistered_extension_lr_is_rejected(lr):
    with pytest.raises(ValueError, match="UNREGISTERED_EXTENSION_LR"):
        ExtensionCandidateConfig(lr, 1)


def test_boundary_and_a2_freeze_preconditions_pass_exactly(tmp_path):
    _valid_preconditions(tmp_path)
    audit = verify_boundary_extension_preconditions(tmp_path)
    assert audit["lower_boundary_triggered"] is False
    assert audit["upper_boundary_triggered"] is True
    assert audit["trigger_datasets"] == ["BDGP"]
    assert audit["total_extension_candidates"] == 40
    assert audit["a2_feasibility_active"] is True


@pytest.mark.parametrize(
    "key,bad_value,error",
    [
        ("LOWER_BOUNDARY_HIT_DATASETS", "HandWrite", "BOUNDARY_TRIGGER_PRECONDITION_FAILED"),
        ("UPPER_BOUNDARY_HIT_DATASETS", "NONE", "BOUNDARY_TRIGGER_PRECONDITION_FAILED"),
        ("FINAL_HYPERPARAMETERS_FROZEN", "true", "BOUNDARY_TRIGGER_PRECONDITION_FAILED"),
    ],
)
def test_trigger_mismatch_fails_closed(tmp_path, key, bad_value, error):
    trigger = dict(REQUIRED_TRIGGER_STATUS)
    trigger[key] = bad_value
    _write_status(tmp_path, TRIGGER_FREEZE_RELATIVE, trigger)
    _write_status(tmp_path, A2_FREEZE_RELATIVE, REQUIRED_A2_STATUS)
    with pytest.raises(RuntimeError, match=error):
        verify_boundary_extension_preconditions(tmp_path)


def test_missing_a2_freeze_fails_closed(tmp_path):
    _write_status(tmp_path, TRIGGER_FREEZE_RELATIVE, REQUIRED_TRIGGER_STATUS)
    with pytest.raises(FileNotFoundError, match="MISSING_BOUNDARY_PRECONDITION"):
        verify_boundary_extension_preconditions(tmp_path)

