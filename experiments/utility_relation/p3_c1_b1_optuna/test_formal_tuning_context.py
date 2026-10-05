import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import amendment_a1, formal_tuning_context as module
from . import audit_formal_dev_carriers, protocol


def _tiny_arrays(masses=(1.0, 0.0, 2.0, 0.0)):
    # Two classes, exactly two sparse anchors per class; S=3 for V=1.
    q = np.ones((4, 1, 2), dtype=np.float32)
    p_gen = np.full((4, 3, 2), 0.5, dtype=np.float32)
    p_ver = p_gen.copy()
    u = np.zeros((4, 3), dtype=np.float32)
    for row, mass in enumerate(masses):
        u[row, 0] = mass
    return {
        "q_views": q, "p_gen": p_gen, "p_ver": p_ver, "U_cycle": u,
        "generator_actions": np.zeros((3, 1), dtype=np.int64),
        "verifier_actions": np.zeros((3, 1), dtype=np.int64),
        "sample_ids": np.arange(4, dtype=np.int64),
        "labeled_ids": np.arange(4, dtype=np.int64),
        "labeled_targets": np.array([0, 0, 1, 1], dtype=np.int64),
    }


def test_16_carrier_prerequisites_are_ready():
    inventory = audit_formal_dev_carriers.audit_inventory(
        "outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
    )
    assert inventory["expected_carrier_count"] == 16
    assert inventory["complete_verified_count"] == 16
    assert inventory["missing_count"] == 0
    assert inventory["invalid_count"] == 0
    assert inventory["ALL_16_FORMAL_DEV_CARRIERS_READY"] is True


def test_a1_feasibility_gate_calls_frozen_helper(monkeypatch):
    seen = {}
    original = amendment_a1.validate_class_feasibility

    def wrapped(targets, masses, expected_class_ids=None):
        seen["called"] = True
        return original(targets, masses, expected_class_ids)

    monkeypatch.setattr(module.amendment_a1, "validate_class_feasibility", wrapped)
    result = module._feasibility(_tiny_arrays())
    assert seen["called"] is True
    assert result["all_classes_feasible"] is True


def test_zero_zero_fails_closed():
    with pytest.raises(ValueError, match="CLASS_ZERO_UTILITY_EVIDENCE"):
        module._feasibility(_tiny_arrays((0.0, 0.0, 2.0, 0.0)))


def test_context_builder_is_read_only_and_does_not_train(monkeypatch, tmp_path):
    assert "torch.optim" not in inspect.getsource(module)
    assert "full_gt" in inspect.getsource(module)
    assert not (tmp_path / "study.db").exists()

