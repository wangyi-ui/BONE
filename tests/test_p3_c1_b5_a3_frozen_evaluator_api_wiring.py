from __future__ import annotations

import inspect
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_optuna import (
    formal_candidate_evaluator,
    formal_candidate_evaluator_r2,
)
from experiments.utility_relation.p3_c1_b5_optimizer_pilot import (
    finalize_optimizer_pilot,
    optimizer_factory,
    protocol,
    training_wiring,
)


def _carrier(dataset, seed):
    return training_wiring._load_pretraining_smoke_carrier(dataset, seed)


def _candidate(name="AdamW", dataset="CCV"):
    return next(c for c in protocol.candidates(dataset) if c.optimizer_name == name)


@lru_cache(maxsize=1)
def _smoke():
    return training_wiring.run_model_load_smoke()


def test_r2_wrapper_does_not_own_load_model():
    assert not hasattr(formal_candidate_evaluator_r2, "_load_model")


def test_frozen_evaluator_owns_load_model():
    assert callable(formal_candidate_evaluator._load_model)


def test_r2_is_a_delegating_wrapper():
    source = inspect.getsource(formal_candidate_evaluator_r2)
    assert "frozen_evaluator.evaluate_candidate" in source
    assert "contains no candidate-evaluation implementation" in source


def test_b5_calls_actual_frozen_loader():
    source = inspect.getsource(training_wiring._load_frozen_model_with_parity)
    assert "formal_candidate_evaluator._load_model" in source
    assert "formal_candidate_evaluator_r2._load_model" not in source


def test_frozen_loader_is_c0_new_model_then_c0_load_state_strict():
    source = inspect.getsource(formal_candidate_evaluator._load_model)
    assert "c0._new_model" in source
    assert "c0._load_state" in source
    assert "strict=True" in source


@pytest.mark.parametrize("dataset,seed", [("CCV", 11), ("CCV", 37), ("ALOI", 11), ("ALOI", 37)])
def test_real_frozen_carrier_constructs_with_exact_state_parity(dataset, seed):
    carrier = _carrier(dataset, seed)
    _model, audit = training_wiring._load_frozen_model_with_parity(
        carrier, torch.device("cpu")
    )
    assert audit["model_state_exact_parity"] is True
    assert audit["carrier_model_state_sha256"] == audit["loaded_initial_model_state_sha256"]


def test_state_mismatch_fails_closed(monkeypatch, tmp_path):
    expected = torch.nn.Linear(2, 2)
    path = tmp_path / "carrier_model_state.pt"
    torch.save(expected.state_dict(), path)
    carrier = SimpleNamespace(state_path=path, state_file_sha256="file")
    observed = torch.nn.Linear(2, 2)
    with torch.no_grad():
        observed.weight.add_(1)
    monkeypatch.setattr(formal_candidate_evaluator, "_load_model", lambda *_: observed)
    with pytest.raises(RuntimeError, match="STATE_TENSOR_MISMATCH"):
        training_wiring._load_frozen_model_with_parity(carrier, torch.device("cpu"))


def test_real_smoke_passes_without_training():
    result = _smoke()
    assert result["MODEL_LOAD_SMOKE"] == "PASS"
    assert result["optimizer_step_executed"] is False
    assert result["candidate_evaluation_executed"] is False


def test_real_smoke_uses_ccv_seed11_fold_a():
    result = _smoke()
    assert (result["dataset"], result["dev_seed"], result["fold"]) == ("CCV", 11, "A")


def test_real_smoke_exposes_frozen_u_and_t_detached():
    result = _smoke()
    assert result["U_cycle_available"] and result["T_rel_available"]
    assert result["U_cycle_detached"] and result["T_rel_detached"]


def test_real_smoke_gt_firewall_closed():
    result = _smoke()
    assert result["full_gt_used"] is False
    assert result["formal_metrics_read"] is False


def test_real_smoke_creates_no_candidate_artifact():
    result = _smoke()
    assert result["candidate_artifact_created"] is False


def test_adamw_injection_exact():
    model = torch.nn.Linear(2, 2)
    _optimizer, audit = optimizer_factory.build_semantic_optimizer(model, _candidate("AdamW"))
    assert audit["optimizer_class"] == "AdamW"
    assert audit["weight_decay"] == 1e-4
    assert audit["betas"] == [0.9, 0.999]


def test_sgd_injection_exact():
    model = torch.nn.Linear(2, 2)
    _optimizer, audit = optimizer_factory.build_semantic_optimizer(model, _candidate("SGD_MOMENTUM"))
    assert audit["optimizer_class"] == "SGD"
    assert audit["momentum"] == 0.9 and audit["nesterov"] is False


def test_native_optimizer_contract_unchanged():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    audit = training_wiring._native_optimizer_audit(optimizer, 1e-4)
    assert audit["native_optimizer_unchanged"] is True


def test_anchor_posterior_remains_detached():
    assert "posterior[batch:].detach()" in inspect.getsource(c0._semantic_epoch)


def test_j_definition_and_coverage_abstention_unchanged():
    from experiments.utility_relation.p3_c1_b1_optuna import objective_a1
    source = inspect.getsource(objective_a1.validation_j_with_audit)
    assert "fold_denominator" in source and "abstained_anchor_count" in source


def test_method_bone_unchanged():
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/bone.py") == "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"


def test_method_amv_unchanged():
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/amv.py") == "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def test_b5_preregistration_freeze_unchanged():
    assert protocol.sha256_file(protocol.PREREGISTERED_FREEZE / "STATUS.txt") == "efb153610c24e3d82d4ba1f9f5824b1698964946c2d887ec2df804b4d851955d"


def test_b5_a1_freeze_unchanged():
    path = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004/STATUS.txt"
    assert protocol.sha256_file(path) == "debdfefa532e675c035b3b20746e708c035ca5ad2407dd0df13dcbab258e0b01"


def test_b5_a2_freeze_unchanged():
    path = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a2_real_candidate_training_wiring_20261004/STATUS.txt"
    assert protocol.sha256_file(path) == "4a7ecc51c1c1465d0d0770084d87db0437541a94b17cd64c5024fea2acd1d2b7"


def test_failed_invocation_left_zero_candidate_artifacts():
    root = protocol.OUTPUT_ROOT
    # A4 follows three completed CCV/AdamW candidates.  The failed SGD
    # invocation itself still produced no candidate artifact.
    assert not (root / "ccv" / "sgd_momentum" / "lr_1e-3.json").exists()


def test_failed_invocation_classification_is_non_scientific():
    status = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a3_frozen_evaluator_api_wiring_repair_20261004/STATUS.txt"
    assert "FAILED_INVOCATION_CLASSIFICATION=ENGINEERING_WIRING_FAILURE_BEFORE_CANDIDATE_COMPLETION" in status.read_text()


def test_finalizer_still_requires_exact_18(tmp_path):
    with pytest.raises(RuntimeError):
        finalize_optimizer_pilot.load_complete_candidate_records(tmp_path)


def test_candidate_schema_rejects_state_parity_failure(tmp_path):
    candidate = _candidate()
    branches = {
        key: {
            "J_BASE": 0.0, "J_OURS": 0.0, "DeltaJ": 0.0,
            "semantic_optimizer_step_count_total": 1,
            "full_gt_used": False, "formal_metrics_read": False,
            "carrier_model_state_sha256": "a",
            "loaded_initial_model_state_sha256": "b",
            "model_state_exact_parity": False,
        }
        for key in training_wiring.BRANCH_KEYS
    }
    payload = {
        "status": "COMPLETE", "completion_status": "REAL_CANDIDATE_COMPLETE",
        "dataset": candidate.dataset, "optimizer_name": candidate.optimizer_name,
        "semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
        "branches": branches, "mean_delta_J": 0.0, "worst_delta_J": 0.0,
        "positive_count": 0, "optimizer_audit": {}, "full_gt_used": False,
        "formal_metrics_read": False, "real_training_started": True,
        "real_training_completed": True, "branch_count": 4, "GT_LOADED": False,
    }
    path = training_wiring.candidate_artifact_path(tmp_path, candidate)
    training_wiring._write_complete_artifact(path, payload)
    with pytest.raises(RuntimeError, match="INITIAL_MODEL_PARITY_INVALID"):
        training_wiring.verify_candidate_artifact(path, candidate)


def test_frozen_sources_are_not_in_b5_package():
    b5_root = Path(training_wiring.__file__).parent
    assert Path(formal_candidate_evaluator.__file__).parent != b5_root
    assert Path(formal_candidate_evaluator_r2.__file__).parent != b5_root


def test_smoke_json_contract_is_serializable():
    json.dumps(_smoke(), sort_keys=True)
