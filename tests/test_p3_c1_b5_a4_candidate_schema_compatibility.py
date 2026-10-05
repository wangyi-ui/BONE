from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import ast
import re

import pytest
import torch

from experiments.utility_relation.p3_c1_b1_optuna import formal_candidate_evaluator
from experiments.utility_relation.p3_c1_b5_optimizer_pilot import (
    candidate_schema,
    finalize_optimizer_pilot,
    optimizer_factory,
    protocol,
    training_wiring,
)


FROZEN_HASHES = {
    "experiments/utility_relation/p3_c1_b1_optuna/study_schema.py": "99768fea522ef529372964371eb6ba148c24b2dd953be177e39f681cfacf323d",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_candidate_evaluator.py": "8e27608a537d477071c0eb4372b550f9e006e6054eb1693dc774a1a0f04369cc",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_candidate_evaluator_r2.py": "ca3c09ee65f5f27105772bb0e21c7faa4929c7e031021efbeeb2a84c56af6711",
    "experiments/utility_relation/p3_c1_b1_optuna/formal_tuning_context.py": "174e231ef5aab79cbac57614a53a1e0731f78d3f097c352d48f992ccf2c5ad7a",
    "method/bone.py": "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092",
    "method/amv.py": "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc",
}

PARENT_STATUS_HASHES = {
    "bone_p3_c1_b5_optimizer_pilot_preregistered_20261004": "efb153610c24e3d82d4ba1f9f5824b1698964946c2d887ec2df804b4d851955d",
    "bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004": "debdfefa532e675c035b3b20746e708c035ca5ad2407dd0df13dcbab258e0b01",
    "bone_p3_c1_b5_a2_real_candidate_training_wiring_20261004": "4a7ecc51c1c1465d0d0770084d87db0437541a94b17cd64c5024fea2acd1d2b7",
    "bone_p3_c1_b5_a3_frozen_evaluator_api_wiring_repair_20261004": "4659a8400fad8724c391e266659897e9918845f552c180d34596dd40b6c97426",
}


def _candidate(dataset: str, optimizer: str, lr: float):
    return protocol.Candidate(dataset, optimizer, lr, protocol.EXPOSURES[dataset])


def test_required_frozen_sources_and_parent_freezes_are_unchanged():
    for relative, expected in FROZEN_HASHES.items():
        assert protocol.sha256_file(protocol.REPO_ROOT / relative) == expected
    for name, digest in PARENT_STATUS_HASHES.items():
        assert protocol.sha256_file(protocol.REPO_ROOT / "experiment_freeze" / name / "STATUS.txt") == digest


def test_a4_freeze_records_current_sources_and_parent_integrity():
    root = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a4_candidate_schema_compatibility_repair_20261004"
    assert "READY_TO_RERUN_CCV_SGD_1E3=true" in (root / "STATUS.txt").read_text()
    for line in (root / "source_sha256.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        assert protocol.sha256_file(protocol.REPO_ROOT / relative) == digest
    for line in (root / "parent_freeze_sha256.txt").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        assert protocol.sha256_file(protocol.REPO_ROOT / relative) == digest


@pytest.mark.parametrize("dataset,optimizer,lrs", [
    ("CCV", "SGD_MOMENTUM", (1e-3, 3e-3, 1e-2)),
    ("ALOI", "SGD_MOMENTUM", (1e-3, 3e-3, 1e-2)),
    ("CCV", "AdamW", (3e-6, 1e-5, 3e-5)),
    ("ALOI", "AdamW", (3e-5, 1e-4, 3e-4)),
])
def test_registered_b5_optimizer_specific_grid_passes(dataset, optimizer, lrs):
    for lr in lrs:
        config = candidate_schema.validate_b5_candidate(_candidate(dataset, optimizer, lr))
        assert config.runtime_fields == (lr, protocol.EXPOSURES[dataset])
        assert config.execution_mode == "REAL_TRAINING"


@pytest.mark.parametrize("candidate", [
    protocol.Candidate("CCV", "SGD_MOMENTUM", 1e-4, 3),
    protocol.Candidate("ALOI", "SGD_MOMENTUM", 3e-5, 1),
    protocol.Candidate("CCV", "AdamW", 1e-3, 3),
    protocol.Candidate("ALOI", "AdamW", 1e-5, 1),
    protocol.Candidate("CCV", "RMSPROP", 1e-3, 3),
])
def test_unregistered_or_cross_family_candidates_fail_closed(candidate):
    with pytest.raises(ValueError):
        candidate_schema.validate_b5_candidate(candidate)


def test_no_tolerance_expands_registered_lr_space_and_adam_is_reuse_only():
    with pytest.raises(ValueError, match="UNREGISTERED_SEMANTIC_LR"):
        candidate_schema.validate_b5_candidate(_candidate("CCV", "SGD_MOMENTUM", 0.0010000000000001))
    adam = candidate_schema.validate_b5_candidate(_candidate("CCV", "Adam", 3e-6))
    assert adam.execution_mode == "REUSE_ONLY"
    with pytest.raises(RuntimeError, match="REUSE_ONLY"):
        candidate_schema.validate_real_training_candidate(_candidate("CCV", "Adam", 3e-6))


def test_real_wiring_passes_b5_runtime_config_to_lowest_frozen_path(monkeypatch, tmp_path):
    candidate = _candidate("CCV", "SGD_MOMENTUM", 1e-3)
    context = SimpleNamespace(dataset="CCV", protocol_fingerprint="frozen")
    captured = {}

    def fake_evaluate(ctx, runtime_config, device, runner):
        captured["config"] = runtime_config
        return {
            (seed, fold): {
                "J_OURS": -0.9, "audit": {"supported_anchor_count": 1, "abstained_anchor_count": 0, "coverage": 1.0},
                "semantic_optimizer_step_count_total": 3, "semantic_steps": 3, "native_epochs": 20,
                "optimizer_audit": {}, "initial_state_sha256": "s", "carrier_model_state_file_sha256": "f",
                "carrier_model_state_sha256": "s", "loaded_initial_model_state_sha256": "s", "model_state_exact_parity": True,
                "model_loading_path": "frozen", "U_cycle_detached": True, "T_rel_detached": True,
                "anchor_posterior_detached": True, "semantic_to_native_grad_leak": False, "native_to_next_semantic_grad_leak": False,
            }
            for seed in (11, 37) for fold in ("A", "B")
        }

    monkeypatch.setattr(training_wiring.formal_candidate_evaluator_r2, "_evaluate_with_runner", fake_evaluate)
    monkeypatch.setattr(training_wiring.formal_base_reference, "load_verified_base_reference", lambda *_: SimpleNamespace(J_BASE_A=-1.0, J_BASE_B=-1.0))
    result = training_wiring.run_real_candidate(candidate, tmp_path, context_loader=lambda _: context, branch_runner=lambda *_: None)
    assert result["status"] == "COMPLETE"
    assert isinstance(captured["config"], candidate_schema.B5CandidateConfig)
    assert captured["config"].semantic_lr == 1e-3


@pytest.mark.parametrize("dataset", ["CCV", "ALOI"])
def test_sgd_schema_and_construction_smoke_without_training(dataset):
    result = training_wiring.run_model_load_smoke(dataset=dataset, optimizer_name="SGD_MOMENTUM", semantic_lr=1e-3)
    assert result["SCHEMA_COMPATIBILITY_SMOKE"] == "PASS"
    assert result["SGD_CONSTRUCTION_SMOKE"] == "PASS"
    assert result["MODEL_LOAD_SMOKE"] == "PASS"
    assert result["optimizer_step_executed"] is False
    assert result["candidate_artifact_created"] is False
    assert result["full_gt_used"] is False and result["formal_metrics_read"] is False
    assert result["optimizer"]["optimizer_class"] == "SGD"
    assert result["frozen_evaluator_runtime_branch_count"] == 4


def test_optimizer_and_native_contracts_and_finalizer_remain_frozen():
    model = torch.nn.Linear(2, 2)
    optimizer, audit = optimizer_factory.build_semantic_optimizer(model, _candidate("CCV", "SGD_MOMENTUM", 1e-3))
    assert type(optimizer).__name__ == "SGD"
    assert audit["lr"] == 1e-3 and audit["momentum"] == 0.9 and audit["weight_decay"] == 0.0 and audit["nesterov"] is False
    native = torch.optim.Adam(model.parameters(), lr=1e-4)
    assert training_wiring._native_optimizer_audit(native, 1e-4)["native_optimizer_unchanged"] is True
    with pytest.raises(RuntimeError):
        finalize_optimizer_pilot.load_complete_candidate_records(Path("/tmp") / "b5-a4-empty")


def test_b5_schema_is_runtime_only_and_has_no_gt_or_metric_imports():
    source = inspect.getsource(candidate_schema)
    imports = [
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert all("metric" not in item.lower() and "ground_truth" not in item.lower() for item in imports)
    assert not re.search(r"\\b(?:ACC|NMI|ARI)\\b", source)
    assert "CandidateConfig(" not in inspect.getsource(training_wiring.run_real_candidate)
    assert "formal_candidate_evaluator_r2._evaluate_with_runner" in inspect.getsource(training_wiring.run_real_candidate)
    assert callable(formal_candidate_evaluator._load_model)


def test_existing_valid_artifacts_are_exactly_the_three_ccv_adamw_records():
    expected = {
        "lr_3e-6.json": "5ae178ff2554d6fadaa8db1c9b67a2502be4376b906146a086b1c1243462b6f5",
        "lr_1e-5.json": "4c6e415cdf3c7d06319c48620d874747326ba133c8312362259ddc29737de259",
        "lr_3e-5.json": "26ca01c7c7bde35cd9195738ef710379c73b299edb4ed95c72f1a66ed197712b",
    }
    root = protocol.OUTPUT_ROOT / "ccv" / "adamw"
    assert {path.name for path in root.glob("*.json")} == set(expected)
    for name, digest in expected.items():
        assert protocol.sha256_file(root / name) == digest
        payload = json.loads((root / name).read_text())
        assert payload["completion_status"] == "REAL_CANDIDATE_COMPLETE"
        assert payload["branch_count"] == 4 and payload["full_gt_used"] is False and payload["formal_metrics_read"] is False
    assert not (protocol.OUTPUT_ROOT / "ccv" / "sgd_momentum" / "lr_1e-3.json").exists()
