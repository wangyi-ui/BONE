from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.utility_relation.p3_c1_b5_optimizer_pilot import (
    finalize_optimizer_pilot,
    protocol,
    run_candidate,
    training_wiring,
)


def _candidate(name="AdamW", dataset="CCV", lr=3e-6):
    return next(item for item in protocol.candidates(dataset) if item.optimizer_name == name and item.semantic_lr == lr)


def _payload(candidate, started=True, completed=True):
    branches = {}
    for key in training_wiring.BRANCH_KEYS:
        branches[key] = {
            "J_BASE": -1.0,
            "J_OURS": -0.9,
            "DeltaJ": 0.1,
            "semantic_optimizer_step_count_total": 1,
            "full_gt_used": False,
            "formal_metrics_read": False,
            "carrier_model_state_sha256": "state",
            "loaded_initial_model_state_sha256": "state",
            "model_state_exact_parity": True,
        }
    return {
        "status": "COMPLETE",
        "completion_status": "REAL_CANDIDATE_COMPLETE",
        "dataset": candidate.dataset,
        "optimizer_name": candidate.optimizer_name,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "branches": branches,
        "mean_delta_J": 0.1,
        "worst_delta_J": 0.1,
        "positive_count": 4,
        "optimizer_audit": {},
        "full_gt_used": False,
        "formal_metrics_read": False,
        "GT_LOADED": False,
        "real_training_started": started,
        "real_training_completed": completed,
        "branch_count": 4,
    }


def test_preflight_only_is_not_a_completed_candidate(tmp_path):
    with pytest.raises(RuntimeError, match="REQUIRES_SEPARATE_AUTHORIZATION"):
        run_candidate.run_candidate(_candidate(), output_root=tmp_path, execute=True, authorize_real_candidate=False)
    assert not list(tmp_path.rglob("*.json"))


def test_authorized_candidate_enters_training_backend_without_training_in_unit_test(monkeypatch, tmp_path):
    calls = []
    expected = _payload(_candidate())

    def fake_run(candidate, output_root):
        calls.append(candidate.key())
        return {"status": "COMPLETE", "completion_status": "REAL_CANDIDATE_COMPLETE", "payload": expected}

    monkeypatch.setattr(run_candidate, "_verify_evaluator_api_wiring_repair", lambda: {"status": "PASS"})
    monkeypatch.setattr(training_wiring, "run_real_candidate", fake_run)
    result = run_candidate.run_candidate(_candidate(), output_root=tmp_path, execute=True, authorize_real_candidate=True)
    assert calls == [_candidate().key()]
    assert result["completion_status"] == "REAL_CANDIDATE_COMPLETE"
    assert result["real_training_started"] is True


def test_candidate_artifact_schema_hash_and_four_branches(tmp_path):
    candidate = _candidate()
    path = training_wiring.candidate_artifact_path(tmp_path, candidate)
    training_wiring._write_complete_artifact(path, _payload(candidate))
    observed = training_wiring.verify_candidate_artifact(path, candidate)
    assert set(observed["branches"]) == {"11A", "11B", "37A", "37B"}
    assert observed["branch_count"] == 4
    assert observed["real_training_completed"] is True
    assert observed["full_gt_used"] is False and observed["formal_metrics_read"] is False


def test_real_wiring_persists_only_after_all_four_frozen_branches(monkeypatch, tmp_path):
    candidate = _candidate()
    context = SimpleNamespace(dataset="CCV", protocol_fingerprint="frozen")
    branch_calls = []

    def fake_evaluate(ctx, candidate_config, device, runner):
        assert ctx is context and device == "cuda:0"
        results = {}
        for seed in (11, 37):
            for fold in ("A", "B"):
                branch_calls.append(f"{seed}{fold}")
                results[(seed, fold)] = {
                    "J_OURS": -0.9,
                    "audit": {"supported_anchor_count": 1, "abstained_anchor_count": 0, "coverage": 1.0},
                    "semantic_optimizer_step_count_total": 3,
                    "semantic_steps": 3,
                    "native_epochs": 20,
                    "optimizer_audit": {"native": {"native_optimizer_unchanged": True}},
                    "initial_state_sha256": "state",
                    "carrier_model_state_file_sha256": "file-state",
                    "carrier_model_state_sha256": "state",
                    "loaded_initial_model_state_sha256": "state",
                    "model_state_exact_parity": True,
                    "model_loading_path": "frozen",
                    "U_cycle_detached": True, "T_rel_detached": True,
                    "anchor_posterior_detached": True,
                    "semantic_to_native_grad_leak": False,
                    "native_to_next_semantic_grad_leak": False,
                }
        return results

    monkeypatch.setattr(training_wiring.formal_candidate_evaluator_r2, "_evaluate_with_runner", fake_evaluate)
    monkeypatch.setattr(
        training_wiring.formal_base_reference,
        "load_verified_base_reference",
        lambda ctx, seed: SimpleNamespace(J_BASE_A=-1.0, J_BASE_B=-1.0),
    )
    result = training_wiring.run_real_candidate(
        candidate, output_root=tmp_path, context_loader=lambda dataset: context,
        branch_runner=lambda *args: None,
    )
    assert result["status"] == "COMPLETE"
    assert branch_calls == ["11A", "11B", "37A", "37B"]
    assert training_wiring.verify_candidate_artifact(result["artifact"], candidate)["branch_count"] == 4


def test_incomplete_training_cannot_verify_as_candidate(tmp_path):
    candidate = _candidate()
    path = training_wiring.candidate_artifact_path(tmp_path, candidate)
    training_wiring._write_complete_artifact(path, _payload(candidate, started=False, completed=False))
    with pytest.raises(RuntimeError, match="TRAINING_NOT_COMPLETE"):
        training_wiring.verify_candidate_artifact(path, candidate)


def test_optimizer_audit_and_frozen_j_abstention_contracts():
    assert training_wiring.canonical_lr_slug(3e-6) == "3e-6"
    assert training_wiring.canonical_lr_slug(1e-4) == "1e-4"
    # Frozen coverage-aware J keeps abstained anchors in the denominator.
    from experiments.utility_relation.p3_c1_b1_optuna.objective_a1 import validation_j_with_audit
    import numpy as np
    score, audit = validation_j_with_audit(
        np.asarray([[0.0], [1.0]]),
        np.asarray([[[1.0, 0.0]], [[0.75, 0.25]]]),
        np.asarray([0, 0]),
    )
    assert audit["abstained_anchor_count"] == 1
    assert audit["fold_denominator"] == 2
    assert score == pytest.approx(float(np.log(1.5) / 2.0))


def test_finalizer_fails_closed_before_exact_18(tmp_path):
    with pytest.raises(RuntimeError):
        finalize_optimizer_pilot.load_complete_candidate_records(tmp_path)


def test_parent_and_method_freezes_unchanged():
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/bone.py") == "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
    assert protocol.sha256_file(protocol.REPO_ROOT / "method/amv.py") == "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"
    assert protocol.sha256_file(protocol.PREREGISTERED_FREEZE / "STATUS.txt") == "efb153610c24e3d82d4ba1f9f5824b1698964946c2d887ec2df804b4d851955d"
    a1 = protocol.REPO_ROOT / "experiment_freeze/bone_p3_c1_b5_a1_real_candidate_execution_authorized_20261004/STATUS.txt"
    assert "REAL_CANDIDATES_ALREADY_RUN=false" in a1.read_text()
