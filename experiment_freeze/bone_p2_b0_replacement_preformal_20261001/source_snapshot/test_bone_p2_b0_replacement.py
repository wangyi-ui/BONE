"""P2-B0 replacement diagnostics; no formal REG_ONLY or REPLACE_TRUE_U training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import method.bone as native_bone
import utils.cl_loss as cl_loss
from experiments.utility_relation import bone_p2_a1_relation_action as p2a1_action
from experiments.utility_relation import bone_p2_b0_protocol as protocol
from experiments.utility_relation import bone_p2_b0_replacement as replacement
from experiments.utility_relation import evaluate_bone_p2_b0_replacement as evaluator
from experiments.utility_relation import run_bone_p2_b0_replacement as runner
from experiments.utility_relation.materialize_bone_p2_a0_carrier import sha256_file
from method.bone import BONE


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


class ToyBONE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.label_fc = torch.nn.Sequential(torch.nn.Linear(7, 7), torch.nn.Softmax(dim=1))
        self.reg = 0.7

    def forward(self, views, test: bool = False):
        labels = [self.label_fc(view) for view in views]
        return {"label": torch.argmax(torch.stack(labels).mean(dim=0), dim=1)}


def _batch(size: int = 3) -> list[torch.Tensor]:
    values = torch.arange(size * 7, dtype=torch.float32).reshape(size, 7) + 1
    return [values + view for view in range(4)]


def test_parent_controls_are_verified_and_bound_to_base_and_true_u() -> None:
    metrics = runner.verify_parent_controls()
    assert set(metrics) == {"NATIVE", "ADDON_TRUE_U"}
    assert metrics["NATIVE"]["ACC"] == pytest.approx(0.7476190476190476)
    assert metrics["ADDON_TRUE_U"]["ACC"] == pytest.approx(0.7571428571428571)


def test_four_condition_design_and_frozen_initial_state_parity() -> None:
    assert protocol.ARMS == ("NATIVE", "ADDON_TRUE_U", "REG_ONLY", "REPLACE_TRUE_U")
    hashes = [runner.model_state_sha256(runner.instantiate_initial_model(protocol.CARRIER_DIR, 20)) for _ in protocol.NEW_ARMS]
    assert len(set(hashes)) == 1
    assert hashes[0] == protocol.FROZEN_INITIAL_MODEL_SHA256


def test_replace_phase_a_reuses_exact_p2a1_true_u_primitive() -> None:
    assert replacement.semantic_batch_loss is p2a1_action.semantic_batch_loss
    assert replacement.semantic_epoch is p2a1_action.semantic_epoch
    carrier = runner.load_frozen_carrier(protocol.CARRIER_DIR, 20)
    t_rel = p2a1_action.detached_frozen(carrier.t_rel, torch.device("cpu"))
    utility = p2a1_action.detached_frozen(carrier.u_cycle, torch.device("cpu"))
    assert not t_rel.requires_grad and not utility.requires_grad


def test_phase_b_hook_assignments_and_exact_reg_only_loss() -> None:
    model, batch = ToyBONE(), _batch()
    loss, q_views, assign_matrices = replacement.reg_only_phase_b_loss(model, batch)
    expected = model.reg * cl_loss.con_reg_loss(assign_matrices)
    assert q_views.shape == (4, 3, 7)
    assert len(assign_matrices) == 4
    assert all(item.shape == (7, 3) for item in assign_matrices)
    assert torch.allclose(loss, expected)


def test_reg_only_gradient_reaches_model_parameters() -> None:
    model = ToyBONE()
    loss, _q_views, _assign = replacement.reg_only_phase_b_loss(model, _batch())
    loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in model.parameters())


def test_replacement_never_calls_inter_view_consistency(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("con_inter_view must not enter replacement Phase B")
    monkeypatch.setattr(cl_loss, "con_inter_view", forbidden)
    monkeypatch.setattr(native_bone, "con_inter_view", forbidden)
    for _arm in protocol.NEW_ARMS:
        loss, _q_views, _assign = replacement.reg_only_phase_b_loss(ToyBONE(), _batch())
        assert torch.isfinite(loss)


def test_replacement_epoch_audits_zero_inter_view_calls(monkeypatch) -> None:
    calls = 0
    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("con_inter_view must not enter replacement epoch")
    monkeypatch.setattr(cl_loss, "con_inter_view", forbidden)
    monkeypatch.setattr(native_bone, "con_inter_view", forbidden)
    for _arm in protocol.NEW_ARMS:
        model = ToyBONE()
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol.NATIVE_LR)
        loader = [(_batch(2), torch.zeros(2), torch.arange(2))]
        assert isinstance(replacement.replacement_native_epoch(model, optimizer, loader), str)
    assert calls == 0


def test_native_bone_test_false_still_calls_original_inter_view(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("native inter-view path reached")
    monkeypatch.setattr(native_bone, "con_inter_view", forbidden)
    model = BONE(m_list=[7, 7, 7, 7], k=7, a=1, v=4, layer=1, seed=20, device="cpu")
    with pytest.raises(RuntimeError, match="native inter-view path reached"):
        model([torch.ones((2, 7)) for _ in range(4)], test=False)


def test_replacement_phase_b_has_no_no_grad_and_reuses_paired_schedule() -> None:
    source = Path(replacement.__file__).read_text(encoding="utf-8")
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "torch.no_grad" not in source
    assert "_native_loader" in runner_source
    assert protocol.REFINEMENT_EPOCHS == 20
    assert protocol.SEMANTIC_LR == 1e-4


def test_semantic_and_native_optimizers_are_independent() -> None:
    model = ToyBONE()
    semantic = torch.optim.Adam(model.parameters(), lr=protocol.SEMANTIC_LR)
    native = torch.optim.Adam(model.parameters(), lr=protocol.NATIVE_LR)
    assert semantic is not native


def test_paired_native_order_and_gt_isolation_contract() -> None:
    views = [np.zeros((210, dimension), dtype=np.float32) for dimension in (24, 512, 256, 254)]
    orders = []
    for _ in protocol.NEW_ARMS:
        loader = runner._native_loader(views, 20)
        ids = np.concatenate([audit_ids.numpy().astype(np.int64) for _batch, _dummy, audit_ids in loader])
        orders.append(replacement.array_sha256(ids))
    assert orders[0] == orders[1]
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'archive["labels"]' not in source
    assert "get_cluster_metrics" not in source


def test_preregistered_evaluator_gate_is_exact() -> None:
    go = {"Delta_REPLACE_ADDON_ACC": .01, "Delta_REPLACE_REG_ACC": .01, "Interaction_ACC": .001,
          "Delta_REPLACE_ADDON_NMI": .001, "Delta_REPLACE_ADDON_Purity": 0.0}
    assert evaluator.replacement_verdict(go) == "P2_B0_REPLACEMENT_GO"
    failed = dict(go, Delta_REPLACE_ADDON_ACC=0.0)
    assert evaluator.replacement_verdict(failed) == "P2_B0_REPLACEMENT_FAIL_CLOSED"
    inconclusive = dict(go, Delta_REPLACE_ADDON_ACC=.005)
    assert evaluator.replacement_verdict(inconclusive) == "P2_B0_REPLACEMENT_INCONCLUSIVE"


def test_new_arm_evaluator_refuses_missing_seal_before_gt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing P2-B0 pre-GT seal"):
        evaluator.verify_new_arm_seal(tmp_path)


def test_native_sources_and_frozen_parent_artifacts_are_unchanged() -> None:
    assert sha256_file(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert sha256_file(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256
    assert (protocol.PARENT_P2_A1_DIR / "BASE/final_predictions.npz").is_file()
    assert (protocol.PARENT_P2_A1_DIR / "TRUE_U/final_predictions.npz").is_file()
