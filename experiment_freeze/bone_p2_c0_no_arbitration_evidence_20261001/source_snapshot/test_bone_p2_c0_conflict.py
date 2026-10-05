"""P2-C0 isolated-probe tests; the formal 20-epoch diagnostic is never executed."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.utility_relation import bone_p2_c0_conflict_protocol as protocol
from experiments.utility_relation import evaluate_bone_p2_c0_conflict as evaluator
from experiments.utility_relation import run_bone_p2_c0_conflict_diagnostic as diagnostic
from experiments.utility_relation.evaluate_bone_p2_a1_pilot import verify_pre_gt_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


class ToyBONE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.label_fc = torch.nn.Sequential(torch.nn.Linear(7, 7), torch.nn.Softmax(dim=1))
        self.reg, self.con_metric, self.device = 1.0, "cosine", torch.device("cpu")

    def forward(self, views, test: bool = False):
        outputs = [self.label_fc(view) for view in views]
        return {"label": torch.argmax(torch.stack(outputs).mean(dim=0), dim=1)}


def _full_views() -> list[torch.Tensor]:
    base = torch.arange(210 * 7, dtype=torch.float32).reshape(210, 7) / (210 * 7)
    return [base + view * 1e-3 for view in range(4)]


def _bins() -> dict[str, np.ndarray]:
    u = np.arange(196 * 6, dtype=np.float64).reshape(196, 6) + 1
    u.flat[:12] = 0
    return protocol.build_u_bins(u)


def test_parent_true_u_seal_and_frozen_inputs_are_available() -> None:
    verify_pre_gt_seal(protocol.PARENT_TRUE_U_DIR)
    assert (protocol.CARRIER_DIR / "p2_a0_carrier_pre_gt.npz").is_file()
    assert (protocol.CARRIER_DIR / "carrier_model_state.pt").is_file()


def test_u_bins_are_deterministic_stable_and_gt_free() -> None:
    u = np.ones((196, 6), dtype=np.float64)
    u.flat[:3] = 0
    first, second = protocol.build_u_bins(u), protocol.build_u_bins(u.copy())
    assert all(np.array_equal(first[name], second[name]) for name in protocol.BIN_NAMES)
    assert protocol.bins_sha256(first) == protocol.bins_sha256(second)
    assert np.array_equal(first["LOW"][:3], np.array([3, 4, 5]))
    source = Path(protocol.__file__).read_text(encoding="utf-8")
    assert "labels" not in source.lower()


def test_zero_gradient_and_missing_shared_support_fail_closed() -> None:
    zero = {"weight": torch.zeros(2)}
    nonzero = {"weight": torch.ones(2)}
    with pytest.raises(ValueError, match="zero gradient norm"):
        diagnostic._cosine(zero, nonzero)
    with pytest.raises(ValueError, match="shared-gradient"):
        diagnostic._cosine({"a": None}, {"a": torch.ones(1)})


def test_probe_isolates_live_model_optimizer_and_rng() -> None:
    model, full_views = ToyBONE(), _full_views()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    state_before = copy.deepcopy(optimizer.state_dict())
    model_before = diagnostic.model_state_sha256(model)
    random_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    relation = torch.full((196, 14, 6), 0.5).detach()
    utility = torch.ones((196, 6)).detach()
    row = diagnostic.probe_epoch(
        model, full_views, torch.arange(196), torch.arange(14, 28), relation, utility,
        np.arange(196, dtype=np.int64), [view[:3] for view in full_views], _bins(),
    )
    assert diagnostic.model_state_sha256(model) == model_before
    assert optimizer.state_dict() == state_before
    assert random.getstate() == random_before
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert row["shared_parameter_count"] > 0 and row["shared_element_count"] > 0
    assert all(f"rho_{name}" in row for name in protocol.BIN_NAMES)
    assert row["norm_sem_true"] > 0 and row["norm_inter"] > 0


def test_probe_has_no_optimizer_step_and_bin_loss_has_no_u_weighting() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    probe_source = source[source.index("def probe_epoch"):source.index("def run_diagnostic")]
    assert ".step(" not in probe_source
    bin_source = source[source.index("def _bin_semantic_loss"):source.index("def _gradient_map")]
    assert "action_weights" not in bin_source
    assert "unit_weights" in bin_source


def test_true_u_semantic_replay_and_final_prediction_parity_are_implemented() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert "semantic_epoch(" in source
    assert "_native_epoch(" in source
    assert "np.array_equal(predictions, parent[\"predictions\"])" in source
    assert protocol.EPOCHS == 20


def test_preregistered_gate_is_exact() -> None:
    support = [{"global_inter_conflict": 1 if index < 5 else 0, "rho_HIGH": .2, "rho_LOW": .1,
                "high_minus_low": .1, "low_conflict": 1, "high_conflict": 0} for index in range(20)]
    assert evaluator.diagnostic_verdict(support) == "P2_C0_SUPPORTS_ARBITRATION"
    no_evidence = [dict(row, global_inter_conflict=0, high_minus_low=0.0) for row in support]
    assert evaluator.diagnostic_verdict(no_evidence) == "P2_C0_NO_ARBITRATION_EVIDENCE"
    inconclusive = [dict(row, global_inter_conflict=3, high_minus_low=.01, rho_HIGH=.11, rho_LOW=.1,
                         low_conflict=0, high_conflict=0) for row in support]
    assert evaluator.diagnostic_verdict(inconclusive) == "P2_C0_INCONCLUSIVE"


def test_gt_isolation_and_native_sources_unchanged() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert 'archive["labels"]' not in source
    assert "get_cluster_metrics" not in source
    assert sha256_file(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert sha256_file(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256
