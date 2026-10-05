from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from experiments.utility_relation.p3_c4_a0_utility_action_alignment.alignment import plain_quintile_loss, quintile_masks
from experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition import decomposition
from experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition.decomposition import (
    Components, closure_masks, continuous_masks, extract_components, route_decision, validate_u_parity,
)
from experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition.protocol import (
    DATASETS, REQUIRED_OUTPUT_KEYS, manifest_cells, verify_parent_freeze,
)
from experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition.run_diagnostic import diagnose_cell
from experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition.summarize_diagnostic import summarize
from method.bone import BONE


def _synthetic():
    torch.manual_seed(9)
    model = BONE([3, 4], k=2, a=1, v=2, layer=1, device="cpu", info=False)
    views = [torch.randn(12, 3), torch.randn(12, 4)]
    anchor, query = torch.tensor([0, 1]), torch.arange(2, 12)
    t = torch.tensor([[[.8, .2], [.2, .8]]] * 10)
    closure = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 1.], [1., 1.],
                            [1., 0.], [0., 1.], [1., 1.], [1., 1.], [0., 0.], [0., 0.]])
    conf = torch.linspace(.1, .9, 24).reshape(12, 2)
    support = torch.linspace(.9, .2, 24).reshape(12, 2)
    full_u = closure * torch.sqrt(conf * support)
    return model, views, query, anchor, t, full_u, closure, conf, support, torch.tensor([0, 1])


def test_parent_integrity_and_frozen_c4_native_sources_remain_unchanged():
    receipts = verify_parent_freeze()
    assert receipts["C4_SOURCE_SHA256"] and receipts["C4_TEST_SHA256"]


def test_exact_reconstruction_zero_rule_and_mismatch_fail_closed():
    _, _, query, _, _, u, closure, conf, support, _ = _synthetic()
    components = extract_components(query, closure, conf, support, u[query])
    assert validate_u_parity(components) == (True, True)
    bad = Components(components.closure_q, components.conf_q, components.support_q, components.u_q + .01)
    with pytest.raises(ValueError, match="RECONSTRUCTION"):
        validate_u_parity(bad)
    assert torch.equal(components.u_q[components.closure_q == 0], torch.zeros_like(components.u_q[components.closure_q == 0]))


def test_closure_and_continuous_conditioning_groups_are_exact_and_c4_compatible():
    _, _, query, _, _, u, closure, conf, support, _ = _synthetic()
    c = extract_components(query, closure, conf, support, u[query])
    c1, c0 = closure_masks(c.closure_q)
    assert torch.equal(c1 | c0, torch.ones_like(c1, dtype=torch.bool)) and not bool((c1 & c0).any())
    masks = continuous_masks(c.conf_q, c.closure_q)
    assert all(not bool(mask[c0].any()) for mask in masks)
    conditioned = torch.where(c.closure_q == 1, c.conf_q, torch.zeros_like(c.conf_q))
    assert all(torch.equal(left, right) for left, right in zip(masks, quintile_masks(conditioned)))
    assert sum(mask.sum().item() for mask in masks) == int(c1.sum())
    # C4's plain loss only receives the group mask, never the tested signal.
    source = inspect.getsource(plain_quintile_loss)
    assert "pair_mask.detach" in source and "full_query_u" not in source


def test_autograd_alignment_is_read_only_and_zero_grad_is_invalid():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    zero = decomposition.gradients(parameter * 0.0, (("p", parameter),))
    assert parameter.grad is None and decomposition.cosine(zero, zero) == (None, "ZERO_GRAD_NORM")
    model, views, query, anchor, t, u, closure, conf, support, labels = _synthetic()
    result = diagnose_cell(model, views, query, anchor, t, u, u[query], closure, conf, support, labels, query, batch_size=10)
    assert result["model_state_unchanged"] and all(p.grad is None for p in model.parameters())
    assert result["C4_U_REPLICATION_PASS"]
    assert REQUIRED_OUTPUT_KEYS.issubset(result | {"stage": "P3-C5-A0", "dataset": "HandWrite", "training_seed": 20,
                                                    "DIAGNOSTIC_ONLY": True, "full_GT_loaded": False,
                                                    "formal_clustering_metrics_computed": False,
                                                    "training_performed": False, "parameter_update_performed": False})


def _positive_ids(count: int) -> set[int]:
    # Two positive cells in six datasets establishes the preregistered dataset gate.
    picks = [dataset * 3 + seed for dataset in range(6) for seed in range(2)]
    picks.extend(range(18, 24))
    return set(picks[:count])


def _cells(closure_count=16, conf_count=16, support_count=16):
    closure_ids, conf_ids, support_ids = _positive_ids(closure_count), _positive_ids(conf_count), _positive_ids(support_count)
    cells = []
    for index, (dataset, seed) in enumerate(manifest_cells()):
        def signed(ids): return 1.0 if index in ids else -1.0
        cells.append({"stage": "P3-C5-A0", "dataset": dataset, "training_seed": seed, "DIAGNOSTIC_ONLY": True,
                      "full_GT_loaded": False, "formal_clustering_metrics_computed": False, "training_performed": False,
                      "parameter_update_performed": False, "U_COMPONENT_RECONSTRUCTION_PARITY": True,
                      "CLOSURE_ZERO_IMPLIES_U_ZERO": True, "C4_U_REPLICATION_PASS": True,
                      "Delta_CLOSURE": signed(closure_ids), "Delta_CONF_Q5_Q1": signed(conf_ids),
                      "Spearman_CONF_ALIGNMENT": signed(conf_ids), "Delta_SUPPORT_Q5_Q1": signed(support_ids),
                      "Spearman_SUPPORT_ALIGNMENT": signed(support_ids)})
    return cells


def test_summary_exact_grid_and_component_pass_fail_gates():
    passing = summarize(_cells())
    assert passing["component_gates"]["CLOSURE_STRUCTURAL_SUPPORT"]
    assert passing["component_gates"]["CONF_GEN_STRUCTURAL_SUPPORT"]
    assert passing["component_gates"]["SUPPORT_VER_STRUCTURAL_SUPPORT"]
    failed = summarize(_cells(closure_count=15, conf_count=15, support_count=15))
    assert not failed["component_gates"]["CLOSURE_STRUCTURAL_SUPPORT"]
    assert not failed["component_gates"]["CONF_GEN_STRUCTURAL_SUPPORT"]
    assert not failed["component_gates"]["SUPPORT_VER_STRUCTURAL_SUPPORT"]
    with pytest.raises(ValueError, match="EXACT_8_X_3"):
        summarize(_cells()[:-1])


@pytest.mark.parametrize(("c", "g", "v", "route", "candidate", "authorized"), [
    (True, False, True, "MINIMAL_COMPONENT_SALVAGE_CONFIRMATION", "closure * support_ver", True),
    (False, False, True, "MINIMAL_COMPONENT_SALVAGE_CONFIRMATION", "support_ver", True),
    (True, True, False, "MINIMAL_COMPONENT_SALVAGE_CONFIRMATION", "closure * conf_gen", True),
    (False, True, True, "MINIMAL_COMPONENT_SALVAGE_CONFIRMATION", "sqrt(conf_gen * support_ver)", True),
    (True, True, True, "STRUCTURAL_AWARE_ACTION_VALUE_DESIGN", None, False),
    (True, False, False, "STRUCTURAL_AWARE_ACTION_VALUE_DESIGN", None, False),
    (False, False, False, "STRUCTURAL_AWARE_ACTION_VALUE_DESIGN", None, False),
])
def test_preregistered_route_table_and_gamma_prohibition(c, g, v, route, candidate, authorized):
    decision = route_decision(c, g, v)
    assert decision == {"ROUTE": route, "MINIMAL_RELIABILITY_SALVAGE_AUTHORIZED": authorized, "minimal_candidate": candidate}
    assert summarize(_cells())["gamma_prohibited"] is True


def test_new_runner_contains_no_mutation_or_gt_surface():
    import experiments.utility_relation.p3_c5_a0_utility_reliability_decomposition.run_diagnostic as runner
    source = inspect.getsource(runner)
    assert "optimizer.step" not in source and ".backward(" not in source and "torch.optim" not in source
    assert "--full-gt" not in source and "--labels" not in source and "--gamma" not in source
