from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest
import torch

from experiments.utility_relation.p3_c3_a0_ucab import protocol
from experiments.utility_relation.p3_c3_a0_ucab.audit import run_analytical_audit
from experiments.utility_relation.p3_c3_a0_ucab.budget import (
    build_budget,
    mean_budget_control,
    relative_action_weights,
    shuffle_budget_control,
    true_budget,
    validate_u_cycle,
)
from experiments.utility_relation.p3_c3_a0_ucab.semantic_epoch import reduce_action_losses, run_semantic_epoch


def _u() -> torch.Tensor:
    return torch.tensor([[0.2, 0.7, 0.3], [0.0, 0.0, 0.0], [0.9, 0.4, 0.1], [0.1, 0.6, 0.8]], dtype=torch.float64)


def _losses(requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        [[[0.3, 0.8, 0.5], [0.2, 0.4, 0.7], [0.9, 0.2, 0.4], [0.5, 0.3, 0.6]],
         [[0.6, 0.1, 0.4], [0.8, 0.3, 0.2], [0.7, 0.5, 0.1], [0.2, 0.9, 0.3]]],
        dtype=torch.float64, requires_grad=requires_grad,
    )


def test_protocol_freeze_integrity_and_legacy_hash():
    verified = protocol.verify_preregistered_protocol()
    assert str(protocol.LEGACY_SEMANTIC_CORE) in verified
    assert protocol.sha256_file(protocol.LEGACY_SEMANTIC_CORE) == protocol.LEGACY_SEMANTIC_CORE_SHA256


@pytest.mark.parametrize("value", [torch.tensor([0.1]), torch.tensor([[float("nan")]]), torch.tensor([[-0.1]]), torch.tensor([[1.1]])])
def test_utility_validation_fails_closed(value: torch.Tensor):
    with pytest.raises(ValueError):
        validate_u_cycle(value)


def test_alpha_support_and_detach_contract():
    u = _u().requires_grad_(True)
    alpha, supported = relative_action_weights(u)
    assert alpha.shape == u.shape and alpha.requires_grad is False
    assert torch.allclose(alpha[supported].sum(dim=1), torch.ones(int(supported.sum()), dtype=u.dtype))
    assert torch.equal(alpha[~supported], torch.zeros_like(alpha[~supported]))


def test_true_budget_bounds_zero_constant_large_actions_and_one():
    budget = true_budget(_u())
    assert bool(((budget >= 0) & (budget <= 1)).all()) and budget[1] == 0
    for actions in (2, 6, 20, 70):
        assert torch.allclose(true_budget(torch.full((3, actions), 0.3, dtype=torch.float64)), torch.full((3,), 0.3, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    one = true_budget(torch.tensor([[0.2, 1.0, 0.4]], dtype=torch.float64))
    assert torch.isfinite(one).all() and one.item() == 1.0


def test_mean_and_shuffle_controls_are_exact_and_deterministic():
    u, ids = _u(), torch.tensor([40, 10, 30, 20])
    alpha, supported = relative_action_weights(u)
    budget = true_budget(u)
    mean = mean_budget_control(budget, supported)
    assert torch.equal(mean[~supported], torch.zeros_like(mean[~supported]))
    assert torch.allclose(mean[supported], torch.full_like(mean[supported], budget[supported].mean()))
    assert torch.allclose(mean[supported].mean(), budget[supported].mean())
    first = shuffle_budget_control(budget, supported, ids)
    second = shuffle_budget_control(budget, supported, ids)
    assert torch.equal(first, second) and torch.equal(first[~supported], torch.zeros_like(first[~supported]))
    assert torch.equal(torch.sort(first[supported]).values, torch.sort(budget[supported]).values)
    assert bool((first[supported] != budget[supported]).all())
    assert torch.allclose(first[supported].mean(), budget[supported].mean())
    assert torch.allclose(first[supported].var(), budget[supported].var())
    assert first[supported].min() == budget[supported].min() and first[supported].max() == budget[supported].max()
    with pytest.raises(ValueError, match="AT_LEAST_TWO"):
        shuffle_budget_control(torch.tensor([0.5]), torch.tensor([True]), torch.tensor([4]))
    assert alpha.requires_grad is False


def test_linear_redundancy_current_parity_and_scale_behavior():
    u, losses = _u(), _losses()
    current = reduce_action_losses(losses, u, "CURRENT")
    reconstructed = (losses * u.unsqueeze(0)).sum() / (losses.shape[0] * u.sum())
    linear = (losses * u.unsqueeze(0)).sum() / (losses.shape[0] * u.shape[0] * u.shape[1])
    assert torch.allclose(current, reconstructed, atol=1e-12, rtol=1e-12)
    assert torch.allclose(linear, u.mean() * current, atol=1e-12, rtol=1e-12)
    assert torch.allclose(current, reduce_action_losses(losses, 0.5 * u, "CURRENT"), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(reduce_action_losses(losses, u, "TRUE_BUDGET"), reduce_action_losses(losses, 0.5 * u, "TRUE_BUDGET"), atol=1e-12, rtol=1e-12)


def test_true_budget_nonlinearity_chunk_and_gradient_parity():
    u, losses = _u(), _losses()
    assert not torch.allclose(true_budget(0.5 * u), 0.5 * true_budget(u), atol=1e-12, rtol=1e-12)
    full = reduce_action_losses(losses, u, "TRUE_BUDGET")
    assert torch.allclose(full, reduce_action_losses(losses, u, "TRUE_BUDGET", chunk_size=1), atol=1e-12, rtol=1e-12)
    assert torch.allclose(full, reduce_action_losses(losses, u, "TRUE_BUDGET", chunk_size=2), atol=1e-12, rtol=1e-12)
    gradients = []
    for size in (4, 1, 2):
        action = _losses(requires_grad=True)
        reduce_action_losses(action, u, "TRUE_BUDGET", chunk_size=size).backward()
        gradients.append(action.grad.clone())
    assert all(torch.allclose(gradients[0], item, atol=1e-12, rtol=1e-12) for item in gradients[1:])


def test_u_never_receives_gradient_action_losses_do():
    u, action = _u().requires_grad_(True), _losses(requires_grad=True)
    alpha, budget, _ = build_budget("TRUE_BUDGET", u)
    assert not alpha.requires_grad and not budget.requires_grad
    reduce_action_losses(action, u, "TRUE_BUDGET").backward()
    assert u.grad is None and action.grad is not None and bool((action.grad != 0).any())


def test_current_integrated_path_dispatches_to_frozen_legacy(monkeypatch):
    from experiments.utility_relation.p3_c3_a0_ucab import semantic_epoch

    sentinel = object()
    called = {}

    def frozen(*args):
        called["args"] = args
        return sentinel

    monkeypatch.setattr(semantic_epoch.legacy, "_semantic_epoch", frozen)
    result = run_semantic_epoch(*([object()] * 9), arm="CURRENT")
    assert result is sentinel and len(called["args"]) == 9


def test_ucab_integrated_epoch_has_one_cpu_optimizer_step_and_cleans_gradients():
    class HookModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.label_fc = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Sigmoid())

        def forward(self, views, test=True):
            return {"label": [self.label_fc(view) for view in views]}

    class CountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    model = HookModel().double()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    full_views = [torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.7, 0.3], [0.3, 0.7]], dtype=torch.float64)]
    query_rows, anchor_rows = torch.tensor([0, 1, 2]), torch.tensor([3, 4])
    target = torch.full((3, 2, 2), 0.4, dtype=torch.float64, requires_grad=True)
    utility = torch.tensor([[0.2, 0.8], [0.5, 0.3], [0.6, 0.1]], dtype=torch.float64, requires_grad=True)
    result = run_semantic_epoch(model, optimizer, full_views, query_rows, anchor_rows, target, utility, [2, 0, 1], 2, "TRUE_BUDGET", query_chunk_size=1)
    assert isinstance(result, float) and optimizer.steps == 1
    assert utility.grad is None and target.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_analytical_audit_and_source_firewall():
    assert all(run_analytical_audit().values())
    root = Path(__file__).resolve().parents[1] / "experiments/utility_relation/p3_c3_a0_ucab"
    forbidden = ("accuracy", "nmi", "ari", "purity", "evaluator", "loadmat")
    for source in root.glob("*.py"):
        text = source.read_text().lower()
        assert not any(re.search(rf"\\b{item}\\b", text) for item in forbidden)
        tree = ast.parse(text)
        imports = {alias.name.lower() for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not any("evaluate" in item for item in imports)
