"""CPU-only analytical checks for the preregistered UCAB equations."""

from __future__ import annotations

import torch

from .budget import build_budget, relative_action_weights, true_budget, validate_u_cycle
from .protocol import LEGACY_SEMANTIC_CORE, LEGACY_SEMANTIC_CORE_SHA256, sha256_file, verify_preregistered_protocol
from .semantic_epoch import reduce_action_losses


def _allclose(left: torch.Tensor, right: torch.Tensor, tolerance: float = 1e-12) -> bool:
    return bool(torch.allclose(left, right, rtol=tolerance, atol=tolerance))


def run_analytical_audit() -> dict[str, bool]:
    """Return named, deterministic CPU pass/fail checks without data or model execution."""
    u = torch.tensor([[0.2, 0.7, 0.3], [0.0, 0.0, 0.0], [0.9, 0.4, 0.1], [0.1, 0.6, 0.8]], dtype=torch.float64)
    losses = torch.tensor(
        [[[0.3, 0.8, 0.5], [0.2, 0.4, 0.7], [0.9, 0.2, 0.4], [0.5, 0.3, 0.6]],
         [[0.6, 0.1, 0.4], [0.8, 0.3, 0.2], [0.7, 0.5, 0.1], [0.2, 0.9, 0.3]]], dtype=torch.float64,
    )
    query_ids = torch.tensor([40, 10, 30, 20], dtype=torch.int64)
    alpha, supported = relative_action_weights(u)
    budget = true_budget(u)
    _, mean_budget, _ = build_budget("MEAN_BUDGET", u, query_ids)
    _, shuffled_budget, _ = build_budget("SHUFFLE_BUDGET", u, query_ids)
    current = reduce_action_losses(losses, u, "CURRENT")
    linear = (losses * u.unsqueeze(0)).sum() / (losses.shape[0] * u.shape[0] * u.shape[1])
    true_full = reduce_action_losses(losses, u, "TRUE_BUDGET")
    true_chunks = reduce_action_losses(losses, u, "TRUE_BUDGET", chunk_size=1)
    equal_u = torch.full((4, 70), 0.3, dtype=torch.float64)
    scaled = u * 0.5
    expected_shuffle = torch.zeros_like(budget)
    sorted_positions = torch.tensor([3, 2, 0])
    expected_shuffle[sorted_positions] = torch.roll(budget[sorted_positions], -1)
    detached_u = u.detach().clone().requires_grad_(True)
    detached_loss = losses.detach().clone().requires_grad_(True)
    reduction = reduce_action_losses(detached_loss, detached_u, "TRUE_BUDGET")
    reduction.backward()
    protocol_ok = True
    try:
        verify_preregistered_protocol()
    except Exception:
        protocol_ok = False
    return {
        "PREREGISTRATION_INTEGRITY_PASS": protocol_ok,
        "LEGACY_SEMANTIC_CORE_SHA256_PASS": sha256_file(LEGACY_SEMANTIC_CORE) == LEGACY_SEMANTIC_CORE_SHA256,
        "UTILITY_RANGE_VALIDATION_PASS": bool(validate_u_cycle(u).shape == u.shape),
        "TRUE_BUDGET_BOUNDS_PASS": bool(((budget >= 0) & (budget <= 1)).all()),
        "ZERO_UTILITY_ZERO_BUDGET_PASS": bool(budget[1] == 0),
        "EQUAL_UTILITY_IDENTITY_PASS": _allclose(true_budget(equal_u), torch.full((4,), 0.3, dtype=torch.float64)),
        "MEAN_BUDGET_MASS_MATCH_PASS": _allclose(mean_budget[supported].mean(), budget[supported].mean()) and bool((mean_budget[~supported] == 0).all()),
        "SHUFFLE_BUDGET_MULTISET_PASS": _allclose(torch.sort(shuffled_budget[supported]).values, torch.sort(budget[supported]).values),
        # The cyclic mapping is checked by positions, not by comparing values: equal
        # budget values would still be a valid index derangement.
        "SHUFFLE_BUDGET_DERANGEMENT_PASS": _allclose(shuffled_budget, expected_shuffle),
        "LINEAR_MEAN_U_REDUNDANCY_PASS": _allclose(linear, u.mean() * current),
        "NONLINEAR_TRUE_BUDGET_NOT_GLOBAL_RESCALING_PASS": not _allclose(true_budget(scaled), 0.5 * budget),
        "CHUNK_REDUCTION_PARITY_PASS": _allclose(true_full, true_chunks),
        "DETACH_CONTRACT_PASS": (not alpha.requires_grad and not budget.requires_grad and detached_u.grad is None and detached_loss.grad is not None),
    }
