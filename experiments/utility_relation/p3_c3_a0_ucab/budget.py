"""Detached, parameter-free UCAB budget construction."""

from __future__ import annotations

import torch

from .protocol import ARMS


def validate_u_cycle(u_cycle: torch.Tensor) -> torch.Tensor:
    """Validate and stop gradients through utility, shape ``[N_q, S]``."""
    if not isinstance(u_cycle, torch.Tensor) or u_cycle.ndim != 2:
        raise ValueError("U_CYCLE_MUST_BE_A_2D_TENSOR")
    if not u_cycle.is_floating_point() or not bool(torch.isfinite(u_cycle).all().item()):
        raise ValueError("U_CYCLE_MUST_BE_FINITE_FLOAT")
    if not bool(((u_cycle >= 0) & (u_cycle <= 1)).all().item()):
        raise ValueError("U_CYCLE_MUST_BE_IN_CLOSED_UNIT_INTERVAL")
    return u_cycle.detach()


def relative_action_weights(u_cycle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached ``alpha[N_q,S]`` and ``supported_mask[N_q]`` exactly."""
    # u: [N_q, S], mass: [N_q], alpha: [N_q, S]
    u = validate_u_cycle(u_cycle)
    mass = u.sum(dim=1)
    supported = mass > 0
    safe_mass = torch.where(supported, mass, torch.ones_like(mass))
    alpha = torch.where(supported[:, None], u / safe_mass[:, None], torch.zeros_like(u))
    return alpha.detach(), supported.detach()


def true_budget(u_cycle: torch.Tensor) -> torch.Tensor:
    """Compute detached TRUE budget ``B[N_q]`` in stable float64 log-domain."""
    # u: [N_q, S]; float64 avoids product underflow for large S.
    u = validate_u_cycle(u_cycle)
    log_complement = torch.log1p(-u.to(dtype=torch.float64))
    budget = (1.0 - torch.exp(log_complement.mean(dim=1))).to(dtype=u.dtype)
    if not bool(torch.isfinite(budget).all().item()):
        raise FloatingPointError("TRUE_BUDGET_NONFINITE")
    return budget.detach()


def _validate_budget_and_support(budget: torch.Tensor, supported_mask: torch.Tensor) -> None:
    if budget.ndim != 1 or supported_mask.ndim != 1 or budget.shape != supported_mask.shape:
        raise ValueError("BUDGET_SUPPORT_SHAPE_MISMATCH")
    if not bool(torch.isfinite(budget).all().item()) or not bool(((budget >= 0) & (budget <= 1)).all().item()):
        raise ValueError("BUDGET_MUST_BE_FINITE_AND_IN_CLOSED_UNIT_INTERVAL")
    if supported_mask.dtype != torch.bool:
        raise ValueError("SUPPORTED_MASK_MUST_BE_BOOL")


def mean_budget_control(true_budget_values: torch.Tensor, supported_mask: torch.Tensor) -> torch.Tensor:
    """Remove sample-specific magnitude while retaining support and mean budget."""
    _validate_budget_and_support(true_budget_values, supported_mask)
    budget = true_budget_values.detach()
    result = torch.zeros_like(budget)
    if bool(supported_mask.any().item()):
        result[supported_mask] = budget[supported_mask].mean()
    return result.detach()


def shuffle_budget_control(
    true_budget_values: torch.Tensor, supported_mask: torch.Tensor, query_ids: torch.Tensor,
) -> torch.Tensor:
    """Deterministically cyclically derange budgets within supported query IDs."""
    _validate_budget_and_support(true_budget_values, supported_mask)
    if query_ids.ndim != 1 or query_ids.shape != true_budget_values.shape:
        raise ValueError("QUERY_ID_SHAPE_MISMATCH")
    if not bool(torch.isfinite(query_ids).all().item()):
        raise ValueError("QUERY_IDS_MUST_BE_FINITE")
    positions = torch.nonzero(supported_mask, as_tuple=False).flatten()
    if positions.numel() < 2:
        raise ValueError("SHUFFLE_BUDGET_REQUIRES_AT_LEAST_TWO_SUPPORTED_SAMPLES")
    ids = query_ids.detach()[positions]
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("SUPPORTED_QUERY_IDS_MUST_BE_UNIQUE")
    sorted_positions = positions[torch.argsort(ids)]
    values = true_budget_values.detach()[sorted_positions]
    result = torch.zeros_like(true_budget_values)
    # id_j receives B(id_{j+1}); this cyclic shift has no fixed index.
    result[sorted_positions] = torch.roll(values, shifts=-1, dims=0)
    return result.detach()


def build_budget(arm: str, u_cycle: torch.Tensor, query_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached ``(alpha[N_q,S], budget[N_q], supported_mask[N_q])``."""
    if arm not in ARMS:
        raise ValueError(f"UNKNOWN_UCAB_ARM:{arm}")
    if arm == "CURRENT":
        raise ValueError("CURRENT_USES_FROZEN_LEGACY_OBJECTIVE")
    alpha, supported = relative_action_weights(u_cycle)
    budget = true_budget(u_cycle)
    if arm == "MEAN_BUDGET":
        budget = mean_budget_control(budget, supported)
    elif arm == "SHUFFLE_BUDGET":
        if query_ids is None:
            raise ValueError("SHUFFLE_BUDGET_REQUIRES_QUERY_IDS")
        budget = shuffle_budget_control(budget, supported, query_ids)
    return alpha.detach(), budget.detach(), supported.detach()
