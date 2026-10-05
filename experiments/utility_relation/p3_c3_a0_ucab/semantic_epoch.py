"""UCAB semantic reductions and the opt-in legacy-compatible semantic epoch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as legacy

from .budget import build_budget, validate_u_cycle
from .protocol import ARMS


def _views_tensor(action_losses_by_view: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    """Validate action losses and expose the shape ``[V, N_q, S]``."""
    values = action_losses_by_view if isinstance(action_losses_by_view, torch.Tensor) else torch.stack(tuple(action_losses_by_view))
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("ACTION_LOSSES_MUST_HAVE_SHAPE_V_NQ_S")
    if not bool(torch.isfinite(values.detach()).all().item()):
        raise ValueError("ACTION_LOSSES_MUST_BE_FINITE")
    return values


def reduce_action_losses(
    action_losses_by_view: torch.Tensor | Sequence[torch.Tensor],
    u_cycle: torch.Tensor,
    arm: str,
    query_ids: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Pure CPU/GPU-agnostic CURRENT or UCAB reduction, without optimizer actions."""
    # losses: [V, N_q, S]; u: [N_q, S], all utility-derived tensors detached.
    losses = _views_tensor(action_losses_by_view)
    u = validate_u_cycle(u_cycle)
    views, query_count, actions = losses.shape
    if u.shape != (query_count, actions):
        raise ValueError("ACTION_LOSS_UTILITY_SHAPE_MISMATCH")
    if chunk_size is None:
        chunk_size = query_count
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("CHUNK_SIZE_MUST_BE_POSITIVE")

    if arm == "CURRENT":
        denominator = u.sum()
        if not bool(denominator > 0):
            raise ValueError("CURRENT_REQUIRES_POSITIVE_UTILITY_MASS")
        total = losses.new_zeros(())
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            total = total + (losses[:, start:stop, :] * u[start:stop].unsqueeze(0)).sum()
        return total / (views * denominator)
    if arm not in ARMS:
        raise ValueError(f"UNKNOWN_UCAB_ARM:{arm}")

    alpha, budget, _ = build_budget(arm, u, query_ids)
    total = losses.new_zeros(())
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        # alpha_chunk: [B,S], budget_chunk: [B], action_loss: [V,B,S]
        per_sample = (losses[:, start:stop, :] * alpha[start:stop].unsqueeze(0)).sum(dim=(0, 2)) / views
        total = total + (budget[start:stop] * per_sample).sum()
    return total / query_count


def run_semantic_epoch(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    full_views: Sequence[torch.Tensor],
    query_rows: torch.Tensor,
    anchor_rows: torch.Tensor,
    t_rel: torch.Tensor,
    u_cycle: torch.Tensor,
    order: np.ndarray,
    k: int,
    arm: str,
    query_ids: torch.Tensor | None = None,
    query_chunk_size: int | None = None,
) -> float:
    """Run one semantic Phase A epoch; CURRENT dispatches directly to frozen code."""
    if arm == "CURRENT":
        return legacy._semantic_epoch(model, optimizer, full_views, query_rows, anchor_rows, t_rel, u_cycle, order, k)
    if arm not in ARMS:
        raise ValueError(f"UNKNOWN_UCAB_ARM:{arm}")
    if not np.array_equal(np.sort(order), np.arange(query_rows.numel())):
        raise ValueError("semantic query order must be a complete permutation")
    u = validate_u_cycle(u_cycle)
    t = t_rel.detach()  # T_rel is frozen target evidence and must not receive semantic gradients.
    query_count, actions = u.shape
    if t.shape != (query_count, anchor_rows.numel(), actions):
        raise ValueError("T_REL_UTILITY_SHAPE_MISMATCH")
    if query_rows.numel() != query_count or len(full_views) < 1:
        raise ValueError("QUERY_OR_VIEW_SHAPE_MISMATCH")
    ids = query_rows.detach() if query_ids is None else query_ids.detach()
    if ids.shape != (query_count,):
        raise ValueError("QUERY_ID_SHAPE_MISMATCH")
    alpha, budget, _ = build_budget(arm, u, ids)
    batch_size = legacy.protocol.SEMANTIC_BATCH_SIZE if query_chunk_size is None else query_chunk_size
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("QUERY_CHUNK_SIZE_MUST_BE_POSITIVE")

    optimizer.zero_grad(set_to_none=True)
    final_loss = 0.0
    for start in range(0, query_count, batch_size):
        positions = torch.as_tensor(order[start:start + batch_size], device=query_rows.device)
        selected_rows = query_rows[positions]
        observed: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed.append(output)

        handle = model.label_fc.register_forward_hook(capture)
        try:
            model([torch.cat((view[selected_rows], view[anchor_rows]), dim=0) for view in full_views], test=True)
        finally:
            handle.remove()
        batch, labeled = positions.numel(), anchor_rows.numel()
        if len(observed) != len(full_views) or any(value.shape != (batch + labeled, k) for value in observed):
            raise ValueError("semantic posterior hook shape mismatch")
        batch_t, batch_alpha, batch_budget = t[positions], alpha[positions], budget[positions]
        view_terms: list[torch.Tensor] = []
        for posterior in observed:
            query, anchor = posterior[:batch], posterior[batch:].detach()  # anchor posterior stop-gradient.
            relation = query @ anchor.T  # [B,L], frozen legacy relation primitive.
            safe = relation.clamp(torch.finfo(relation.dtype).eps, 1.0 - torch.finfo(relation.dtype).eps)
            positive = -(batch_t * torch.log(safe.unsqueeze(2))).sum(dim=1) / (2 * 2)
            negative = -((1.0 - batch_t) * torch.log1p(-safe.unsqueeze(2))).sum(dim=1) / (2 * 2 * (k - 1))
            if positive.shape != (batch, actions) or negative.shape != (batch, actions):
                raise ValueError("semantic action tensor shape mismatch")
            view_terms.append(((positive + negative) * batch_alpha).sum(dim=1))
        # One fixed N_q denominator across all query chunks retains absolute B magnitude.
        loss = (batch_budget * (sum(view_terms) / len(full_views))).sum() / query_count
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite UCAB semantic relation loss")
        loss.backward()
        final_loss += float(loss.detach().cpu())
    optimizer.step()  # Exactly one semantic optimizer step per semantic epoch.
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("semantic gradients leaked into native Phase B")
    return final_loss
