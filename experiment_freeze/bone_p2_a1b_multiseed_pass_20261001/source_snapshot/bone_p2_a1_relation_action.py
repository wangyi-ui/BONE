"""BONE-specific frozen relation action used by P2-A1 semantic Phase A."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch

from experiments.utility_relation.bone_p2_a1_protocol import K, L, SEMANTIC_BATCH_SIZE, S, V


M_PER_CLASS = 2


def detached_frozen(values: np.ndarray, device: torch.device) -> torch.Tensor:
    """Transfer a carrier tensor without creating a gradient path to it."""
    return torch.as_tensor(values, dtype=torch.float32, device=device).detach()


def same_forward_posteriors(
    model: torch.nn.Module,
    query_views: Sequence[torch.Tensor],
    anchor_views: Sequence[torch.Tensor],
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Extract query and detached anchor posteriors from one shared BONE forward."""
    if len(query_views) != V or len(anchor_views) != V:
        raise ValueError("BONE requires four query and four anchor views")
    batch_size = query_views[0].shape[0]
    if batch_size < 1 or batch_size > SEMANTIC_BATCH_SIZE:
        raise ValueError("semantic query batch size is invalid")
    if any(view.shape[0] != batch_size for view in query_views):
        raise ValueError("query view batch sizes disagree")
    if any(view.shape[0] != L for view in anchor_views):
        raise ValueError("anchor views must contain exactly 14 rows")
    observed: list[torch.Tensor] = []

    def capture(_module, _inputs, output: torch.Tensor) -> None:
        observed.append(output)

    handle = model.label_fc.register_forward_hook(capture)
    try:
        semantic_views = [torch.cat((query, anchor), dim=0) for query, anchor in zip(query_views, anchor_views)]
        model(semantic_views, test=True)
    finally:
        handle.remove()
    if len(observed) != V or any(item.shape != (batch_size + L, K) for item in observed):
        raise ValueError("label hook did not produce four shared-batch posterior tensors")
    q_query = [item[:batch_size] for item in observed]
    q_anchor = [item[batch_size:].detach() for item in observed]
    return q_query, q_anchor


def relation_action_loss(
    q_query: torch.Tensor,
    q_anchor: torch.Tensor,
    t_batch: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return raw posterior relation matrix and weighted full-epoch numerator contribution."""
    if q_query.ndim != 2 or q_query.shape[1] != K or q_anchor.shape != (L, K):
        raise ValueError("posterior shapes must be [B,7] and [14,7]")
    if t_batch.shape != (q_query.shape[0], L, S) or weights.shape != (q_query.shape[0], S):
        raise ValueError("frozen relation target or action weight shape is invalid")
    if q_anchor.requires_grad or t_batch.requires_grad or weights.requires_grad:
        raise ValueError("anchors, relation targets, and action weights must be detached")
    relation = q_query @ q_anchor.T
    epsilon = torch.finfo(relation.dtype).eps
    safe_relation = relation.clamp(epsilon, 1.0 - epsilon)
    positive = -(t_batch * torch.log(safe_relation.unsqueeze(2))).sum(dim=1) / (2 * M_PER_CLASS)
    negative = -((1.0 - t_batch) * torch.log1p(-safe_relation.unsqueeze(2))).sum(dim=1)
    negative = negative / (2 * M_PER_CLASS * (K - 1))
    action_loss = positive + negative
    return relation, action_loss


def semantic_batch_loss(
    model: torch.nn.Module,
    query_views: Sequence[torch.Tensor],
    anchor_views: Sequence[torch.Tensor],
    t_batch: torch.Tensor,
    weights: torch.Tensor,
    full_epoch_denominator: torch.Tensor,
) -> Tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute one semantic minibatch contribution; this function never steps an optimizer."""
    if full_epoch_denominator.ndim != 0 or not bool(full_epoch_denominator > 0):
        raise ValueError("full semantic epoch denominator must be positive")
    q_query, q_anchor = same_forward_posteriors(model, query_views, anchor_views)
    numerators: list[torch.Tensor] = []
    relations: list[torch.Tensor] = []
    for query, anchor in zip(q_query, q_anchor):
        relation, action_loss = relation_action_loss(query, anchor, t_batch, weights)
        relations.append(relation)
        numerators.append((weights * action_loss).sum())
    return sum(numerators) / (V * full_epoch_denominator), relations


def semantic_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    full_views: Sequence[torch.Tensor],
    unlabeled_ids: torch.Tensor,
    labeled_ids: torch.Tensor,
    relation_targets: torch.Tensor,
    action_weights: torch.Tensor,
    query_order: np.ndarray,
) -> int:
    """Run exactly one full semantic epoch and exactly one optimizer step."""
    if query_order.shape != (unlabeled_ids.numel(),):
        raise ValueError("semantic query order must cover every frozen unlabeled position once")
    if not np.array_equal(np.sort(query_order), np.arange(unlabeled_ids.numel())):
        raise ValueError("semantic query order is not a permutation")
    if relation_targets.requires_grad or action_weights.requires_grad:
        raise ValueError("U_cycle and T_rel must remain detached")
    denominator = action_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("full semantic epoch denominator must be positive")
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, unlabeled_ids.numel(), SEMANTIC_BATCH_SIZE):
        positions = torch.as_tensor(query_order[start:start + SEMANTIC_BATCH_SIZE], device=unlabeled_ids.device)
        query_ids = unlabeled_ids[positions]
        query_views = [view[query_ids] for view in full_views]
        anchor_views = [view[labeled_ids] for view in full_views]
        loss, _relations = semantic_batch_loss(
            model, query_views, anchor_views, relation_targets[positions], action_weights[positions], denominator
        )
        loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("semantic gradients were not cleared before native Phase B")
    return 1


def zero_action_semantic_graph(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    full_views: Sequence[torch.Tensor],
    unlabeled_ids: torch.Tensor,
    labeled_ids: torch.Tensor,
    relation_targets: torch.Tensor,
    action_weights: torch.Tensor,
) -> None:
    """Test-only Phase-A identity path; action scale is fixed internally to zero."""
    order = np.arange(unlabeled_ids.numel(), dtype=np.int64)
    denominator = action_weights.sum()
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, unlabeled_ids.numel(), SEMANTIC_BATCH_SIZE):
        positions = torch.as_tensor(order[start:start + SEMANTIC_BATCH_SIZE], device=unlabeled_ids.device)
        query_ids = unlabeled_ids[positions]
        loss, _relations = semantic_batch_loss(
            model, [view[query_ids] for view in full_views], [view[labeled_ids] for view in full_views],
            relation_targets[positions], action_weights[positions], denominator
        )
        action_scale = 0.0
        (loss * action_scale).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
