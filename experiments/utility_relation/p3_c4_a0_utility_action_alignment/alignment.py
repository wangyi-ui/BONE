"""Autograd-only exact-loss diagnostics for the frozen CURRENT semantic step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class GradientResult:
    values: tuple[torch.Tensor, ...]
    none_parameter_names: tuple[str, ...]
    norm: float


def semantic_parameter_set(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Trace the exact frozen CURRENT set: ``optim.Adam(model.parameters())``.

    The source call is in ``run_bone_p3_c0_cross_dataset_formal._semantic_epoch``;
    no narrower guessed submodule set is substituted here.
    """
    values = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not values:
        raise RuntimeError("CURRENT_SEMANTIC_PARAMETER_SET_EMPTY")
    return values


def validate_carrier(query_rows: torch.Tensor, anchor_rows: torch.Tensor, t_rel: torch.Tensor,
                     u_cycle: torch.Tensor, query_u_cycle: torch.Tensor) -> None:
    if query_rows.ndim != 1 or anchor_rows.ndim != 1 or u_cycle.ndim != 2 or t_rel.ndim != 3:
        raise ValueError("CARRIER_RANK_MISMATCH")
    if not torch.equal(query_u_cycle, u_cycle[query_rows]):
        raise ValueError("QUERY_U_CYCLE_PROJECTION_MISMATCH")
    if t_rel.shape != (query_rows.numel(), anchor_rows.numel(), query_u_cycle.shape[1]):
        raise ValueError("T_REL_SHAPE_MISMATCH")
    if query_u_cycle.shape[1] <= 1:
        raise ValueError("ACTION_COUNT_MUST_EXCEED_ONE")
    if not bool(torch.isfinite(t_rel).all()) or not bool(torch.isfinite(query_u_cycle).all()) or bool((query_u_cycle < 0).any()):
        raise ValueError("CARRIER_NOT_FINITE_NONNEGATIVE")


def uniform_u(u: torch.Tensor) -> torch.Tensor:
    return (u.detach() > 0).to(dtype=u.dtype)


def shuffle_u(u: torch.Tensor) -> torch.Tensor:
    if u.ndim != 2 or u.shape[1] <= 1:
        raise ValueError("SHUFFLE_REQUIRES_S_GREATER_THAN_ONE")
    out = torch.roll(u.detach(), shifts=1, dims=1)
    # A constant row has an unavoidable identity value, so require nonidentity
    # only when the input itself offers a meaningful action assignment.
    varying = (u.max(dim=1).values != u.min(dim=1).values)
    if bool(varying.any()) and bool(torch.equal(out[varying], u[varying])):
        raise RuntimeError("SHUFFLE_U_IDENTITY")
    return out


def action_relation_losses(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                           anchor_rows: torch.Tensor, positions: torch.Tensor, t_rel: torch.Tensor,
                           k: int) -> torch.Tensor:
    """Return frozen per-view relation losses ``[V,B,S]`` without reduction.

    This is a line-for-line factorization of the frozen CURRENT primitive.  It
    retains its clamp only because the source primitive has that clamp.
    """
    selected_rows = query_rows[positions]
    observed: list[torch.Tensor] = []
    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        observed.append(output)
    handle = model.label_fc.register_forward_hook(capture)
    try:
        model([torch.cat((view[selected_rows], view[anchor_rows]), dim=0) for view in full_views], test=True)
    finally:
        handle.remove()
    batch, labeled, actions = positions.numel(), anchor_rows.numel(), t_rel.shape[2]
    if len(observed) != len(full_views) or any(x.shape != (batch + labeled, k) for x in observed):
        raise RuntimeError("SEMANTIC_POSTERIOR_HOOK_SHAPE_MISMATCH")
    batch_t = t_rel[positions].detach()
    terms = []
    for posterior in observed:
        query, anchor = posterior[:batch], posterior[batch:].detach()
        relation = query @ anchor.T
        safe = relation.clamp(torch.finfo(relation.dtype).eps, 1.0 - torch.finfo(relation.dtype).eps)
        positive = -(batch_t * torch.log(safe.unsqueeze(2))).sum(dim=1) / (2 * 2)
        negative = -((1.0 - batch_t) * torch.log1p(-safe.unsqueeze(2))).sum(dim=1) / (2 * 2 * (k - 1))
        if positive.shape != (batch, actions):
            raise RuntimeError("SEMANTIC_ACTION_LOSS_SHAPE_MISMATCH")
        terms.append(positive + negative)
    return torch.stack(terms)


def current_semantic_loss(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                          anchor_rows: torch.Tensor, positions: torch.Tensor, t_rel: torch.Tensor,
                          full_query_u: torch.Tensor, k: int) -> torch.Tensor:
    """Frozen CURRENT batch contribution with its *full* utility denominator."""
    denominator = full_query_u.detach().sum()
    if not bool(denominator > 0):
        raise ValueError("CURRENT_REQUIRES_POSITIVE_UTILITY_MASS")
    losses = action_relation_losses(model, full_views, query_rows, anchor_rows, positions, t_rel, k)
    return (losses * full_query_u[positions].detach().unsqueeze(0)).sum() / (losses.shape[0] * denominator)


def plain_quintile_loss(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                        anchor_rows: torch.Tensor, positions: torch.Tensor, t_rel: torch.Tensor,
                        pair_mask: torch.Tensor, k: int) -> torch.Tensor:
    losses = action_relation_losses(model, full_views, query_rows, anchor_rows, positions, t_rel, k)
    if pair_mask.shape != losses.shape[1:]:
        raise ValueError("QUINTILE_MASK_SHAPE_MISMATCH")
    count = pair_mask.sum()
    if not bool(count > 0):
        raise ValueError("EMPTY_QUINTILE_IN_BATCH")
    return (losses * pair_mask.detach().unsqueeze(0)).sum() / (losses.shape[0] * count)


def native_loss(model: torch.nn.Module, full_views: Sequence[torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
    """Call BONE's frozen native structural objective directly."""
    loss = model([view[rows] for view in full_views])["loss"]
    if loss is None or not bool(torch.isfinite(loss)):
        raise FloatingPointError("NONFINITE_NATIVE_BONE_LOSS")
    return loss


def gradients(loss: torch.Tensor, named_parameters: tuple[tuple[str, torch.nn.Parameter], ...]) -> GradientResult:
    """Obtain gradients without ``backward``, `.grad` writes, or optimiser use."""
    parameters = tuple(value for _, value in named_parameters)
    before = tuple(parameter.grad for parameter in parameters)
    if any(value is not None for value in before):
        raise RuntimeError("PARAMETER_GRAD_ALREADY_POPULATED")
    got = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=False, create_graph=False)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("AUTOGRAD_POPULATED_PARAMETER_GRAD")
    values, none = [], []
    norm_sq = 0.0
    for (name, parameter), value in zip(named_parameters, got):
        if value is None:
            none.append(name); value = torch.zeros_like(parameter)
        values.append(value.detach())
        norm_sq += float(value.detach().double().square().sum().cpu())
    return GradientResult(tuple(values), tuple(none), norm_sq ** 0.5)


def cosine(left: GradientResult, right: GradientResult) -> tuple[float | None, str | None]:
    if left.norm == 0.0 or right.norm == 0.0:
        return None, "ZERO_GRAD_NORM"
    dot = 0.0
    for a, b in zip(left.values, right.values):
        dot += float((a.detach().double() * b.detach().double()).sum().cpu())
    return dot / (left.norm * right.norm), None


def quintile_masks(u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Stable equal-count masks over positive (supported) action pairs."""
    supported = (u.detach() > 0).cpu().numpy()
    coords = np.argwhere(supported)
    if coords.shape[0] < 5:
        raise ValueError("FIVE_SUPPORTED_PAIRS_REQUIRED")
    values = u.detach().cpu().numpy()[supported]
    # lexsort's secondary key makes ties independent of incidental loader order.
    order = np.lexsort((coords[:, 1], coords[:, 0], values))
    masks = []
    for part in np.array_split(order, 5):
        mask = np.zeros_like(supported, dtype=bool)
        mask[tuple(coords[part].T)] = True
        masks.append(torch.as_tensor(mask, device=u.device))
    return tuple(masks)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64); i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]: j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0; i = j
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    a, b = np.asarray(tuple(left), dtype=np.float64), np.asarray(tuple(right), dtype=np.float64)
    if a.size != b.size or a.size < 2 or np.std(a) == 0 or np.std(b) == 0: return None
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])
