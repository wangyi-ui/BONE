"""Pre-GT utility discriminability and sparse-anchor semantic direction tests."""
from __future__ import annotations

import math
import numpy as np
import torch

from .alignment import spearman


def _percentiles(values: np.ndarray, points: tuple[int, ...]) -> dict[str, float]:
    return {f"p{point}": float(np.percentile(values, point)) for point in points}


def utility_diagnostics(u: torch.Tensor) -> dict:
    values = u.detach().double().cpu().numpy()
    if values.ndim != 2 or values.shape[1] <= 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("INVALID_UTILITY_MATRIX")
    mass = values.sum(axis=1); supported = mass > 0
    if not supported.any(): raise ValueError("NO_SUPPORTED_UTILITY_SAMPLE")
    alpha = values[supported] / mass[supported, None]
    log_alpha = np.zeros_like(alpha)
    positive = alpha > 0
    log_alpha[positive] = np.log(alpha[positive])
    entropy_raw = -(alpha * log_alpha).sum(axis=1)
    entropy = entropy_raw / math.log(values.shape[1])
    ordered = np.sort(values[supported], axis=1)
    effective = np.exp(entropy_raw)
    flat = values.ravel()
    return {"support_fraction": float(supported.mean()), "U_mean": float(flat.mean()), "U_std": float(flat.std()),
            "U_quantiles": _percentiles(flat, (10, 25, 50, 75, 90)),
            "entropy_mean": float(entropy.mean()), "entropy_median": float(np.median(entropy)),
            "top1_top2_margin_mean": float((ordered[:, -1] - ordered[:, -2]).mean()),
            "top1_top2_margin_median": float(np.median(ordered[:, -1] - ordered[:, -2])),
            "effective_action_number_mean": float(effective.mean()),
            "effective_action_number_over_S_mean": float((effective / values.shape[1]).mean())}


def semantic_margins(t_rel: torch.Tensor, anchor_labels: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Aggregate T_rel by its sparse anchor labels and return ``[Nq,S]`` margins."""
    t = t_rel.detach()
    labels = anchor_labels.detach().to(device=t.device, dtype=torch.long)
    if t.ndim != 3 or labels.ndim != 1 or t.shape[1] != labels.numel():
        raise ValueError("T_REL_ANCHOR_LABEL_SHAPE_MISMATCH")
    classes = int(labels.max().item()) + 1 if k is None else int(k)
    if classes < 2 or bool((labels < 0).any()) or bool((labels >= classes).any()):
        raise ValueError("INVALID_SPARSE_ANCHOR_LABELS")
    grouped = []
    for class_id in range(classes):
        selected = labels == class_id
        if not bool(selected.any()): raise ValueError("ANCHOR_CLASS_MISSING")
        grouped.append(t[:, selected, :].mean(dim=1))
    evidence = torch.stack(grouped, dim=2)  # [Nq,S,C]
    mass = evidence.sum(dim=2)
    if bool((mass <= 0).any()): raise ValueError("ZERO_SEMANTIC_CLASS_MASS")
    posterior = evidence / mass.unsqueeze(2)
    tops = posterior.topk(k=2, dim=2).values
    return tops[:, :, 0] - tops[:, :, 1]


def direction_diagnostics(u: torch.Tensor, margin: torch.Tensor) -> dict:
    if u.shape != margin.shape: raise ValueError("U_MARGIN_SHAPE_MISMATCH")
    supported = u.detach() > 0
    uv, mv = u.detach()[supported].double().cpu().numpy(), margin.detach()[supported].double().cpu().numpy()
    if uv.size == 0: raise ValueError("NO_SUPPORTED_UTILITY_PAIR")
    low, high = np.percentile(uv, 20), np.percentile(uv, 80)
    q1, q5 = mv[uv <= low], mv[uv >= high]
    threshold_u, threshold_m = np.percentile(uv, 75), np.percentile(mv, 25)
    high_u = uv >= threshold_u
    if not high_u.any(): raise ValueError("NO_HIGH_U_PAIR")
    return {"Spearman_U_semantic_margin": spearman(uv, mv),
            "mean_margin_lowest_U_quintile": float(q1.mean()), "mean_margin_highest_U_quintile": float(q5.mean()),
            "DeltaMargin_high_low": float(q5.mean() - q1.mean()),
            "HIGH_U_LOW_MARGIN_CONFLICT_RATE": float(np.mean(mv[high_u] <= threshold_m)),
            "semantic_margin_mean": float(mv.mean()), "semantic_margin_median": float(np.median(mv))}
