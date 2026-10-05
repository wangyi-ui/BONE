"""Pure decomposition and aggregation helpers for P3-C5-A0.

All relation-loss calls are delegated to C4's frozen alignment module.  Thus
the native objective, semantic parameter set, autograd behavior, loss clamp,
and zero-norm interpretation remain exactly C4's definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np
import torch

from experiments.utility_relation.p3_c4_a0_utility_action_alignment.alignment import (
    cosine, gradients, native_loss, plain_quintile_loss, quintile_masks, semantic_parameter_set, spearman,
)
from .protocol import CELL_COUNT, POSITIVE_CELL_MINIMUM, POSITIVE_DATASET_MINIMUM, ROUTE_SALVAGE, ROUTE_STRUCTURAL_AWARE


@dataclass(frozen=True)
class Components:
    # Every tensor is [Nq,S], selected from historical full-row arrays.
    closure_q: torch.Tensor
    conf_q: torch.Tensor
    support_q: torch.Tensor
    u_q: torch.Tensor


def extract_components(query_rows: torch.Tensor, closure: torch.Tensor, conf_gen: torch.Tensor,
                       support_ver: torch.Tensor, query_u_cycle: torch.Tensor) -> Components:
    """Project historical components to query rows without reconstructing labels."""
    if query_rows.ndim != 1 or any(value.ndim != 2 for value in (closure, conf_gen, support_ver, query_u_cycle)):
        raise ValueError("COMPONENT_RANK_MISMATCH")
    closure_q = closure[query_rows].detach()  # [Nq,S]: no component gradients enter the loss.
    conf_q = conf_gen[query_rows].detach()  # [Nq,S]: frozen generator confidence.
    support_q = support_ver[query_rows].detach()  # [Nq,S]: frozen verifier support.
    if query_u_cycle.shape != closure_q.shape or conf_q.shape != closure_q.shape or support_q.shape != closure_q.shape:
        raise ValueError("COMPONENT_QUERY_SHAPE_MISMATCH")
    return Components(closure_q, conf_q, support_q, query_u_cycle.detach())


def validate_u_parity(components: Components) -> tuple[bool, bool]:
    """Require the frozen product and closure hard-zero invariant exactly."""
    if not all(bool(torch.isfinite(value).all()) for value in components.__dict__.values()):
        raise ValueError("COMPONENT_NONFINITE")
    if bool((components.conf_q < 0).any()) or bool((components.support_q < 0).any()):
        raise ValueError("COMPONENT_NEGATIVE_CONTINUOUS_VALUE")
    reconstructed = components.closure_q * torch.sqrt(components.conf_q * components.support_q)
    parity = bool(torch.allclose(reconstructed, components.u_q, rtol=1e-6, atol=1e-7))
    zero_implies = bool(torch.equal(components.u_q[components.closure_q == 0], torch.zeros_like(components.u_q[components.closure_q == 0])))
    if not parity:
        raise ValueError("U_COMPONENT_RECONSTRUCTION_PARITY_FAILED")
    if not zero_implies:
        raise ValueError("CLOSURE_ZERO_IMPLIES_U_ZERO_FAILED")
    return parity, zero_implies


def closure_masks(closure_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C1 and C0 masks, each [Nq,S]."""
    if not bool(((closure_q == 0) | (closure_q == 1)).all()):
        raise ValueError("CLOSURE_MUST_BE_BINARY")
    return closure_q == 1, closure_q == 0


def continuous_masks(signal_q: torch.Tensor, closure_q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """C4 deterministic quintiles restricted to closure=1 pairs; masks [Nq,S]."""
    conditioned = torch.where(closure_q == 1, signal_q, torch.zeros_like(signal_q))
    return quintile_masks(conditioned)


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    total = sum(weight for _, weight in values)
    return None if total == 0 else sum(value * weight for value, weight in values) / total


def collect_group_alignment(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                            anchor_rows: torch.Tensor, t_rel: torch.Tensor, k: int, global_mask: torch.Tensor,
                            batch_positions: Iterable[torch.Tensor], named_parameters: tuple[tuple[str, torch.nn.Parameter], ...]) -> tuple[float | None, int, int]:
    """C4 plain relation loss against native gradients for one action group.

    group_mask: [Bq,S].  T_rel: [Nq,A,S].  views[v]: [N,Dv].
    """
    values: list[tuple[float, int]] = []
    invalid = 0
    for positions in batch_positions:
        local = global_mask[positions]  # [Bq,S]
        if not bool(local.any()):
            continue
        native = gradients(native_loss(model, full_views, query_rows[positions]), named_parameters)
        semantic = gradients(plain_quintile_loss(model, full_views, query_rows, anchor_rows, positions, t_rel, local, k), named_parameters)
        value, reason = cosine(semantic, native)
        if reason:
            invalid += 1
        else:
            values.append((float(value), int(positions.numel())))
        del native, semantic
    return _weighted_mean(values), len(values), invalid


def component_alignment(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                        anchor_rows: torch.Tensor, t_rel: torch.Tensor, anchor_labels: torch.Tensor,
                        masks: Sequence[torch.Tensor], batch_size: int) -> tuple[list[float | None], list[int], list[int]]:
    """Collect selected-action C4-compatible alignments for masks [Nq,S]."""
    if batch_size < 1:
        raise ValueError("BATCH_SIZE_MUST_BE_POSITIVE")
    k = int(anchor_labels.max().item()) + 1
    parameters = semantic_parameter_set(model)
    batches = tuple(torch.arange(start, min(start + batch_size, query_rows.numel()), device=query_rows.device)
                    for start in range(0, query_rows.numel(), batch_size))
    reports = [collect_group_alignment(model, full_views, query_rows, anchor_rows, t_rel, k, mask, batches, parameters)
               for mask in masks]
    return [item[0] for item in reports], [item[1] for item in reports], [item[2] for item in reports]


def component_metrics(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    if len(values) != 5 or any(value is None for value in values):
        return None, None
    typed = [float(value) for value in values]
    return typed[4] - typed[0], spearman(range(1, 6), typed)


def validate_cell(result: dict) -> None:
    required = ("U_COMPONENT_RECONSTRUCTION_PARITY", "CLOSURE_ZERO_IMPLIES_U_ZERO", "C4_U_REPLICATION_PASS")
    if not all(result.get(key) is True for key in required):
        raise ValueError("INVALID_C5_CELL_INTEGRITY")
    for key in ("Delta_CLOSURE", "Delta_CONF_Q5_Q1", "Spearman_CONF_ALIGNMENT", "Delta_SUPPORT_Q5_Q1", "Spearman_SUPPORT_ALIGNMENT"):
        value = result.get(key)
        if value is None or not np.isfinite(value):
            raise ValueError(f"INVALID_C5_COMPONENT_METRIC:{key}")
    if result.get("full_GT_loaded") or result.get("formal_clustering_metrics_computed") or result.get("training_performed") or result.get("parameter_update_performed"):
        raise ValueError("C5_DIAGNOSTIC_ONLY_FLAG_VIOLATION")


def _support(cells: Sequence[dict], delta_key: str, spearman_key: str | None = None) -> dict:
    deltas = [float(item[delta_key]) for item in cells]
    dataset_means = {dataset: fmean([float(item[delta_key]) for item in cells if item["dataset"] == dataset])
                     for dataset in sorted({item["dataset"] for item in cells})}
    report = {"valid_component_cell_count": len(cells), "macro_mean_delta": fmean(deltas),
              "positive_delta_cells": sum(value > 0 for value in deltas),
              "positive_delta_datasets": sum(value > 0 for value in dataset_means.values())}
    passed = (len(cells) == CELL_COUNT and report["macro_mean_delta"] > 0 and
              report["positive_delta_cells"] >= POSITIVE_CELL_MINIMUM and
              report["positive_delta_datasets"] >= POSITIVE_DATASET_MINIMUM)
    if spearman_key is not None:
        ranks = [float(item[spearman_key]) for item in cells]
        report.update({"macro_mean_spearman": fmean(ranks), "positive_spearman_cells": sum(value > 0 for value in ranks)})
        passed = passed and report["macro_mean_spearman"] > 0 and report["positive_spearman_cells"] >= POSITIVE_CELL_MINIMUM
    report["supported"] = passed
    return report


def aggregate_component_gates(cells: Sequence[dict]) -> dict:
    closure = _support(cells, "Delta_CLOSURE")
    conf = _support(cells, "Delta_CONF_Q5_Q1", "Spearman_CONF_ALIGNMENT")
    support = _support(cells, "Delta_SUPPORT_Q5_Q1", "Spearman_SUPPORT_ALIGNMENT")
    return {"closure": closure, "conf_gen": conf, "support_ver": support,
            "CLOSURE_STRUCTURAL_SUPPORT": closure["supported"],
            "CONF_GEN_STRUCTURAL_SUPPORT": conf["supported"],
            "SUPPORT_VER_STRUCTURAL_SUPPORT": support["supported"]}


def route_decision(closure_supported: bool, conf_supported: bool, support_supported: bool) -> dict:
    """Implement the preregistered route table; return description only."""
    if not conf_supported and not support_supported:
        return {"ROUTE": ROUTE_STRUCTURAL_AWARE, "MINIMAL_RELIABILITY_SALVAGE_AUTHORIZED": False,
                "minimal_candidate": None}
    if conf_supported and support_supported:
        if closure_supported:
            return {"ROUTE": ROUTE_STRUCTURAL_AWARE, "MINIMAL_RELIABILITY_SALVAGE_AUTHORIZED": False,
                    "minimal_candidate": None}
        candidate = "sqrt(conf_gen * support_ver)"
    elif conf_supported:
        candidate = "closure * conf_gen" if closure_supported else "conf_gen"
    else:
        candidate = "closure * support_ver" if closure_supported else "support_ver"
    return {"ROUTE": ROUTE_SALVAGE, "MINIMAL_RELIABILITY_SALVAGE_AUTHORIZED": True,
            "minimal_candidate": candidate}
