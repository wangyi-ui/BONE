"""Read-only one-cell runner for P3-C4-A0.

Inputs are pre-GT carrier/model/view artifacts and sparse anchor labels only.
It never accepts or opens a full-label/GT input and refuses to overwrite output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from method.bone import BONE

from .alignment import (cosine, current_semantic_loss, gradients, native_loss, plain_quintile_loss,
                        quintile_masks, semantic_parameter_set, shuffle_u, uniform_u, validate_carrier)
from .protocol import OUTPUT_ROOT, require_cell, verify_frozen_sources
from .semantic_direction import direction_diagnostics, semantic_margins, utility_diagnostics


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _weighted_mean(pairs: list[tuple[float, int]]) -> float | None:
    denominator = sum(weight for _, weight in pairs)
    return None if denominator == 0 else sum(value * weight for value, weight in pairs) / denominator


def _new_model_from_state(views: Sequence[torch.Tensor], state: dict, device: torch.device) -> tuple[BONE, int]:
    weight = state.get("label_fc.0.weight")
    if weight is None or weight.ndim != 2: raise ValueError("MODEL_STATE_MISSING_LABEL_FC")
    k, r = weight.shape
    if r % k: raise ValueError("MODEL_STATE_INVALID_ANCHOR_WIDTH")
    layers = len([key for key in state if key.startswith("fc_Z.0.") and key.endswith(".weight")])
    if layers < 1: raise ValueError("MODEL_STATE_MISSING_FROZEN_LAYER_STACK")
    model = BONE([int(view.shape[1]) for view in views], int(k), int(r // k), len(views), layer=layers,
                 device=str(device), info=False).to(device)
    model.load_state_dict(state, strict=True)
    return model, int(k)


def _batch_positions(n: int, batch_size: int, device: torch.device):
    for start in range(0, n, batch_size):
        yield torch.arange(start, min(start + batch_size, n), device=device)


def diagnose_cell(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                  anchor_rows: torch.Tensor, t_rel: torch.Tensor, u_cycle: torch.Tensor,
                  query_u_cycle: torch.Tensor, anchor_labels: torch.Tensor, raw_query_ids: torch.Tensor,
                  batch_size: int = 128) -> dict:
    """Run all arms with autograd only; model state and `.grad` remain untouched."""
    if batch_size < 1: raise ValueError("BATCH_SIZE_MUST_BE_POSITIVE")
    validate_carrier(query_rows, anchor_rows, t_rel, u_cycle, query_u_cycle)
    if raw_query_ids.shape != query_rows.shape or torch.unique(raw_query_ids).numel() != raw_query_ids.numel():
        raise ValueError("RAW_QUERY_ID_AUDIT_MISMATCH")
    parameters = semantic_parameter_set(model)
    initial = _state_digest(model)
    true_u, uni_u, shuffled_u = query_u_cycle.detach(), uniform_u(query_u_cycle), shuffle_u(query_u_cycle)
    k = int(anchor_labels.max().item()) + 1
    masks = quintile_masks(true_u)
    arm_values = {"TRUE": [], "UNIFORM": [], "SHUFFLE": []}
    q_values = [[] for _ in range(5)]
    invalid = {"TRUE": 0, "UNIFORM": 0, "SHUFFLE": 0, "Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "Q5": 0}
    native_norms: list[float] = []; semantic_norms = {name: [] for name in invalid}
    for positions in _batch_positions(query_rows.numel(), batch_size, query_rows.device):
        # Native is recomputed only once for this batch; semantic arms each make
        # their own forward graph, avoiding retain_graph memory accumulation.
        native = gradients(native_loss(model, full_views, query_rows[positions]), parameters)
        native_norms.append(native.norm)
        for name, weights in (("TRUE", true_u), ("UNIFORM", uni_u), ("SHUFFLE", shuffled_u)):
            semantic = gradients(current_semantic_loss(model, full_views, query_rows, anchor_rows, positions, t_rel, weights, k), parameters)
            semantic_norms[name].append(semantic.norm)
            value, reason = cosine(semantic, native)
            if reason: invalid[name] += 1
            else: arm_values[name].append((float(value), int(positions.numel())))
        for index, global_mask in enumerate(masks):
            local = global_mask[positions]
            if not bool(local.any()): continue
            semantic = gradients(plain_quintile_loss(model, full_views, query_rows, anchor_rows, positions, t_rel, local, k), parameters)
            semantic_norms[f"Q{index + 1}"].append(semantic.norm)
            value, reason = cosine(semantic, native)
            if reason: invalid[f"Q{index + 1}"] += 1
            else: q_values[index].append((float(value), int(positions.numel())))
        del native
        if torch.cuda.is_available() and query_rows.is_cuda: torch.cuda.empty_cache()
    if _state_digest(model) != initial or any(p.grad is not None for _, p in parameters):
        raise RuntimeError("READ_ONLY_AUTOGRAD_CONTRACT_VIOLATION")
    align = {f"A_{name}": _weighted_mean(values) for name, values in arm_values.items()}
    q_align = [_weighted_mean(values) for values in q_values]
    if any(value is None for value in align.values()) or any(value is None for value in q_align):
        # Preserve results for inspection but do not manufacture an epsilon-based answer.
        status = "INVALID_ZERO_GRAD_NORM"
    else: status = "VALID"
    margins = semantic_margins(t_rel, anchor_labels, k)
    result = {"status": status, **align, **{f"A_Q{i + 1}": value for i, value in enumerate(q_align)},
              "Delta_TRUE_UNIFORM": None if align["A_TRUE"] is None or align["A_UNIFORM"] is None else align["A_TRUE"] - align["A_UNIFORM"],
              "Delta_TRUE_SHUFFLE": None if align["A_TRUE"] is None or align["A_SHUFFLE"] is None else align["A_TRUE"] - align["A_SHUFFLE"],
              "DeltaAlignment_Q5_Q1": None if q_align[0] is None or q_align[4] is None else q_align[4] - q_align[0],
              "Spearman_quintile_alignment": None if any(x is None for x in q_align) else __import__("experiments.utility_relation.p3_c4_a0_utility_action_alignment.alignment", fromlist=["spearman"]).spearman(range(1, 6), q_align),
              "valid_batch_count": {name: len(values) for name, values in arm_values.items()},
              "invalid_zero_norm_count": invalid,
              "native_grad_norm_statistics": {"mean": float(np.mean(native_norms)), "max": float(np.max(native_norms)), "min": float(np.min(native_norms))},
              "semantic_grad_norm_statistics": {name: {"mean": float(np.mean(values)) if values else None, "max": float(np.max(values)) if values else None} for name, values in semantic_norms.items()},
              "theta_sem": [name for name, _ in parameters], "theta_sem_trace": "frozen CURRENT: optim.Adam(model.parameters())",
              "detach_contract": {"U_cycle": not true_u.requires_grad, "T_rel": not t_rel.requires_grad, "anchor_posterior": True},
              "raw_query_id_sha256": hashlib.sha256(raw_query_ids.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest(),
              "model_state_sha256_before_after": initial}
    result.update(utility_diagnostics(true_u)); result.update(direction_diagnostics(true_u, margins))
    if torch.cuda.is_available() and query_rows.is_cuda:
        result["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(query_rows.device))
        result["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(query_rows.device))
    else: result["peak_memory_allocated_bytes"] = result["peak_memory_reserved_bytes"] = 0
    return result


def _load_views(path: Path, device: torch.device) -> list[torch.Tensor]:
    payload = np.load(path, allow_pickle=False)
    keys = sorted(key for key in payload.files if key.startswith("view_"))
    if not keys: raise ValueError("VIEWS_NPZ_REQUIRES_VIEW_N_KEYS")
    return [torch.as_tensor(payload[key], device=device, dtype=torch.float32) for key in keys]


def run_from_artifacts(dataset: str, training_seed: int, carrier_npz: Path, model_state: Path, views_npz: Path,
                       sparse_anchor_labels: Path, output_root: Path = OUTPUT_ROOT, device: str = "cpu", batch_size: int = 128) -> dict:
    require_cell(dataset, training_seed); hashes = verify_frozen_sources()
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA_UNAVAILABLE")
    target = torch.device(device); carrier = np.load(carrier_npz, allow_pickle=False)
    required = {"query_rows", "anchor_rows", "T_rel", "U_cycle", "query_U_cycle"}
    if not required.issubset(carrier.files): raise ValueError("CARRIER_SCHEMA_MISMATCH")
    views = _load_views(views_npz, target)
    state = torch.load(model_state, map_location=target, weights_only=True)
    model, _ = _new_model_from_state(views, state, target)
    query_rows = torch.as_tensor(carrier["query_rows"], device=target, dtype=torch.long)
    anchor_rows = torch.as_tensor(carrier["anchor_rows"], device=target, dtype=torch.long)
    t_rel = torch.as_tensor(carrier["T_rel"], device=target, dtype=torch.float32).detach()
    u_cycle = torch.as_tensor(carrier["U_cycle"], device=target, dtype=torch.float32).detach()
    query_u = torch.as_tensor(carrier["query_U_cycle"], device=target, dtype=torch.float32).detach()
    labels = torch.as_tensor(np.load(sparse_anchor_labels, allow_pickle=False), device=target, dtype=torch.long)
    raw = torch.as_tensor(carrier["sample_ids"][carrier["query_rows"]], device=target, dtype=torch.long) if "sample_ids" in carrier.files else query_rows
    result = diagnose_cell(model, views, query_rows, anchor_rows, t_rel, u_cycle, query_u, labels, raw, batch_size)
    result.update({"stage": "P3-C4-A0", "dataset": dataset, "training_seed": training_seed, "DIAGNOSTIC_ONLY": True,
                   "full_GT_loaded": False, "formal_clustering_metrics_computed": False, "frozen_source_hashes": hashes})
    destination = output_root / dataset.lower().replace("-", "_") / f"seed{training_seed}" / "diagnostic.json"
    if destination.exists(): raise RuntimeError("OUTPUT_EXISTS_FAIL_CLOSED")
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--carrier-npz", required=True, type=Path)
    parser.add_argument("--model-state", required=True, type=Path)
    parser.add_argument("--views-npz", required=True, type=Path)
    parser.add_argument("--sparse-anchor-labels", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run_from_artifacts(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__": main()
