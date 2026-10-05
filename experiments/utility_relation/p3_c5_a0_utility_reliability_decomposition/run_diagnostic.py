"""Read-only one-cell P3-C5-A0 reliability decomposition runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from method.bone import BONE
from experiments.utility_relation.p3_c4_a0_utility_action_alignment.run_diagnostic import (
    _new_model_from_state, diagnose_cell as c4_diagnose_cell,
)
from .decomposition import (component_alignment, component_metrics, continuous_masks, closure_masks,
                            extract_components, validate_u_parity)
from .protocol import (OUTPUT_ROOT, REQUIRED_CARRIER_FIELDS, require_cell, verify_parent_freeze)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_c4_replication(c4: dict, expected: dict | None) -> tuple[float | None, float | None, bool]:
    delta, rank = c4["DeltaAlignment_Q5_Q1"], c4["Spearman_quintile_alignment"]
    if delta is None or rank is None:
        return delta, rank, False
    if expected is None:
        return delta, rank, True  # CPU synthetic tests have no real-data parent cell.
    passed = (abs(float(delta) - float(expected["DeltaAlignment_Q5_Q1"])) <= 1e-6 and
              abs(float(rank) - float(expected["Spearman_quintile_alignment"])) <= 1e-6)
    return delta, rank, passed


def diagnose_cell(model: torch.nn.Module, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                  anchor_rows: torch.Tensor, t_rel: torch.Tensor, u_cycle: torch.Tensor,
                  query_u_cycle: torch.Tensor, closure: torch.Tensor, conf_gen: torch.Tensor,
                  support_ver: torch.Tensor, anchor_labels: torch.Tensor, raw_query_ids: torch.Tensor,
                  batch_size: int = 128, expected_c4: dict | None = None) -> dict:
    """Run C5 actions with plain losses and no mutable model operations."""
    before = _state_digest(model)
    components = extract_components(query_rows, closure, conf_gen, support_ver, query_u_cycle)
    parity, zero_implies = validate_u_parity(components)
    c1, c0 = closure_masks(components.closure_q)
    # C4's full current-U arm is executed as the required replication control.
    c4 = c4_diagnose_cell(model, full_views, query_rows, anchor_rows, t_rel, u_cycle, query_u_cycle,
                          anchor_labels, raw_query_ids, batch_size)
    c4_delta, c4_rank, c4_pass = _check_c4_replication(c4, expected_c4)
    closure_values, closure_valid, closure_invalid = component_alignment(
        model, full_views, query_rows, anchor_rows, t_rel, anchor_labels, (c1, c0), batch_size)
    conf_masks = continuous_masks(components.conf_q, components.closure_q)
    conf_values, conf_valid, conf_invalid = component_alignment(
        model, full_views, query_rows, anchor_rows, t_rel, anchor_labels, conf_masks, batch_size)
    support_masks = continuous_masks(components.support_q, components.closure_q)
    support_values, support_valid, support_invalid = component_alignment(
        model, full_views, query_rows, anchor_rows, t_rel, anchor_labels, support_masks, batch_size)
    conf_delta, conf_rank = component_metrics(conf_values)
    support_delta, support_rank = component_metrics(support_values)
    after = _state_digest(model)
    unchanged = before == after and all(parameter.grad is None for parameter in model.parameters())
    if not unchanged:
        raise RuntimeError("READ_ONLY_AUTOGRAD_CONTRACT_VIOLATION")
    result = {
        "U_COMPONENT_RECONSTRUCTION_PARITY": parity, "CLOSURE_ZERO_IMPLIES_U_ZERO": zero_implies,
        "A_CLOSURE_1": closure_values[0], "A_CLOSURE_0": closure_values[1],
        "Delta_CLOSURE": None if None in closure_values else closure_values[0] - closure_values[1],
        **{f"A_CONF_Q{i + 1}": value for i, value in enumerate(conf_values)},
        "Delta_CONF_Q5_Q1": conf_delta, "Spearman_CONF_ALIGNMENT": conf_rank,
        **{f"A_SUPPORT_Q{i + 1}": value for i, value in enumerate(support_values)},
        "Delta_SUPPORT_Q5_Q1": support_delta, "Spearman_SUPPORT_ALIGNMENT": support_rank,
        "C4_U_DeltaAlignment_Q5_Q1_replication": c4_delta,
        "C4_U_Spearman_quintile_alignment_replication": c4_rank, "C4_U_REPLICATION_PASS": c4_pass,
        "closure_action_count_0": int(c0.sum().item()), "closure_action_count_1": int(c1.sum().item()),
        "conf_conditioned_action_count": int(sum(mask.sum().item() for mask in conf_masks)),
        "support_conditioned_action_count": int(sum(mask.sum().item() for mask in support_masks)),
        "invalid_zero_norm_count": {"closure": closure_invalid, "conf_gen": conf_invalid, "support_ver": support_invalid},
        "valid_batch_count": {"closure": closure_valid, "conf_gen": conf_valid, "support_ver": support_valid},
        "detach_contract": {"U_cycle": not query_u_cycle.requires_grad, "T_rel": not t_rel.requires_grad,
                            "components": True, "anchor_posterior": True},
        "model_state_sha256_before": before, "model_state_sha256_after": after,
        "model_state_unchanged": unchanged,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(query_rows.device)) if query_rows.is_cuda else None,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(query_rows.device)) if query_rows.is_cuda else None,
    }
    return result


def _load_views(path: Path, device: torch.device) -> list[torch.Tensor]:
    payload = np.load(path, allow_pickle=False)
    keys = sorted(key for key in payload.files if key.startswith("view_"))
    if not keys:
        raise ValueError("VIEWS_NPZ_REQUIRES_VIEW_N_KEYS")
    return [torch.as_tensor(payload[key], device=device, dtype=torch.float32) for key in keys]


def _frozen_c4_cell(dataset: str, seed: int) -> dict:
    from .protocol import PARENT_FREEZE
    cells = json.loads((PARENT_FREEZE / "FULL24_CELLS.json").read_text())
    matches = [item for item in cells if item.get("dataset") == dataset and item.get("training_seed") == seed]
    if len(matches) != 1:
        raise RuntimeError("PARENT_C4_CELL_LOOKUP_FAILED")
    return matches[0]


def run_from_artifacts(dataset: str, training_seed: int, carrier_npz: Path, model_state: Path, views_npz: Path,
                       sparse_anchor_labels: Path, output_root: Path = OUTPUT_ROOT, device: str = "cpu",
                       batch_size: int = 128) -> dict:
    require_cell(dataset, training_seed)
    parent = verify_parent_freeze()
    if device.startswith("cuda"):
        raise RuntimeError("C5_IMPLEMENTATION_CPU_ONLY")
    target = torch.device(device)
    carrier = np.load(carrier_npz, allow_pickle=False)
    if not REQUIRED_CARRIER_FIELDS.issubset(carrier.files):
        raise ValueError("CARRIER_SCHEMA_MISMATCH")
    views = _load_views(views_npz, target)
    state = torch.load(model_state, map_location=target, weights_only=True)
    model, _ = _new_model_from_state(views, state, target)
    query_rows = torch.as_tensor(carrier["query_rows"], device=target, dtype=torch.long)
    anchor_rows = torch.as_tensor(carrier["anchor_rows"], device=target, dtype=torch.long)
    tensors = {name: torch.as_tensor(carrier[name], device=target, dtype=torch.float32).detach()
               for name in ("T_rel", "U_cycle", "query_U_cycle", "closure", "conf_gen", "support_ver")}
    labels = torch.as_tensor(np.load(sparse_anchor_labels, allow_pickle=False), device=target, dtype=torch.long)
    raw = torch.as_tensor(carrier["sample_ids"][carrier["query_rows"]], device=target, dtype=torch.long) if "sample_ids" in carrier.files else query_rows
    result = diagnose_cell(model, views, query_rows, anchor_rows, tensors["T_rel"], tensors["U_cycle"], tensors["query_U_cycle"],
                           tensors["closure"], tensors["conf_gen"], tensors["support_ver"], labels, raw, batch_size,
                           _frozen_c4_cell(dataset, training_seed))
    result.update({"stage": "P3-C5-A0", "dataset": dataset, "training_seed": training_seed,
                   "DIAGNOSTIC_ONLY": True, "full_GT_loaded": False, "formal_clustering_metrics_computed": False,
                   "training_performed": False, "parameter_update_performed": False, "parent_freeze_hashes": parent})
    destination = output_root / dataset.lower().replace("-", "_") / f"seed{training_seed}" / "diagnostic.json"
    if destination.exists():
        raise RuntimeError("OUTPUT_EXISTS_FAIL_CLOSED")
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True); parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--carrier-npz", required=True, type=Path); parser.add_argument("--model-state", required=True, type=Path)
    parser.add_argument("--views-npz", required=True, type=Path); parser.add_argument("--sparse-anchor-labels", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run_from_artifacts(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
