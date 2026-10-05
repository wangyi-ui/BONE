"""Protected seed20 P2-C0 live-replay conflict diagnostic; probes never alter trajectory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from experiments.utility_relation.bone_p2_a0_protocol import PARENT_ARTIFACT
from experiments.utility_relation.bone_p2_a1_relation_action import (
    detached_frozen, relation_action_loss, same_forward_posteriors, semantic_batch_loss, semantic_epoch,
)
from experiments.utility_relation.bone_p2_a1_protocol import SEMANTIC_BATCH_SIZE, array_sha256, semantic_orders
from experiments.utility_relation.bone_p2_c0_conflict_protocol import (
    BIN_NAMES, CARRIER_DIR, DEFAULT_OUTPUT_DIR, EPOCHS, PARENT_TRUE_U_DIR, TRAINING_SEED, bins_sha256, build_u_bins,
)
from experiments.utility_relation.evaluate_bone_p2_a1_pilot import verify_pre_gt_seal
from experiments.utility_relation.materialize_bone_p2_a0_carrier import load_corrupted_views
from experiments.utility_relation.run_bone_p2_a1_pilot import (
    DEVICE, _full_views, _native_epoch, _native_loader, instantiate_initial_model, load_frozen_carrier, model_state_sha256,
)
from utils.cl_loss import con_inter_view, con_reg_loss


@contextmanager
def preserved_rng_state():
    """Restore every live RNG source after a diagnostic probe has completed."""
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


def _structural_losses(model: torch.nn.Module, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct BONE's native inter/reg terms from one test-mode posterior hook forward."""
    observed: list[torch.Tensor] = []
    handle = model.label_fc.register_forward_hook(lambda _m, _i, output: observed.append(output))
    try:
        model(batch, test=True)
    finally:
        handle.remove()
    if len(observed) != 4:
        raise ValueError("structural probe did not capture four view posteriors")
    inter = con_inter_view(observed, metric=model.con_metric, device=model.device)
    regularization = model.reg * con_reg_loss([item.T for item in observed])
    return inter, regularization, inter + regularization


def _semantic_epoch_loss(model, full_views, unlabeled_ids, labeled_ids, relation_targets, action_weights, order: np.ndarray) -> torch.Tensor:
    """Exact P2-A1 TRUE_U semantic epoch objective, accumulated without optimizer stepping."""
    denominator = action_weights.sum()
    if not bool(denominator > 0):
        raise ValueError("zero TRUE_U denominator")
    total = torch.zeros((), device=DEVICE)
    for start in range(0, unlabeled_ids.numel(), SEMANTIC_BATCH_SIZE):
        positions = torch.as_tensor(order[start:start + SEMANTIC_BATCH_SIZE], device=DEVICE)
        query_ids = unlabeled_ids[positions]
        loss, _relations = semantic_batch_loss(
            model, [view[query_ids] for view in full_views], [view[labeled_ids] for view in full_views],
            relation_targets[positions], action_weights[positions], denominator,
        )
        total = total + loss
    return total


def _bin_semantic_loss(model, full_views, unlabeled_ids, labeled_ids, relation_targets, order: np.ndarray, membership: np.ndarray) -> torch.Tensor:
    """Mean raw relation action loss in a frozen bin; U values never weight losses inside a bin."""
    if membership.size == 0:
        raise ValueError("empty U bin has no semantic probe gradient")
    membership_set = set(np.asarray(membership, dtype=np.int64).tolist())
    total = torch.zeros((), device=DEVICE)
    for start in range(0, unlabeled_ids.numel(), SEMANTIC_BATCH_SIZE):
        positions_np = order[start:start + SEMANTIC_BATCH_SIZE]
        positions = torch.as_tensor(positions_np, device=DEVICE)
        query_ids = unlabeled_ids[positions]
        q_query, q_anchor = same_forward_posteriors(
            model, [view[query_ids] for view in full_views], [view[labeled_ids] for view in full_views]
        )
        raw_mask = np.asarray([[int(position) * 6 + action for action in range(6)] for position in positions_np])
        mask = torch.as_tensor(np.isin(raw_mask, list(membership_set)), device=DEVICE)
        if not bool(mask.any()):
            continue
        t_batch = relation_targets[positions]
        unit_weights = torch.ones((len(positions_np), 6), device=DEVICE).detach()
        for query, anchor in zip(q_query, q_anchor):
            _relation, action_loss = relation_action_loss(query, anchor, t_batch, unit_weights)
            total = total + action_loss[mask].sum()
    return total / (4 * membership.size)


def _gradient_map(loss: torch.Tensor, model: torch.nn.Module, retain_graph: bool = False) -> Dict[str, torch.Tensor | None]:
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    grads = torch.autograd.grad(loss, [parameter for _name, parameter in params], allow_unused=True, retain_graph=retain_graph)
    return {name: grad.detach().clone() if grad is not None else None for (name, _parameter), grad in zip(params, grads)}


def _cosine(left: Dict[str, torch.Tensor | None], right: Dict[str, torch.Tensor | None]) -> tuple[float, int, int]:
    shared = [name for name in left if left[name] is not None and right.get(name) is not None]
    if not shared:
        raise ValueError("zero shared-gradient parameter support")
    a, b = torch.cat([left[name].reshape(-1) for name in shared]), torch.cat([right[name].reshape(-1) for name in shared])
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if not bool(norm_a > 0) or not bool(norm_b > 0):
        raise ValueError("zero gradient norm")
    return float(torch.dot(a, b) / (norm_a * norm_b)), len(shared), int(a.numel())


def probe_epoch(model, full_views, unlabeled_ids, labeled_ids, relation_targets, action_weights, order, native_batch, bins) -> Dict[str, float]:
    """Measure isolated gradients from three identical epoch-start copies without optimizer steps."""
    before_live = model_state_sha256(model)
    with preserved_rng_state():
        probe_sem, probe_inter, probe_native = copy.deepcopy(model), copy.deepcopy(model), copy.deepcopy(model)
        if any(model_state_sha256(probe) != before_live for probe in (probe_sem, probe_inter, probe_native)):
            raise RuntimeError("probe models do not start from live epoch state")
        g_sem = _gradient_map(_semantic_epoch_loss(probe_sem, full_views, unlabeled_ids, labeled_ids, relation_targets, action_weights, order), probe_sem)
        inter_loss, _reg_unused, _native_unused = _structural_losses(probe_inter, native_batch)
        g_inter = _gradient_map(inter_loss, probe_inter)
        _inter_native, reg_loss, native_loss = _structural_losses(probe_native, native_batch)
        g_reg = _gradient_map(reg_loss, probe_native, retain_graph=True)
        g_native = _gradient_map(native_loss, probe_native)
        row = {}
        for label, gradient in (("inter", g_inter), ("native", g_native), ("reg", g_reg)):
            cosine, count, elements = _cosine(g_sem, gradient)
            row[f"cos_sem_{label}"] = cosine
            row["shared_parameter_count"] = count
            row["shared_element_count"] = elements
        row["norm_sem_true"] = float(torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in g_sem.values() if value is not None])))
        row["norm_inter"] = float(torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in g_inter.values() if value is not None])))
        row["norm_native"] = float(torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in g_native.values() if value is not None])))
        row["norm_reg"] = float(torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in g_reg.values() if value is not None])))
        for name in BIN_NAMES:
            probe_bin = copy.deepcopy(model)
            g_bin = _gradient_map(_bin_semantic_loss(probe_bin, full_views, unlabeled_ids, labeled_ids, relation_targets, order, bins[name]), probe_bin)
            rho, _count, _elements = _cosine(g_bin, g_inter)
            row[f"rho_{name}"] = rho
            row[f"norm_sem_{name}"] = float(torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in g_bin.values() if value is not None])))
    if model_state_sha256(model) != before_live:
        raise RuntimeError("probe changed live model")
    row["high_minus_low"] = row["rho_HIGH"] - row["rho_LOW"]
    row["global_inter_conflict"] = int(row["cos_sem_inter"] < 0)
    row["low_conflict"] = int(row["rho_LOW"] < 0)
    row["high_conflict"] = int(row["rho_HIGH"] < 0)
    return row


def run_diagnostic(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Replay TRUE_U trajectory while collecting isolated probes; CLI protects this formal operation."""
    verify_pre_gt_seal(PARENT_TRUE_U_DIR)
    carrier = load_frozen_carrier(CARRIER_DIR, TRAINING_SEED)
    model = instantiate_initial_model(CARRIER_DIR, TRAINING_SEED)
    full_views = _full_views(load_corrupted_views(PARENT_ARTIFACT))
    relation_targets, action_weights = detached_frozen(carrier.t_rel, DEVICE), detached_frozen(carrier.u_cycle, DEVICE)
    unlabeled_ids = torch.as_tensor(carrier.unlabeled_ids, dtype=torch.long, device=DEVICE)
    labeled_ids = torch.as_tensor(carrier.labeled_ids, dtype=torch.long, device=DEVICE)
    orders, bins = semantic_orders(TRAINING_SEED), build_u_bins(carrier.u_cycle)
    semantic_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    native_optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    live_loader, probe_loader = _native_loader(load_corrupted_views(PARENT_ARTIFACT), TRAINING_SEED), _native_loader(load_corrupted_views(PARENT_ARTIFACT), TRAINING_SEED)
    probe_batch, _dummy, _ids = next(iter(probe_loader))
    rows = []
    for epoch in range(EPOCHS):
        row = probe_epoch(model, full_views, unlabeled_ids, labeled_ids, relation_targets, action_weights, orders[epoch], probe_batch, bins)
        row["epoch"] = epoch
        semantic_epoch(model, semantic_optimizer, full_views, unlabeled_ids, labeled_ids, relation_targets, action_weights, orders[epoch])
        _native_epoch(model, native_optimizer, live_loader)
        rows.append(row)
    with torch.no_grad():
        predictions = model(full_views, test=True)["label"].cpu().numpy()
    with np.load(PARENT_TRUE_U_DIR / "final_predictions.npz", allow_pickle=False) as parent:
        if not np.array_equal(predictions, parent["predictions"]):
            raise RuntimeError("P2-C0 replay prediction does not exactly match frozen TRUE_U")
    payload = {"epochs": rows, "bin_counts": {name: int(values.size) for name, values in bins.items()}, "bin_membership_sha256": bins_sha256(bins), "final_prediction_parity": True}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "p2_c0_conflict_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_path = output_dir / "p2_c0_conflict_seal.json"
    seal_path.write_text(json.dumps({"sha256": {"metrics": hashlib.sha256(metrics_path.read_bytes()).hexdigest()}}, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="run the protected 20-epoch P2-C0 diagnostic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to start P2-C0 diagnostic without --execute")
    print(json.dumps(run_diagnostic(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
