"""Opt-in one-dataset/one-seed P3-C0 formal runner; it never reads full GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import scipy.io
import torch
from torch import optim
from torch.utils.data import DataLoader

from experiments.utility_relation import bone_p3_a4_corruption_protocol as corruption_protocol
from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as protocol
from method.amv import amv
from method.bone import BONE
from utils.seed_util import set_seed


def _load_frozen_inputs(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    split_path, mask_path = protocol.artifact_paths(dataset_name)
    if not split_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError("missing frozen P3-A3 sparse split or P3-A4 corruption mask")
    with np.load(split_path, allow_pickle=False) as archive:
        if set(archive.files) != {"labeled_ids", "labeled_targets"}:
            raise ValueError("invalid frozen sparse split schema")
        labels = np.asarray(archive["labeled_ids"], dtype=np.int64)
        targets = np.asarray(archive["labeled_targets"], dtype=np.int64)
    with np.load(mask_path, allow_pickle=False) as archive:
        mask_key = "corruption_mask"
        if mask_key not in archive or "raw_row_ids" not in archive:
            raise ValueError("invalid frozen corruption mask schema")
        mask, raw_ids = np.asarray(archive[mask_key]), np.asarray(archive["raw_row_ids"], dtype=np.int64)
    spec = protocol.DATASET_SPECS[dataset_name]
    k = int(targets.max()) + 1
    if labels.shape != targets.shape or labels.size != 2 * k or np.unique(labels).size != labels.size:
        raise ValueError("frozen split does not contain exactly two unique anchors per class")
    if not np.array_equal(np.bincount(targets, minlength=k), np.full(k, 2)):
        raise ValueError("frozen split class counts are invalid")
    if mask.dtype != np.bool_ or mask.shape != (spec.n, spec.v) or not np.array_equal(raw_ids, np.arange(spec.n)):
        raise ValueError("frozen mask raw-ID shape is invalid")
    corruption_protocol.validate_balanced_half_view_mask(mask, dataset_name)
    return labels, targets, mask, {"p3_a3_split": protocol.sha256_file(split_path), "p3_a4_mask": protocol.sha256_file(mask_path)}


def load_raw_views(dataset_name: str) -> list[np.ndarray]:
    """Read X1..XV only; Y is intentionally unavailable in this training module."""
    spec = protocol.DATASET_SPECS[dataset_name]
    keys = tuple(f"X{i}" for i in range(1, spec.v + 1))
    mat = scipy.io.loadmat(protocol.DATA_ROOT / f"{dataset_name}.mat", variable_names=keys)
    views: list[np.ndarray] = []
    for key, dim in zip(keys, spec.dims):
        stored = np.asarray(mat[key])
        if stored.shape == (dim, spec.n):
            stored = stored.T
        if stored.shape != (spec.n, dim):
            raise ValueError(f"{dataset_name}:{key} does not match frozen native dimensions")
        views.append(np.ascontiguousarray(stored, dtype=np.float32))
    return views


def _container(views: Sequence[np.ndarray], k: int) -> SimpleNamespace:
    spec_dims = [int(view.shape[1]) for view in views]
    return SimpleNamespace(data=[view.T for view in views], m_list=spec_dims, v=len(views), k=k)


def _raw_dataset(views: Sequence[np.ndarray], adapter: b0.SampleIdAdapter) -> b0.RawIdDataset:
    return b0.RawIdDataset([view.T for view in views], adapter)


def _native_loader(views: Sequence[np.ndarray], adapter: b0.SampleIdAdapter, seed: int, batch_size: int) -> DataLoader:
    return DataLoader(_raw_dataset(views, adapter), batch_size=batch_size, generator=torch.Generator().manual_seed(seed),
                      num_workers=0, shuffle=True, drop_last=True)


def _native_epoch(model: BONE, optimizer: optim.Optimizer, loader: DataLoader, device: torch.device) -> tuple[float, str]:
    """One untouched BONE epoch; batch tensors are [B,D_v], IDs are audit-only."""
    observed, losses = [], []
    model.train()
    for data_batch, sample_ids in loader:
        observed.append(sample_ids.detach().cpu().numpy().astype(np.int64))
        optimizer.zero_grad(set_to_none=True)
        output = model([item.to(device=device, dtype=torch.float32) for item in data_batch])
        loss = output["loss"]
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite native BONE loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError("native epoch did not complete with finite losses")
    return float(np.mean(losses)), protocol.sha256_file_from_array(np.concatenate(observed))


def _state_hash(model: torch.nn.Module) -> str:
    return protocol.state_dict_sha256(model.state_dict())


def _load_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _new_model(views: Sequence[np.ndarray], k: int, spec: protocol.DatasetSpec, seed: int, device: torch.device,
               anchors: list[np.ndarray] | None = None) -> BONE:
    return BONE(m_list=[view.shape[1] for view in views], k=k, a=spec.n_anchor, v=spec.v, anchor_init=anchors,
                layer=spec.layer, lamb=0.1, seed=seed, device=device).to(device)


def _semantic_epoch(model: BONE, optimizer: optim.Optimizer, full_views: Sequence[torch.Tensor], query_rows: torch.Tensor,
                    anchor_rows: torch.Tensor, t_rel: torch.Tensor, u_cycle: torch.Tensor, order: np.ndarray,
                    k: int) -> float:
    """Accepted P2-A1B Phase A. q=[B,K], anchors=[L,K], T=[B,L,S], U=[B,S]."""
    if not np.array_equal(np.sort(order), np.arange(query_rows.numel())):
        raise ValueError("semantic query order must be a complete permutation")
    denominator = u_cycle.sum()
    if not bool(denominator > 0):
        raise ValueError("full utility mass must be positive")
    optimizer.zero_grad(set_to_none=True)
    final_loss = 0.0
    for start in range(0, query_rows.numel(), protocol.SEMANTIC_BATCH_SIZE):
        positions = torch.as_tensor(order[start:start + protocol.SEMANTIC_BATCH_SIZE], device=query_rows.device)
        selected_rows = query_rows[positions]
        observed: list[torch.Tensor] = []
        def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed.append(output)
        handle = model.label_fc.register_forward_hook(capture)
        try:
            # Each input is [B+L,D_v]; hook outputs are [B+L,K] for every view.
            model([torch.cat((view[selected_rows], view[anchor_rows]), dim=0) for view in full_views], test=True)
        finally:
            handle.remove()
        batch, labeled, actions = positions.numel(), anchor_rows.numel(), u_cycle.shape[1]
        if len(observed) != len(full_views) or any(value.shape != (batch + labeled, k) for value in observed):
            raise ValueError("semantic posterior hook shape mismatch")
        batch_t, batch_u = t_rel[positions], u_cycle[positions]
        view_numerators = []
        for posterior in observed:
            query, anchor = posterior[:batch], posterior[batch:].detach()
            # relation [B,L] = q_i^v dot stopgrad(q_l^v), with no projection/temperature.
            relation = query @ anchor.T
            safe = relation.clamp(torch.finfo(relation.dtype).eps, 1.0 - torch.finfo(relation.dtype).eps)
            positive = -(batch_t * torch.log(safe.unsqueeze(2))).sum(dim=1) / (2 * 2)
            negative = -((1.0 - batch_t) * torch.log1p(-safe.unsqueeze(2))).sum(dim=1) / (2 * 2 * (k - 1))
            if positive.shape != (batch, actions) or negative.shape != (batch, actions):
                raise ValueError("semantic action tensor shape mismatch")
            view_numerators.append((batch_u * (positive + negative)).sum())
        loss = sum(view_numerators) / (len(full_views) * denominator)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite semantic relation loss")
        loss.backward()
        final_loss += float(loss.detach().cpu())
    optimizer.step()  # Exactly one semantic optimizer step per refinement epoch.
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("semantic gradients leaked into native Phase B")
    return final_loss


def _predict(model: BONE, views: Sequence[np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model([torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views], test=True)["label"].cpu().numpy().astype(np.int64)


def _save_state(model: BONE, path: Path) -> str:
    torch.save(model.state_dict(), path)
    return protocol.sha256_file(path)


def run_formal_dataset_seed(dataset_name: str, training_seed: int, output_root: Path = protocol.OUTPUT_ROOT,
                            query_chunk_size: int = protocol.QUERY_CHUNK_SIZE) -> dict[str, Path]:
    """Run exactly one dataset/seed formal pair. Full labels are never loaded here."""
    if dataset_name not in protocol.FORMAL_DATASETS or training_seed not in protocol.FORMAL_SEEDS:
        raise ValueError("unregistered dataset or seed")
    if not torch.cuda.is_available():
        raise RuntimeError("FAIL-CLOSED: P3-C0 requires CUDA device 0")
    protocol.verify_native_hashes()
    protocol.verify_parent_freeze_hashes()
    torch.cuda.set_device(0)
    device, spec = torch.device("cuda:0"), protocol.DATASET_SPECS[dataset_name]
    labels, targets, mask, parent_hashes = _load_frozen_inputs(dataset_name)
    raw_views = load_raw_views(dataset_name)
    views, adapter = b0.preprocess_with_raw_ids(raw_views, mask)
    if not np.array_equal(np.sort(adapter.sample_ids), np.arange(spec.n)) or np.unique(adapter.row_positions(labels)).size != labels.size:
        raise ValueError("raw sample-ID or sparse-anchor mapping mismatch")
    k = int(targets.max()) + 1
    output_dir = Path(output_root) / protocol.dataset_slug(dataset_name) / f"seed{training_seed}"
    carrier_dir, base_dir, ours_dir, seal_dir = (output_dir / name for name in ("carrier", "base", "ours", "seal"))
    for path in (carrier_dir, base_dir, ours_dir, seal_dir): path.mkdir(parents=True, exist_ok=True)
    set_seed(training_seed)
    anchors = amv(_container(views, k), m=spec.n_anchor, ret_anchor=True, max_iter=100, seed=training_seed, info=True, info_iter=False)
    carrier_model = _new_model(views, k, spec, training_seed, device, anchors)
    carrier_optimizer = optim.Adam(carrier_model.parameters(), lr=protocol.NATIVE_LR)
    carrier_loader = _native_loader(views, adapter, training_seed, spec.batch_size)
    carrier_losses = [_native_epoch(carrier_model, carrier_optimizer, carrier_loader, device)[0] for _ in range(protocol.CARRIER_EPOCHS)]
    if not np.isfinite(carrier_losses).all(): raise FloatingPointError("non-finite carrier training")
    state_path = carrier_dir / "carrier_model_state.pt"
    carrier_state_file_hash = _save_state(carrier_model, state_path)
    carrier_state_hash = _state_hash(carrier_model)
    q_views = b0.extract_q_views(carrier_model, views, device=device)
    payload = b0.build_generic_carrier(q_views, adapter.sample_ids, labels, targets, query_chunk_size=query_chunk_size)
    for name, shape in protocol.expected_shapes(dataset_name, k).items():
        if payload[name].shape != shape or not np.isfinite(payload[name]).all(): raise ValueError(f"invalid frozen formal {name}")
    carrier_npz = carrier_dir / "carrier_pre_gt.npz"
    np.savez_compressed(carrier_npz, **payload)
    (carrier_dir / "carrier_pre_gt_audit.json").write_text(json.dumps({"training_seed": training_seed, "native_epochs": 100,
        "GT_loaded_before_seal": False, "carrier_state_sha256": carrier_state_hash, "carrier_file_sha256": carrier_state_file_hash,
        "parent_hashes": parent_hashes, "query_chunk_size": query_chunk_size}, indent=2, sort_keys=True) + "\n")
    def load_branch() -> BONE:
        model = _new_model(views, k, spec, training_seed, device)
        model.load_state_dict(_load_state(state_path, device), strict=True)
        return model
    base_model, ours_model = load_branch(), load_branch()
    base_initial, ours_initial = _state_hash(base_model), _state_hash(ours_model)
    if base_initial != carrier_state_hash or ours_initial != carrier_state_hash or base_initial != ours_initial:
        raise RuntimeError("BASE/OURS shared carrier initialization mismatch")
    base_optimizer, ours_native_optimizer = optim.Adam(base_model.parameters(), lr=protocol.NATIVE_LR), optim.Adam(ours_model.parameters(), lr=protocol.NATIVE_LR)
    base_loader, ours_loader = _native_loader(views, adapter, training_seed, spec.batch_size), _native_loader(views, adapter, training_seed, spec.batch_size)
    base_orders, ours_orders, semantic_losses = [], [], []
    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views]
    query_rows = torch.as_tensor(payload["query_rows"], dtype=torch.long, device=device)
    anchor_rows = torch.as_tensor(payload["anchor_rows"], dtype=torch.long, device=device)
    t_rel = torch.as_tensor(payload["T_rel"], dtype=torch.float32, device=device).detach()
    u_cycle = torch.as_tensor(payload["query_U_cycle"], dtype=torch.float32, device=device).detach()
    semantic_optimizer = optim.Adam(ours_model.parameters(), lr=protocol.SEMANTIC_LR)
    for order in protocol.semantic_orders(query_rows.numel(), training_seed):
        semantic_losses.append(_semantic_epoch(ours_model, semantic_optimizer, full_views, query_rows, anchor_rows, t_rel, u_cycle, order, k))
        ours_loss, ours_order = _native_epoch(ours_model, ours_native_optimizer, ours_loader, device)
        base_loss, base_order = _native_epoch(base_model, base_optimizer, base_loader, device)
        if not np.isfinite((ours_loss, base_loss)).all(): raise FloatingPointError("non-finite refinement loss")
        ours_orders.append(ours_order); base_orders.append(base_order)
    if base_orders != ours_orders or not np.isfinite(semantic_losses).all(): raise RuntimeError("paired native schedule or semantic loss failure")
    base_state, ours_state = base_dir / "final_model_state.pt", ours_dir / "final_model_state.pt"
    _save_state(base_model, base_state); _save_state(ours_model, ours_state)
    for path, model in ((base_dir / "pre_gt_predictions.npz", base_model), (ours_dir / "pre_gt_predictions.npz", ours_model)):
        np.savez_compressed(path, sample_ids=adapter.sample_ids, predictions=_predict(model, views, device))
    audit = {"dataset": dataset_name, "training_seed": training_seed, "FORMAL_RESULT": True, "ENGINEERING_ONLY": False,
             "GT_loaded_before_seal": False, "full_gt_used_for_training": False, "full_gt_used_for_model_selection": False,
             "full_gt_used_for_hyperparameter_selection": False, "carrier_state_sha256": carrier_state_hash,
             "initial_BASE_state_sha256": base_initial, "initial_OURS_state_sha256": ours_initial,
             "base_native_epochs": 20, "ours_semantic_epochs": 20, "ours_native_phase_b_epochs": 20,
             "semantic_lr": protocol.SEMANTIC_LR, "native_lr": protocol.NATIVE_LR, "native_order_hashes": base_orders,
             "semantic_schedule_sha256": protocol.sha256_file_from_array(protocol.semantic_orders(query_rows.numel(), training_seed)),
             "parent_hashes": parent_hashes, "native_hashes": protocol.verify_native_hashes()}
    audit_path = seal_dir / "pre_gt_audit.json"; audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    split_path, mask_path = protocol.artifact_paths(dataset_name)
    targets = {"carrier_state": state_path, "carrier_npz": carrier_npz, "base_state": base_state, "ours_state": ours_state,
               "base_predictions": base_dir / "pre_gt_predictions.npz", "ours_predictions": ours_dir / "pre_gt_predictions.npz", "audit": audit_path,
               "runner_source": Path(__file__), "protocol_source": Path(protocol.__file__), "p3_a1_source": protocol.PARENT_FREEZE_PATHS["p3_a1_source"],
               "p3_a3_source": protocol.PARENT_FREEZE_PATHS["p3_a3_source"], "p3_a4_source": protocol.PARENT_FREEZE_PATHS["p3_a4_source"],
               "p3_b0_source": protocol.PARENT_FREEZE_PATHS["p3_b0_source"], "p3_b1_runner": protocol.PARENT_FREEZE_PATHS["p3_b1_runner"],
               "p3_a3_split": split_path, "p3_a4_mask": mask_path}
    seal_path = seal_dir / "pre_gt_seal.json"; seal_path.write_text(json.dumps({"sha256": {name: protocol.sha256_file(path) for name, path in targets.items()}}, indent=2, sort_keys=True) + "\n")
    return {"root": output_dir, "carrier": carrier_dir, "base": base_dir, "ours": ours_dir, "seal": seal_dir}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.FORMAL_DATASETS, required=True)
    parser.add_argument("--training-seed", type=int, choices=protocol.FORMAL_SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--query-chunk-size", type=int, default=protocol.QUERY_CHUNK_SIZE)
    parser.add_argument("--execute", action="store_true", help="run one formal dataset/seed pair")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute: raise SystemExit("refusing to launch formal GPU training without --execute")
    print(json.dumps({key: str(value) for key, value in run_formal_dataset_seed(args.dataset, args.training_seed, args.output_root, args.query_chunk_size).items()}, indent=2))


if __name__ == "__main__": main()
