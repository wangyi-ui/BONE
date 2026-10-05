"""Formal MSRC development-carrier adapter for both frozen dev seeds.

This removes the engineering-smoke seed11 execution restriction from the
formal route.  It is not a scientific protocol change: all inputs, model
configuration, training primitives, and carrier materialization semantics are
the same frozen MSRC semantics used by ``msrc_adapter.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from method.amv import amv
from utils.seed_util import set_seed

from . import msrc_adapter, protocol, sparse_cv
from .carrier_adapter import _carrier_arrays
from .parent_integrity import sha256_file


MSRC_SPEC = msrc_adapter.MSRC_SPEC
ACTIVE_PROTOCOL = "P3-C1-B1-A1"


def validate_formal_msrc_request(dev_seed: int, device: str) -> None:
    if isinstance(dev_seed, bool) or dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("UNREGISTERED_DEV_SEED")
    if device != "cuda:0":
        raise ValueError("FORMAL_MSRC_CARRIER_REQUIRES_CUDA_0")


def scientific_configuration() -> dict:
    """Expose frozen seed-independent configuration for parity auditing."""
    _, _, split_path = msrc_adapter.p3_b0_materializer.load_frozen_sparse_split("MSRC")
    _, _, mask_path = msrc_adapter.p3_b0_materializer.load_frozen_corruption_mask("MSRC")
    return {
        "dataset": "MSRC",
        "active_protocol": ACTIVE_PROTOCOL,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "native_lr": msrc_adapter.p2_carrier.LR,
        "batch_size": MSRC_SPEC.batch_size,
        "sample_count": MSRC_SPEC.n,
        "view_count": MSRC_SPEC.v,
        "feature_dims": tuple(MSRC_SPEC.dims),
        "n_anchor": MSRC_SPEC.n_anchor,
        "layer": MSRC_SPEC.layer,
        "cluster_count": msrc_adapter.p1_protocol.K,
        "action_count": action_protocol.expected_action_count(MSRC_SPEC.v),
        "sparse_label_path": str(split_path),
        "sparse_label_sha256": sha256_file(split_path),
        "corruption_path": str(mask_path),
        "corruption_sha256": sha256_file(mask_path),
        "snr_db": msrc_adapter.p1_protocol.TARGET_SNR_DB,
        "corruption_seed": msrc_adapter.p1_protocol.CORRUPTION_SEED,
    }


def validate_generated_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "q_views", "p_gen", "p_ver", "U_cycle", "generator_actions",
        "verifier_actions", "sample_ids", "labeled_ids", "labeled_targets",
    }
    if set(arrays) < required:
        raise ValueError("MISSING_FORMAL_MSRC_CARRIER_KEYS")
    q, p_gen, p_ver, utility = (
        np.asarray(arrays["q_views"]), np.asarray(arrays["p_gen"]),
        np.asarray(arrays["p_ver"]), np.asarray(arrays["U_cycle"]),
    )
    if q.shape != (MSRC_SPEC.n, MSRC_SPEC.v, msrc_adapter.p1_protocol.K):
        raise ValueError("INVALID_FORMAL_MSRC_Q_SHAPE")
    if p_gen.shape != p_ver.shape or p_gen.shape[:1] != (MSRC_SPEC.n,) or p_gen.shape[2] != q.shape[2]:
        raise ValueError("INVALID_FORMAL_MSRC_POSTERIOR_SHAPE")
    if utility.shape != p_gen.shape[:2]:
        raise ValueError("INVALID_FORMAL_MSRC_U_SHAPE")
    if not all(np.isfinite(value).all() for value in (q, p_gen, p_ver, utility)):
        raise ValueError("NONFINITE_FORMAL_MSRC_CARRIER")
    y_gen, y_ver = np.argmax(p_gen, axis=2), np.argmax(p_ver, axis=2)
    conf_gen = np.max(p_gen, axis=2)
    support_ver = np.take_along_axis(p_ver, y_gen[..., None], axis=2)[..., 0]
    expected_u = (y_gen == y_ver) * np.sqrt(conf_gen * support_ver)
    if not np.allclose(utility, expected_u, rtol=1e-6, atol=1e-7):
        raise ValueError("U_FORMULA_PARITY_FAILURE")
    if arrays["generator_actions"].shape[0] != action_protocol.expected_action_count(MSRC_SPEC.v):
        raise ValueError("INVALID_FORMAL_MSRC_ACTION_COUNT")


def build_formal_msrc_development_carrier(
    dev_seed: int,
    device: str,
    output_dir: str | Path,
) -> dict:
    """Train one formal MSRC carrier using frozen lower-level primitives."""
    validate_formal_msrc_request(dev_seed, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.cuda.set_device(0)
    torch_device = torch.device(device)
    views, labeled_ids, labeled_targets, split_path, mask_path = msrc_adapter._load_frozen_msrc_inputs()
    adapter = b0.SampleIdAdapter(np.arange(MSRC_SPEC.n, dtype=np.int64))
    folds = sparse_cv.build_sparse_label_folds(
        labeled_ids, labeled_targets, expected_class_ids=range(msrc_adapter.p1_protocol.K)
    )

    set_seed(dev_seed)
    native_data = msrc_adapter.p2_carrier._native_data(views)
    anchors = amv(
        native_data, m=MSRC_SPEC.n_anchor, ret_anchor=True, max_iter=100,
        seed=dev_seed, info=True, info_iter=False,
    )
    model = c0._new_model(
        views, msrc_adapter.p1_protocol.K, MSRC_SPEC, dev_seed, torch_device, anchors
    )
    optimizer = optim.Adam(model.parameters(), lr=msrc_adapter.p2_carrier.LR)
    loader = c0._native_loader(views, adapter, dev_seed, MSRC_SPEC.batch_size)
    losses, order_hashes = [], []
    for epoch in range(protocol.CARRIER_EPOCHS):
        loss, order_hash = c0._native_epoch(model, optimizer, loader, torch_device)
        losses.append(loss); order_hashes.append(order_hash)
        if (epoch + 1) % 10 == 0:
            print(f"formal MSRC carrier epoch={epoch + 1}/{protocol.CARRIER_EPOCHS} loss={loss:.8f}", flush=True)
    if len(losses) != protocol.CARRIER_EPOCHS or not np.isfinite(losses).all():
        raise RuntimeError("FORMAL_MSRC_CARRIER_EPOCH_FAILURE")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "carrier_model_state.pt"
    state_file_sha256 = c0._save_state(model, state_path)
    state_sha256 = c0._state_hash(model)
    q_views = b0.extract_q_views(model, views, device=torch_device)
    arrays = _carrier_arrays(q_views)
    arrays.update({
        "sample_ids": adapter.sample_ids.copy(),
        "labeled_ids": labeled_ids.copy(),
        "labeled_targets": labeled_targets.copy(),
    })
    validate_generated_arrays(arrays)
    carrier_path = output_dir / "carrier_pre_gt.npz"
    np.savez_compressed(carrier_path, **arrays)

    utility_masses = arrays["U_cycle"].sum(axis=1)
    labeled_rows = adapter.row_positions(labeled_ids)
    audit = {
        "dataset": "MSRC", "dev_seed": dev_seed,
        "active_protocol": ACTIVE_PROTOCOL,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "completed_epochs": len(losses),
        "native_lr": msrc_adapter.p2_carrier.LR,
        "batch_size": MSRC_SPEC.batch_size,
        "carrier_state_sha256": state_sha256,
        "carrier_state_file_sha256": state_file_sha256,
        "native_order_hashes": order_hashes,
        "corruption_sha256": sha256_file(mask_path),
        "sparse_label_sha256": sha256_file(split_path),
        "view_count": MSRC_SPEC.v,
        "cluster_count": msrc_adapter.p1_protocol.K,
        "action_count": int(arrays["generator_actions"].shape[0]),
        "sparse_cv_split_hash": sparse_cv.split_audit_hash(folds),
        "u_shape": list(arrays["U_cycle"].shape),
        "u_finite": True,
        "u_positive_count": int(np.count_nonzero(arrays["U_cycle"] > 0)),
        "u_zero_count": int(np.count_nonzero(arrays["U_cycle"] == 0)),
        "zero_mass_sample_count": int(np.count_nonzero(utility_masses == 0)),
        "zero_mass_labeled_count": int(np.count_nonzero(utility_masses[labeled_rows] == 0)),
        "u_formula_parity": True,
        "full_gt_loaded": False,
        "full_gt_used": False,
        "formal_gt_metrics_computed": False,
        "optuna_study_created": False,
        "refinement_run": False,
    }
    audit_path = output_dir / "carrier_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model": model, "views": views, "adapter": adapter, "spec": MSRC_SPEC,
        "k": msrc_adapter.p1_protocol.K, "state_path": state_path,
        "carrier_path": carrier_path, "audit_path": audit_path,
        "state_sha256": state_sha256, "folds": folds, "arrays": arrays,
    }
