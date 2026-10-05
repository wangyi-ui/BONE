"""Thin read-only MSRC input adapter for the frozen generic B1 GPU bridge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

from experiments.weak_quality import bone_p1_msrc_protocol as p1_protocol
from experiments.utility_relation import bone_p2_a0_protocol as p2_protocol
from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import bone_p3_a4_corruption_protocol as corruption_protocol
from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import materialize_bone_p2_a0_carrier as p2_carrier
from experiments.utility_relation import materialize_bone_p3_a4_corruptions as p3_a4_materializer
from experiments.utility_relation import materialize_bone_p3_b0_carrier as p3_b0_materializer
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from method.amv import amv
from utils.seed_util import set_seed

from . import protocol, sparse_cv
from .carrier_adapter import _carrier_arrays
from .study_schema import CandidateConfig


MSRC_SPEC = SimpleNamespace(
    n=p1_protocol.N,
    v=p1_protocol.V,
    dims=p1_protocol.FEATURE_DIMS,
    batch_size=p2_carrier.BATCH_SIZE,
    n_anchor=p2_carrier.N_ANCHOR,
    layer=p2_carrier.LAYER,
)


def validate_msrc_request(dataset: str, dev_seed: int, semantic_lr: float,
                          semantic_exposure: int, device: str) -> CandidateConfig:
    if dataset != "MSRC" or dataset not in protocol.DATASETS:
        raise ValueError("ONLY_MSRC_ADAPTER_SMOKE_IS_ALLOWED")
    if dev_seed != protocol.DEV_SEEDS[0]:
        raise ValueError("ONLY_DEV_SEED11_IS_ALLOWED")
    if device != "cuda:0":
        raise ValueError("MSRC_SMOKE_REQUIRES_CUDA_0")
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    if (candidate.semantic_lr, candidate.semantic_exposure) != (1e-5, 5):
        raise ValueError("ONLY_FIXED_MSRC_SMOKE_CANDIDATE_IS_ALLOWED")
    return candidate


def expected_msrc_carrier_dir(output_root: str | Path) -> Path:
    return Path(output_root) / "msrc" / f"dev_seed{protocol.DEV_SEEDS[0]}" / "carrier"


def _load_frozen_msrc_inputs():
    if p3_a4_materializer.sha256_file(p3_a4_materializer.P1_ARTIFACT) != p3_a4_materializer.P1_ARTIFACT_SHA256:
        raise RuntimeError("FROZEN_MSRC_PARENT_ARTIFACT_HASH_MISMATCH")
    views = p2_carrier.load_corrupted_views(p3_a4_materializer.P1_ARTIFACT)
    labeled_ids, labeled_targets, split_path = p3_b0_materializer.load_frozen_sparse_split("MSRC")
    mask, raw_ids, mask_path = p3_b0_materializer.load_frozen_corruption_mask("MSRC")
    with np.load(p3_a4_materializer.P1_ARTIFACT, allow_pickle=False) as archive:
        parent_mask = np.asarray(archive["corruption_mask"], dtype=bool)
        parent_raw_ids = np.asarray(archive["row_ids"], dtype=np.int64)
    if not np.array_equal(mask, parent_mask) or not np.array_equal(raw_ids, parent_raw_ids):
        raise RuntimeError("MSRC_P1_P3_CORRUPTION_PARITY_FAILURE")
    p1_protocol.validate_mask(mask)
    corruption_protocol.validate_balanced_half_view_mask(mask, "MSRC")
    action_protocol.verify_msrc_action_order()
    return views, labeled_ids, labeled_targets, split_path, mask_path


def build_msrc_development_carrier(dev_seed: int, device: str, output_dir: str | Path) -> dict:
    """Train one MSRC seed11 carrier from frozen, already-corrupted features."""
    validate_msrc_request("MSRC", dev_seed, 1e-5, 5, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.cuda.set_device(0)
    torch_device = torch.device(device)
    views, labeled_ids, labeled_targets, split_path, mask_path = _load_frozen_msrc_inputs()
    adapter = b0.SampleIdAdapter(np.arange(MSRC_SPEC.n, dtype=np.int64))
    folds = sparse_cv.build_sparse_label_folds(
        labeled_ids, labeled_targets, expected_class_ids=range(p1_protocol.K)
    )

    set_seed(dev_seed)
    native_data = p2_carrier._native_data(views)
    anchors = amv(native_data, m=MSRC_SPEC.n_anchor, ret_anchor=True, max_iter=100,
                  seed=dev_seed, info=True, info_iter=False)
    model = c0._new_model(views, p1_protocol.K, MSRC_SPEC, dev_seed, torch_device, anchors)
    optimizer = optim.Adam(model.parameters(), lr=p2_carrier.LR)
    loader = c0._native_loader(views, adapter, dev_seed, MSRC_SPEC.batch_size)
    losses, order_hashes = [], []
    for epoch in range(protocol.CARRIER_EPOCHS):
        loss, order_hash = c0._native_epoch(model, optimizer, loader, torch_device)
        losses.append(loss)
        order_hashes.append(order_hash)
        if (epoch + 1) % 10 == 0:
            print(f"MSRC carrier epoch={epoch + 1}/{protocol.CARRIER_EPOCHS} loss={loss:.8f}", flush=True)
    if len(losses) != protocol.CARRIER_EPOCHS or not np.isfinite(losses).all():
        raise RuntimeError("MSRC_CARRIER_EPOCH_FAILURE")

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
    if arrays["generator_actions"].shape[0] != action_protocol.expected_action_count(MSRC_SPEC.v):
        raise RuntimeError("MSRC_ACTION_COUNT_MISMATCH")
    carrier_path = output_dir / "carrier_pre_gt.npz"
    np.savez_compressed(carrier_path, **arrays)

    validation_mass = {}
    for fold in folds:
        rows = adapter.row_positions(fold.val_raw_row_ids)
        masses = arrays["U_cycle"][rows].sum(axis=1)
        validation_mass[fold.fold_name] = [float(value) for value in masses]
    if any(value <= 0 for masses in validation_mass.values() for value in masses):
        raise RuntimeError("ZERO_UTILITY_MASS")

    audit = {
        "dataset": "MSRC", "dev_seed": dev_seed,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "native_lr": p2_carrier.LR, "batch_size": MSRC_SPEC.batch_size,
        "carrier_state_sha256": state_sha256,
        "carrier_state_file_sha256": state_file_sha256,
        "sample_id_order_hash": c0_protocol.sha256_file_from_array(adapter.sample_ids),
        "native_order_hashes": order_hashes,
        "corruption_parent_hash": p3_b0_materializer.sha256_file(mask_path),
        "sparse_label_parent_hash": p3_b0_materializer.sha256_file(split_path),
        "action_space_parent_hash": p3_b0_materializer.sha256_file(Path(action_protocol.__file__)),
        "action_space_protocol_hash": action_protocol.canonical_action_sha256(MSRC_SPEC.v),
        "view_count": MSRC_SPEC.v, "cluster_count": p1_protocol.K,
        "action_count": int(arrays["generator_actions"].shape[0]),
        "sparse_cv_split_hash": sparse_cv.split_audit_hash(folds),
        "GT_loaded_before_seal": False,
        "U_shape": list(arrays["U_cycle"].shape),
        "U_finite": bool(np.isfinite(arrays["U_cycle"]).all()),
        "U_min": float(arrays["U_cycle"].min()),
        "U_max": float(arrays["U_cycle"].max()),
        "U_mean": float(arrays["U_cycle"].mean()),
        "U_positive_count": int(np.count_nonzero(arrays["U_cycle"] > 0)),
        "validation_anchor_utility_mass": validation_mass,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False,
        "full_gt_used_for_selection": False,
    }
    audit_path = output_dir / "carrier_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model": model, "views": views, "adapter": adapter, "spec": MSRC_SPEC,
        "k": p1_protocol.K, "carrier_path": carrier_path, "state_path": state_path,
        "audit_path": audit_path, "state_sha256": state_sha256,
        "folds": folds, "arrays": arrays,
    }
