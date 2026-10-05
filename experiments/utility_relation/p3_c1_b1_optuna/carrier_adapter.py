"""B1 development-carrier adapter over frozen P3-B0/P3-C0 primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from method.amv import amv
from utils.seed_util import set_seed

from . import protocol, sparse_cv


def validate_carrier_request(dataset: str, dev_seed: int, device: str) -> None:
    if dataset not in protocol.DATASETS or dataset not in c0_protocol.DATASET_SPECS:
        raise ValueError("UNSUPPORTED_OR_UNREGISTERED_DATASET")
    if dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("UNREGISTERED_DEV_SEED")
    if dev_seed in protocol.FORMAL_SEEDS:
        raise ValueError("FORMAL_SEED_FORBIDDEN")
    if device != "cuda:0":
        raise ValueError("ENGINEERING_SMOKE_REQUIRES_CUDA_0")


def expected_carrier_dir(output_root: str | Path, dataset: str, dev_seed: int) -> Path:
    slug = c0_protocol.dataset_slug(dataset)
    return Path(output_root) / slug / f"dev_seed{dev_seed}" / "carrier"


def _carrier_arrays(q_views: np.ndarray) -> dict[str, np.ndarray]:
    """Reuse frozen P3-B0 action/posterior/U primitives without constructing T."""
    b0.validate_q_views(q_views)
    actions, generator, verifier = b0.action_arrays(q_views.shape[1])
    p_gen, p_ver = b0._posteriors_for_chunk(q_views, actions)
    _, _, _, _, u_cycle = b0.compute_u_cycle(p_gen, p_ver)
    return {
        "q_views": np.ascontiguousarray(q_views),
        "p_gen": np.ascontiguousarray(p_gen),
        "p_ver": np.ascontiguousarray(p_ver),
        "U_cycle": np.ascontiguousarray(u_cycle),
        "generator_actions": generator,
        "verifier_actions": verifier,
    }


def build_development_carrier(
    dataset: str,
    dev_seed: int,
    device: str,
    output_dir: str | Path,
) -> dict:
    """Train exactly one 100-native-epoch pre-GT development carrier."""
    validate_carrier_request(dataset, dev_seed, device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.cuda.set_device(0)
    torch_device = torch.device(device)
    spec = c0_protocol.DATASET_SPECS[dataset]
    labeled_ids, labeled_targets, corruption_mask, parent_hashes = c0._load_frozen_inputs(dataset)
    raw_views = c0.load_raw_views(dataset)
    views, adapter = b0.preprocess_with_raw_ids(raw_views, corruption_mask)
    if not np.array_equal(np.sort(adapter.sample_ids), np.arange(spec.n)):
        raise RuntimeError("SAMPLE_ID_SET_MISMATCH")
    folds = sparse_cv.build_sparse_label_folds(
        labeled_ids, labeled_targets, expected_class_ids=range(int(labeled_targets.max()) + 1)
    )

    k = int(labeled_targets.max()) + 1
    set_seed(dev_seed)
    anchors = amv(c0._container(views, k), m=spec.n_anchor, ret_anchor=True, max_iter=100,
                  seed=dev_seed, info=True, info_iter=False)
    model = c0._new_model(views, k, spec, dev_seed, torch_device, anchors)
    optimizer = optim.Adam(model.parameters(), lr=c0_protocol.NATIVE_LR)
    loader = c0._native_loader(views, adapter, dev_seed, spec.batch_size)
    losses, order_hashes = [], []
    for epoch in range(protocol.CARRIER_EPOCHS):
        loss, order_hash = c0._native_epoch(model, optimizer, loader, torch_device)
        losses.append(loss)
        order_hashes.append(order_hash)
        if (epoch + 1) % 10 == 0:
            print(f"carrier epoch={epoch + 1}/{protocol.CARRIER_EPOCHS} loss={loss:.8f}", flush=True)
    if len(losses) != protocol.CARRIER_EPOCHS or not np.isfinite(losses).all():
        raise RuntimeError("CARRIER_EPOCH_FAILURE")

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
    carrier_path = output_dir / "carrier_pre_gt.npz"
    np.savez_compressed(carrier_path, **arrays)

    validation_mass = {}
    for fold in folds:
        rows = adapter.row_positions(fold.val_raw_row_ids)
        masses = arrays["U_cycle"][rows].sum(axis=1)
        validation_mass[fold.fold_name] = [float(value) for value in masses]
    if any(value <= 0 for masses in validation_mass.values() for value in masses):
        raise RuntimeError("ZERO_UTILITY_MASS")

    split_path, mask_path = c0_protocol.artifact_paths(dataset)
    audit = {
        "dataset": dataset,
        "dev_seed": dev_seed,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "native_lr": c0_protocol.NATIVE_LR,
        "batch_size": spec.batch_size,
        "carrier_state_sha256": state_sha256,
        "carrier_state_file_sha256": state_file_sha256,
        "sample_id_order_hash": c0_protocol.sha256_file_from_array(adapter.sample_ids),
        "native_order_hashes": order_hashes,
        "corruption_parent_hash": c0_protocol.sha256_file(mask_path),
        "sparse_label_parent_hash": c0_protocol.sha256_file(split_path),
        "parent_hashes": parent_hashes,
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
        "full_gt_used_for_selection": False,
        "full_gt_used_for_smoke_objective": False,
    }
    audit_path = output_dir / "carrier_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model": model, "views": views, "adapter": adapter, "spec": spec, "k": k,
        "carrier_path": carrier_path, "state_path": state_path, "audit_path": audit_path,
        "state_sha256": state_sha256, "folds": folds, "arrays": arrays,
    }
