"""Thin A1 orchestration over frozen B1 training and posterior primitives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0

from . import objective_a1, protocol, trial_runner
from .study_schema import CandidateConfig


def build_fold_semantics_a1(arrays: Mapping[str, np.ndarray], adapter, folds) -> dict[str, dict]:
    """Build unchanged frozen WHAT tensors while permitting A1 abstention at validation."""
    p_gen = np.asarray(arrays["p_gen"])
    u_carrier = np.asarray(arrays["U_cycle"])
    all_labeled = set(
        int(value) for fold in folds for value in (*fold.train_raw_row_ids, *fold.val_raw_row_ids)
    )
    query_rows = np.asarray(
        [row for row, raw_id in enumerate(adapter.sample_ids) if int(raw_id) not in all_labeled],
        dtype=np.int64,
    )
    payloads = {}
    for fold in folds:
        train_rows = adapter.row_positions(fold.train_raw_row_ids)
        val_rows = adapter.row_positions(fold.val_raw_row_ids)
        targets = np.asarray(fold.train_targets, dtype=np.int64)
        if not np.array_equal(targets, np.arange(len(targets))):
            raise ValueError("FOLD_CLASS_ORDER_MUST_BE_CONTIGUOUS")

        # Exact frozen WHAT semantics: sparse anchor signatures never use U.
        anchor_signature = p_gen[train_rows].mean(axis=1)
        class_signature = anchor_signature.copy()
        energy = np.einsum("usk,ck->usc", p_gen[query_rows], class_signature)
        denominator = energy.sum(axis=2)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("NONPOSITIVE_WHAT_DENOMINATOR")
        pi = energy / denominator[..., None]
        t_rel = np.take(pi, targets, axis=2).transpose(0, 2, 1)
        val_mass = u_carrier[val_rows].sum(axis=1)
        if not np.isfinite(val_mass).all() or np.any(val_mass < 0):
            raise ValueError("INVALID_VALIDATION_UTILITY_MASS")
        payloads[fold.fold_name] = {
            "query_rows": query_rows.copy(),
            "train_anchor_rows": train_rows,
            "train_anchor_raw_ids": np.asarray(fold.train_raw_row_ids, dtype=np.int64),
            "validation_rows": val_rows,
            "validation_raw_ids": np.asarray(fold.val_raw_row_ids, dtype=np.int64),
            "validation_targets": np.asarray(fold.val_targets, dtype=np.int64),
            "class_signature": class_signature,
            "Pi_train": pi,
            "T_rel": t_rel,
            "U_train": u_carrier[query_rows].copy(),
            "U_validation": u_carrier[val_rows].copy(),
            "validation_utility_mass": val_mass,
            "frozen_hash": trial_runner._array_hash(
                class_signature, pi, t_rel, u_carrier[query_rows], u_carrier[val_rows]
            ),
        }
    return payloads


def run_single_candidate_a1(
    carrier: dict,
    semantic_lr: float,
    semantic_exposure: int,
    output_dir: str | Path,
    device_name: str = "cuda:0",
) -> dict:
    """Run one A1 candidate using only frozen B1 training primitives."""
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    if device_name != "cuda:0":
        raise ValueError("A1_MSRC_SMOKE_REQUIRES_CUDA_0")
    views, adapter, spec, k = carrier["views"], carrier["adapter"], carrier["spec"], carrier["k"]
    device = torch.device(device_name)
    fold_payloads = build_fold_semantics_a1(carrier["arrays"], adapter, carrier["folds"])
    frozen_before = {fold: payload["frozen_hash"] for fold, payload in fold_payloads.items()}

    def load_branch():
        model = c0._new_model(views, k, spec, protocol.DEV_SEEDS[0], device)
        model.load_state_dict(c0._load_state(Path(carrier["state_path"]), device), strict=True)
        return model

    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views]
    semantic_orders = {
        fold: np.stack([
            np.random.default_rng(protocol.DEV_SEEDS[0] + index).permutation(payload["query_rows"].size)
            for index in range(candidate.semantic_exposure)
        ])
        for fold, payload in fold_payloads.items()
    }

    def semantic_step(model, optimizer, fold, epoch):
        payload = fold_payloads[fold]
        return c0._semantic_epoch(
            model,
            optimizer,
            full_views,
            torch.as_tensor(payload["query_rows"], dtype=torch.long, device=device),
            torch.as_tensor(payload["train_anchor_rows"], dtype=torch.long, device=device),
            torch.as_tensor(payload["T_rel"].copy(), dtype=torch.float32, device=device).detach(),
            torch.as_tensor(payload["U_train"].copy(), dtype=torch.float32, device=device).detach(),
            semantic_orders[fold][epoch],
            k,
        )

    schedule = trial_runner.execute_isolated_schedule(
        load_branch=load_branch,
        state_hash=c0._state_hash,
        make_native_optimizer=lambda model: optim.Adam(model.parameters(), lr=c0_protocol.NATIVE_LR),
        make_semantic_optimizer=lambda model, lr: optim.Adam(model.parameters(), lr=lr),
        make_native_loader=lambda: c0._native_loader(
            views, adapter, protocol.DEV_SEEDS[0], spec.batch_size
        ),
        native_epoch=lambda model, optimizer, loader: c0._native_epoch(model, optimizer, loader, device),
        semantic_epoch=semantic_step,
        semantic_lr=candidate.semantic_lr,
        semantic_exposure=candidate.semantic_exposure,
    )
    if any(fold_payloads[fold]["frozen_hash"] != frozen_before[fold] for fold in protocol.FOLDS):
        raise RuntimeError("FROZEN_SEMANTIC_TENSOR_MUTATION")
    if len(set(schedule.initial_hashes.values())) != 1:
        raise RuntimeError("CARRIER_STATE_PARITY_FAILURE")

    scores, coverage = {}, {}
    for fold, payload in fold_payloads.items():
        base_pi = trial_runner._validation_pi(
            schedule.models["base"], views, device,
            payload["validation_rows"], payload["class_signature"],
        )
        ours_pi = trial_runner._validation_pi(
            schedule.models[fold], views, device,
            payload["validation_rows"], payload["class_signature"],
        )
        base_j, base_audit = objective_a1.validation_j_with_audit(
            payload["U_validation"], base_pi, payload["validation_targets"]
        )
        ours_j, ours_audit = objective_a1.validation_j_with_audit(
            payload["U_validation"], ours_pi, payload["validation_targets"]
        )
        if any(base_audit[key] != ours_audit[key] for key in (
            "supported_anchor_count", "abstained_anchor_count", "coverage", "fold_denominator"
        )):
            raise RuntimeError("BASE_OURS_COVERAGE_MISMATCH")
        scores[fold] = {
            "J_BASE": base_j,
            "J_OURS": ours_j,
            "DeltaJ": objective_a1.paired_delta_j(base_j, ours_j),
            "base_anchor_scores": base_audit["anchor_scores"],
            "ours_anchor_scores": ours_audit["anchor_scores"],
        }
        coverage[fold] = base_audit

    output_dir = Path(output_dir)
    (output_dir / "base").mkdir(parents=True, exist_ok=False)
    (output_dir / "ours_foldA").mkdir(parents=True, exist_ok=False)
    (output_dir / "ours_foldB").mkdir(parents=True, exist_ok=False)
    (output_dir / "audit").mkdir(parents=True, exist_ok=False)
    c0._save_state(schedule.models["base"], output_dir / "base" / "final_model_state.pt")
    c0._save_state(schedule.models["A"], output_dir / "ours_foldA" / "final_model_state.pt")
    c0._save_state(schedule.models["B"], output_dir / "ours_foldB" / "final_model_state.pt")
    audit = {
        "initial_state_sha256": schedule.initial_hashes,
        "native_counts": schedule.native_counts,
        "semantic_counts": schedule.semantic_counts,
        "native_order_hashes": schedule.native_order_hashes,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "U_carrier_frozen": True,
        "T_rel_frozen": True,
        "Pi_train_frozen": True,
        "class_signature_frozen": True,
        "validation_rows_excluded_from_semantic_training": True,
        "fold_frozen_hashes": frozen_before,
        "coverage_audit_only": True,
        "coverage": coverage,
        "scores": scores,
        "SMOKE_PARTIAL_OBJECTIVE_ONLY": True,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False,
        "full_gt_used_for_selection": False,
    }
    audit_path = output_dir / "audit" / "trial_a1_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schedule": schedule,
        "scores": scores,
        "coverage": coverage,
        "fold_payloads": fold_payloads,
        "audit_path": audit_path,
    }
