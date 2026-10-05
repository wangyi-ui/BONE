"""One-candidate B1 refinement runner; no Optuna study and no GT path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0

from . import objective, protocol
from .study_schema import CandidateConfig


@dataclass
class ScheduleResult:
    models: dict
    initial_hashes: dict[str, str]
    native_counts: dict[str, int]
    semantic_counts: dict[str, int]
    native_order_hashes: dict[str, list[str]]


def execute_isolated_schedule(
    load_branch: Callable[[], object],
    state_hash: Callable[[object], str],
    make_native_optimizer: Callable[[object], object],
    make_semantic_optimizer: Callable[[object, float], object],
    make_native_loader: Callable[[], object],
    native_epoch: Callable[[object, object, object], tuple[float, str]],
    semantic_epoch: Callable[[object, object, str, int], float],
    semantic_lr: float,
    semantic_exposure: int,
) -> ScheduleResult:
    """Run BASE/A/B from three isolated clones with independent optimizers."""
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    models = {name: load_branch() for name in ("base", "A", "B")}
    initial = {name: state_hash(model) for name, model in models.items()}
    if len(set(initial.values())) != 1:
        raise RuntimeError("INITIAL_BRANCH_STATE_MISMATCH")
    native_optimizers = {name: make_native_optimizer(model) for name, model in models.items()}
    semantic_optimizers = {
        fold: make_semantic_optimizer(models[fold], candidate.semantic_lr) for fold in protocol.FOLDS
    }
    if len({id(value) for value in (*native_optimizers.values(), *semantic_optimizers.values())}) != 5:
        raise RuntimeError("OPTIMIZER_STATE_REUSE")
    loaders = {name: make_native_loader() for name in models}
    native_counts = {name: 0 for name in models}
    semantic_counts = {name: 0 for name in protocol.FOLDS}
    orders = {name: [] for name in models}

    for epoch in range(protocol.REFINEMENT_HORIZON):
        loss, order_hash = native_epoch(models["base"], native_optimizers["base"], loaders["base"])
        if not np.isfinite(loss):
            raise FloatingPointError("NONFINITE_BASE_NATIVE_LOSS")
        native_counts["base"] += 1
        orders["base"].append(order_hash)
    for fold in protocol.FOLDS:
        for epoch in range(protocol.REFINEMENT_HORIZON):
            if epoch < candidate.semantic_exposure:
                semantic_loss = semantic_epoch(models[fold], semantic_optimizers[fold], fold, epoch)
                if not np.isfinite(semantic_loss):
                    raise FloatingPointError("NONFINITE_SEMANTIC_LOSS")
                semantic_counts[fold] += 1
            loss, order_hash = native_epoch(models[fold], native_optimizers[fold], loaders[fold])
            if not np.isfinite(loss):
                raise FloatingPointError("NONFINITE_OURS_NATIVE_LOSS")
            native_counts[fold] += 1
            orders[fold].append(order_hash)
            print(f"fold={fold} refinement={epoch + 1}/{protocol.REFINEMENT_HORIZON}", flush=True)
    if orders["base"] != orders["A"] or orders["base"] != orders["B"]:
        raise RuntimeError("PAIRED_NATIVE_ORDER_MISMATCH")
    return ScheduleResult(models, initial, native_counts, semantic_counts, orders)


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def build_fold_semantics(arrays: Mapping[str, np.ndarray], adapter, folds) -> dict[str, dict]:
    """Build frozen B1 WHAT tensors using one training anchor per class/fold."""
    p_gen = np.asarray(arrays["p_gen"])
    u_carrier = np.asarray(arrays["U_cycle"])
    all_labeled = set(int(value) for fold in folds for value in (*fold.train_raw_row_ids, *fold.val_raw_row_ids))
    query_rows = np.asarray(
        [row for row, raw_id in enumerate(adapter.sample_ids) if int(raw_id) not in all_labeled], dtype=np.int64
    )
    payloads = {}
    for fold in folds:
        train_rows = adapter.row_positions(fold.train_raw_row_ids)
        val_rows = adapter.row_positions(fold.val_raw_row_ids)
        targets = np.asarray(fold.train_targets, dtype=np.int64)
        if not np.array_equal(targets, np.arange(len(targets))):
            raise ValueError("FOLD_CLASS_ORDER_MUST_BE_CONTIGUOUS")
        anchor_signature = p_gen[train_rows].mean(axis=1)
        class_signature = anchor_signature.copy()
        energy = np.einsum("usk,ck->usc", p_gen[query_rows], class_signature)
        denominator = energy.sum(axis=2)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("NONPOSITIVE_WHAT_DENOMINATOR")
        pi = energy / denominator[..., None]
        t_rel = np.take(pi, targets, axis=2).transpose(0, 2, 1)
        val_mass = u_carrier[val_rows].sum(axis=1)
        if np.any(val_mass <= 0):
            raise ValueError("ZERO_UTILITY_MASS")
        payloads[fold.fold_name] = {
            "query_rows": query_rows.copy(), "train_anchor_rows": train_rows,
            "validation_rows": val_rows, "validation_targets": np.asarray(fold.val_targets, dtype=np.int64),
            "class_signature": class_signature, "Pi_train": pi,
            "T_rel": t_rel, "U_train": u_carrier[query_rows].copy(),
            "U_validation": u_carrier[val_rows].copy(),
            "validation_utility_mass": val_mass,
            "frozen_hash": _array_hash(class_signature, pi, t_rel, u_carrier[query_rows], u_carrier[val_rows]),
        }
    return payloads


def _validation_pi(model, views, device, validation_rows, class_signature) -> np.ndarray:
    q_views = b0.extract_q_views(model, views, device=device)
    actions, _, _ = b0.action_arrays(q_views.shape[1])
    p_gen, _ = b0._posteriors_for_chunk(q_views[validation_rows], actions)
    energy = np.einsum("usk,ck->usc", p_gen, class_signature)
    denominator = energy.sum(axis=2)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("NONPOSITIVE_VALIDATION_DENOMINATOR")
    return energy / denominator[..., None]


def run_single_candidate(carrier: dict, semantic_lr: float, semantic_exposure: int, output_dir: str | Path) -> dict:
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    views, adapter, spec, k = carrier["views"], carrier["adapter"], carrier["spec"], carrier["k"]
    device = torch.device("cuda:0")
    fold_payloads = build_fold_semantics(carrier["arrays"], adapter, carrier["folds"])
    frozen_before = {fold: item["frozen_hash"] for fold, item in fold_payloads.items()}

    def load_branch():
        model = c0._new_model(views, k, spec, protocol.DEV_SEEDS[0], device)
        model.load_state_dict(c0._load_state(carrier["state_path"], device), strict=True)
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
            model, optimizer, full_views,
            torch.as_tensor(payload["query_rows"], dtype=torch.long, device=device),
            torch.as_tensor(payload["train_anchor_rows"], dtype=torch.long, device=device),
            torch.as_tensor(payload["T_rel"].copy(), dtype=torch.float32, device=device).detach(),
            torch.as_tensor(payload["U_train"].copy(), dtype=torch.float32, device=device).detach(),
            semantic_orders[fold][epoch], k,
        )

    schedule = execute_isolated_schedule(
        load_branch=load_branch,
        state_hash=c0._state_hash,
        make_native_optimizer=lambda model: optim.Adam(model.parameters(), lr=c0_protocol.NATIVE_LR),
        make_semantic_optimizer=lambda model, lr: optim.Adam(model.parameters(), lr=lr),
        make_native_loader=lambda: c0._native_loader(views, adapter, protocol.DEV_SEEDS[0], spec.batch_size),
        native_epoch=lambda model, optimizer, loader: c0._native_epoch(model, optimizer, loader, device),
        semantic_epoch=semantic_step,
        semantic_lr=candidate.semantic_lr,
        semantic_exposure=candidate.semantic_exposure,
    )
    if any(fold_payloads[fold]["frozen_hash"] != frozen_before[fold] for fold in protocol.FOLDS):
        raise RuntimeError("FROZEN_SEMANTIC_TENSOR_MUTATION")
    if any(value != carrier["state_sha256"] for value in schedule.initial_hashes.values()):
        raise RuntimeError("CARRIER_STATE_PARITY_FAILURE")

    scores = {}
    for fold, payload in fold_payloads.items():
        base_pi = _validation_pi(schedule.models["base"], views, device, payload["validation_rows"], payload["class_signature"])
        ours_pi = _validation_pi(schedule.models[fold], views, device, payload["validation_rows"], payload["class_signature"])
        base_j = objective.validation_j(payload["U_validation"], base_pi, payload["validation_targets"])
        ours_j = objective.validation_j(payload["U_validation"], ours_pi, payload["validation_targets"])
        scores[fold] = {"J_BASE": base_j, "J_OURS": ours_j, "DeltaJ": objective.paired_delta_j(base_j, ours_j)}

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
        "scores": scores,
        "SMOKE_PARTIAL_OBJECTIVE_ONLY": True,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_selection": False,
        "full_gt_used_for_smoke_objective": False,
    }
    audit_path = output_dir / "audit" / "trial_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"schedule": schedule, "scores": scores, "fold_payloads": fold_payloads, "audit_path": audit_path}
