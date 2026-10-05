"""A2 execution wiring over the frozen B1/B2 pre-GT evaluator.

This module owns only orchestration and persistence.  Branch construction,
semantic loss, native consolidation, posterior construction, and J scoring
remain delegated to the frozen B1/B2 primitives.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import uuid
from typing import Callable, Mapping

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_optuna import (
    canonical_fold_order_r3,
    formal_base_reference,
    formal_candidate_evaluator,
    formal_candidate_evaluator_r2,
    formal_carrier_materializer_r1,
    formal_tuning_context,
    objective_a1,
    protocol as b1_protocol,
    trial_runner,
    trial_runner_a1,
    sparse_cv,
)

from . import candidate_schema, optimizer_factory, protocol
from .context import build_context


BRANCH_KEYS = ("11A", "11B", "37A", "37B")
REQUIRED_FALSE_FLAGS = ("full_gt_used", "formal_metrics_read")


def _load_frozen_model_with_parity(carrier, device: torch.device):
    """Load through the frozen evaluator and fail closed on state drift.

    The model construction/loading implementation intentionally remains owned
    by ``formal_candidate_evaluator._load_model``.  The second checkpoint read
    below is audit-only and uses the same frozen C0 loader; it does not provide
    a fallback construction or loading path.
    """
    model = formal_candidate_evaluator._load_model(carrier, device)
    carrier_state = c0._load_state(carrier.state_path, device)
    loaded_state = model.state_dict()
    if set(carrier_state) != set(loaded_state):
        raise RuntimeError("B5_INITIAL_MODEL_STATE_KEY_MISMATCH")
    for name in carrier_state:
        expected = carrier_state[name]
        observed = loaded_state[name]
        if (
            expected.dtype != observed.dtype
            or expected.shape != observed.shape
            or not torch.equal(expected, observed)
        ):
            raise RuntimeError(f"B5_INITIAL_MODEL_STATE_TENSOR_MISMATCH:{name}")
    carrier_hash = c0_protocol.state_dict_sha256(carrier_state)
    loaded_hash = c0._state_hash(model)
    if loaded_hash != carrier_hash:
        raise RuntimeError("B5_INITIAL_MODEL_STATE_HASH_MISMATCH")
    return model, {
        "carrier_model_state_file_sha256": carrier.state_file_sha256,
        "carrier_model_state_sha256": carrier_hash,
        "loaded_initial_model_state_sha256": loaded_hash,
        "model_state_exact_parity": True,
        "model_loading_path": (
            "formal_candidate_evaluator._load_model -> "
            "c0._new_model + c0._load_state + strict=True"
        ),
    }


def _load_pretraining_smoke_carrier(dataset: str, dev_seed: int):
    """Build a verified, shape-only carrier that cannot enter training.

    Loading CCV.mat is intentionally unnecessary for an API smoke because the
    frozen model constructor consumes only ``view.shape[1]``.  Broadcast view
    proxies preserve the exact frozen dimensions while allocating one scalar;
    the real checkpoint and pre-GT carrier payload are still hash-verified and
    loaded from the frozen carrier directory.
    """
    root = protocol.REPO_ROOT / "outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
    directory = formal_carrier_materializer_r1.frozen_materializer.carrier_directory(
        root, dataset, dev_seed
    )
    formal_carrier_materializer_r1.verify_complete_carrier(
        directory, dataset, dev_seed
    )
    state_path = directory / "carrier_model_state.pt"
    pre_gt_path = directory / "carrier_pre_gt.npz"
    audit_path = directory / "carrier_audit.json"
    arrays = formal_tuning_context._load_arrays(pre_gt_path)
    formal_tuning_context._validate_carrier_payload(dataset, arrays)
    spec = c0_protocol.DATASET_SPECS[dataset]
    views = tuple(
        np.broadcast_to(
            np.zeros((1, 1), dtype=np.float32),
            (spec.n, feature_dim),
        )
        for feature_dim in spec.dims
    )
    adapter = b0.SampleIdAdapter(np.asarray(arrays["sample_ids"], dtype=np.int64))
    k = int(np.asarray(arrays["q_views"]).shape[2])
    raw_folds = sparse_cv.build_sparse_label_folds(
        np.asarray(arrays["labeled_ids"], dtype=np.int64),
        np.asarray(arrays["labeled_targets"], dtype=np.int64),
        expected_class_ids=range(k),
    )
    folds, _ = canonical_fold_order_r3.canonicalize_sparse_folds(raw_folds, k)
    return formal_tuning_context.CarrierContext(
        dataset=dataset,
        dev_seed=dev_seed,
        directory=directory,
        state_path=state_path,
        pre_gt_path=pre_gt_path,
        audit_path=audit_path,
        arrays=arrays,
        views=views,
        adapter=adapter,
        spec=spec,
        k=k,
        folds=folds,
        state_file_sha256=protocol.sha256_file(state_path),
        pre_gt_sha256=protocol.sha256_file(pre_gt_path),
    )


def run_model_load_smoke(
    *,
    dataset: str = "CCV",
    dev_seed: int = 11,
    fold: str = "A",
    optimizer_name: str = "AdamW",
    semantic_lr: float = 3e-6,
    device_name: str = "cpu",
) -> dict:
    """Exercise B5 schema/runtime construction without taking an optimizer step."""
    if dataset not in protocol.DATASETS or dev_seed not in b1_protocol.DEV_SEEDS or fold not in b1_protocol.FOLDS:
        raise ValueError("B5_MODEL_LOAD_SMOKE_SCOPE_INVALID")
    candidate = next(
        (
            item for item in protocol.candidates(dataset)
            if item.optimizer_name == optimizer_name and item.semantic_lr == semantic_lr
        ),
        None,
    )
    if candidate is None:
        raise ValueError("B5_MODEL_LOAD_SMOKE_CANDIDATE_INVALID")
    runtime_config = candidate_schema.validate_real_training_candidate(candidate)

    class _SchemaSmokeContext:
        def __init__(self):
            self.dataset = dataset

        def carrier(self, seed: int):
            return _load_pretraining_smoke_carrier(dataset, seed)

    runtime_calls = []

    def _runtime_only_runner(carrier, branch_fold, evaluator_config, device_name, branch_payload):
        if evaluator_config != runtime_config:
            raise RuntimeError("B5_SCHEMA_RUNTIME_CONFIG_IDENTITY_FAILURE")
        if evaluator_config.runtime_fields != (candidate.semantic_lr, candidate.semantic_exposure):
            raise RuntimeError("B5_SCHEMA_RUNTIME_FIELDS_FAILURE")
        if not branch_payload["U_train"].size or not branch_payload["T_rel"].size:
            raise RuntimeError("B5_SCHEMA_RUNTIME_PAYLOAD_FAILURE")
        runtime_calls.append((carrier.dev_seed, branch_fold, device_name))
        return {
            "semantic_steps": evaluator_config.semantic_exposure,
            "native_epochs": b1_protocol.REFINEMENT_HORIZON,
        }

    # This is the lowest reusable frozen evaluator path.  It consumes the
    # B5-owned runtime config directly, has no B1 CandidateConfig constructor,
    # and the runner deliberately performs no model forward/backward/step/J.
    formal_candidate_evaluator_r2._evaluate_with_runner(
        _SchemaSmokeContext(), runtime_config, device_name, _runtime_only_runner
    )
    carrier = _load_pretraining_smoke_carrier(dataset, dev_seed)
    payload = trial_runner_a1.build_fold_semantics_a1(
        carrier.arrays, carrier.adapter, carrier.folds
    )[fold]
    device = torch.device(device_name)
    model, parity = _load_frozen_model_with_parity(carrier, device)
    semantic_optimizer, optimizer_audit = optimizer_factory.build_semantic_optimizer(
        model, candidate
    )
    if len(semantic_optimizer.state) != 0:
        raise RuntimeError("B5_MODEL_LOAD_SMOKE_OPTIMIZER_STATE_NOT_EMPTY")
    u_cycle = torch.as_tensor(
        np.asarray(payload["U_train"]).copy(), dtype=torch.float32, device=device
    ).detach()
    t_rel = torch.as_tensor(
        np.asarray(payload["T_rel"]).copy(), dtype=torch.float32, device=device
    ).detach()
    if u_cycle.requires_grad or t_rel.requires_grad or not u_cycle.numel() or not t_rel.numel():
        raise RuntimeError("B5_MODEL_LOAD_SMOKE_FROZEN_PAYLOAD_INVALID")
    return {
        "MODEL_LOAD_SMOKE": "PASS",
        "SCHEMA_COMPATIBILITY_SMOKE": "PASS",
        "SGD_CONSTRUCTION_SMOKE": "PASS" if candidate.optimizer_name == "SGD_MOMENTUM" else "NOT_REQUESTED",
        "dataset": dataset,
        "dev_seed": dev_seed,
        "fold": fold,
        "optimizer": optimizer_audit,
        **parity,
        "U_cycle_available": True,
        "T_rel_available": True,
        "U_cycle_detached": True,
        "T_rel_detached": True,
        "optimizer_step_executed": False,
        "frozen_evaluator_runtime_path": "formal_candidate_evaluator_r2._evaluate_with_runner",
        "frozen_evaluator_runtime_branch_count": len(runtime_calls),
        "b1_candidate_config_constructed": False,
        "candidate_evaluation_executed": False,
        "candidate_artifact_created": False,
        "full_gt_used": False,
        "formal_metrics_read": False,
    }


def canonical_lr_slug(value: float) -> str:
    """Stable compact slug for the frozen LR grid (e.g. ``3e-6``)."""
    text = f"{float(value):.0e}"
    mantissa, exponent = text.split("e")
    exponent_value = int(exponent)
    mantissa = mantissa.rstrip("0").rstrip(".") or "0"
    return f"{mantissa}e{exponent_value:+d}".replace("e+", "e")


def candidate_artifact_path(output_root: str | Path, candidate: protocol.Candidate) -> Path:
    return (
        Path(output_root)
        / candidate.dataset.lower()
        / candidate.optimizer_name.lower()
        / f"lr_{canonical_lr_slug(candidate.semantic_lr)}.json"
    )


def _canonical_payload_bytes(payload: Mapping) -> bytes:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _payload_hash(payload: Mapping) -> str:
    import hashlib

    return hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()


def _write_complete_artifact(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["artifact_sha256"] = _payload_hash(payload)
    staging = path.with_name(f".{path.name}.staging_{uuid.uuid4().hex}")
    staging.write_bytes(_canonical_payload_bytes(payload).replace(b"\n", b"\n", 1))
    # Include the hash field in the persisted JSON while retaining a stable
    # hash over the canonical payload without that self-referential field.
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)
    return payload["artifact_sha256"]


def verify_candidate_artifact(path: str | Path, candidate: protocol.Candidate | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("B5_CANDIDATE_ARTIFACT_MISSING")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset", "optimizer_name", "semantic_lr", "semantic_exposure", "branches",
        "mean_delta_J", "worst_delta_J", "positive_count", "optimizer_audit",
        "full_gt_used", "formal_metrics_read", "real_training_started",
        "real_training_completed", "branch_count", "GT_LOADED", "artifact_sha256",
    }
    if not required.issubset(payload):
        raise RuntimeError("B5_CANDIDATE_ARTIFACT_SCHEMA_INVALID")
    observed = protocol.Candidate(
        str(payload["dataset"]), str(payload["optimizer_name"]),
        float(payload["semantic_lr"]), int(payload["semantic_exposure"]),
    )
    if candidate is not None and observed != candidate:
        raise RuntimeError("B5_CANDIDATE_ARTIFACT_IDENTITY_INVALID")
    if payload.get("status") != "COMPLETE" or payload.get("completion_status") != "REAL_CANDIDATE_COMPLETE":
        raise RuntimeError("B5_CANDIDATE_NOT_COMPLETE")
    if payload.get("real_training_started") is not True or payload.get("real_training_completed") is not True:
        raise RuntimeError("B5_CANDIDATE_TRAINING_NOT_COMPLETE")
    if payload.get("branch_count") != 4 or set(payload["branches"]) != set(BRANCH_KEYS):
        raise RuntimeError("B5_CANDIDATE_BRANCH_SET_INVALID")
    if any(payload.get(name) is not False for name in REQUIRED_FALSE_FLAGS) or payload.get("GT_LOADED") is not False:
        raise RuntimeError("B5_CANDIDATE_GT_FIREWALL_INVALID")
    if payload.get("artifact_sha256") != _payload_hash(payload):
        raise RuntimeError("B5_CANDIDATE_ARTIFACT_HASH_INVALID")
    for key in BRANCH_KEYS:
        branch = payload["branches"][key]
        if any(name not in branch for name in (
            "J_BASE", "J_OURS", "DeltaJ", "semantic_optimizer_step_count_total",
            "carrier_model_state_sha256", "loaded_initial_model_state_sha256",
            "model_state_exact_parity",
        )):
            raise RuntimeError("B5_CANDIDATE_BRANCH_SCHEMA_INVALID")
        if branch.get("full_gt_used") is not False or branch.get("formal_metrics_read") is not False:
            raise RuntimeError("B5_CANDIDATE_BRANCH_GT_FIREWALL_INVALID")
        if int(branch["semantic_optimizer_step_count_total"]) <= 0:
            raise RuntimeError("B5_CANDIDATE_SEMANTIC_STEP_COUNT_INVALID")
        if (
            branch["carrier_model_state_sha256"]
            != branch["loaded_initial_model_state_sha256"]
            or branch["model_state_exact_parity"] is not True
        ):
            raise RuntimeError("B5_CANDIDATE_INITIAL_MODEL_PARITY_INVALID")
    return payload


def _native_lr(carrier) -> float:
    if carrier.dataset == "MSRC":
        from experiments.utility_relation.p3_c1_b1_optuna import msrc_adapter
        return float(msrc_adapter.p2_carrier.LR)
    return float(c0_protocol.NATIVE_LR)


def _native_optimizer_audit(native_optimizer, native_lr: float) -> dict:
    group = native_optimizer.param_groups[0]
    unchanged = (
        type(native_optimizer).__name__ == "Adam"
        and float(group["lr"]) == native_lr
        and tuple(group["betas"]) == (0.9, 0.999)
        and float(group["eps"]) == 1e-8
        and float(group["weight_decay"]) == 0.0
    )
    if not unchanged:
        raise RuntimeError("B5_NATIVE_OPTIMIZER_CHANGED")
    return {
        "optimizer_class": type(native_optimizer).__name__,
        "lr": float(group["lr"]),
        "betas": list(group["betas"]),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "native_optimizer_unchanged": True,
    }


def _make_branch_runner(b5_candidate: protocol.Candidate):
    def run_branch(carrier, fold, _candidate_config, device_name: str, payload: Mapping):
        if device_name != "cuda:0":
            raise ValueError("B5_REAL_CANDIDATE_REQUIRES_CUDA_0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED_FOR_B5_REAL_CANDIDATE")
        device = torch.device(device_name)
        model, model_load_audit = _load_frozen_model_with_parity(carrier, device)
        initial_hash = model_load_audit["loaded_initial_model_state_sha256"]
        native_lr = _native_lr(carrier)
        native_optimizer = optim.Adam(model.parameters(), lr=native_lr)
        semantic_optimizer, semantic_audit = optimizer_factory.build_semantic_optimizer(model, b5_candidate)
        optimizer_factory.assert_independent(native_optimizer, semantic_optimizer)
        native_audit = _native_optimizer_audit(native_optimizer, native_lr)
        loader = c0._native_loader(carrier.views, carrier.adapter, carrier.dev_seed, carrier.spec.batch_size)
        full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in carrier.views]
        semantic_orders = [
            np.random.default_rng(carrier.dev_seed + epoch).permutation(payload["query_rows"].size)
            for epoch in range(b5_candidate.semantic_exposure)
        ]
        semantic_count = 0
        native_order_hashes = []
        for epoch in range(b1_protocol.REFINEMENT_HORIZON):
            if epoch < b5_candidate.semantic_exposure:
                loss = c0._semantic_epoch(
                    model, semantic_optimizer, full_views,
                    torch.as_tensor(payload["query_rows"], dtype=torch.long, device=device),
                    torch.as_tensor(payload["train_anchor_rows"], dtype=torch.long, device=device),
                    torch.as_tensor(np.asarray(payload["T_rel"]).copy(), dtype=torch.float32, device=device).detach(),
                    torch.as_tensor(np.asarray(payload["U_train"]).copy(), dtype=torch.float32, device=device).detach(),
                    semantic_orders[epoch], carrier.k,
                )
                if not np.isfinite(loss):
                    raise FloatingPointError("NONFINITE_SEMANTIC_LOSS")
                semantic_count += 1
            loss, order_hash = c0._native_epoch(model, native_optimizer, loader, device)
            if not np.isfinite(loss):
                raise FloatingPointError("NONFINITE_NATIVE_LOSS")
            native_order_hashes.append(order_hash)
        if semantic_count <= 0 or len(native_order_hashes) != b1_protocol.REFINEMENT_HORIZON:
            raise RuntimeError("B5_CANDIDATE_SCHEDULE_CONTRACT_FAILURE")
        pi = trial_runner._validation_pi(
            model, carrier.views, device, payload["validation_rows"], payload["class_signature"]
        )
        j_ours, objective_audit = objective_a1.validation_j_with_audit(
            payload["U_validation"], pi, payload["validation_targets"]
        )
        return {
            "J_OURS": float(j_ours), "audit": objective_audit,
            "initial_state_sha256": initial_hash,
            **model_load_audit,
            "semantic_steps": semantic_count, "native_epochs": len(native_order_hashes),
            "semantic_optimizer_step_count_total": semantic_count,
            "optimizer_audit": {
                "semantic": semantic_audit,
                "native": native_audit,
                "optimizer_state_independent": True,
            },
            "native_order_hashes": native_order_hashes,
            "U_cycle_detached": True, "T_rel_detached": True,
            "anchor_posterior_detached": True,
            "semantic_to_native_grad_leak": False,
            "native_to_next_semantic_grad_leak": False,
            "full_gt_used": False, "formal_metrics_read": False,
        }
    return run_branch


def _branch_key(seed: int, fold: str) -> str:
    return f"{seed}{fold}"


def run_real_candidate(
    candidate: protocol.Candidate,
    output_root: str | Path = protocol.OUTPUT_ROOT,
    *,
    device: str = "cuda:0",
    context_loader: Callable[[str], object] | None = None,
    branch_runner: Callable | None = None,
) -> dict:
    """Execute and persist one authorized candidate; never loads full GT."""
    candidate_config = candidate_schema.validate_real_training_candidate(candidate)
    path = candidate_artifact_path(output_root, candidate)
    if path.is_file():
        return {"status": "ALREADY_COMPLETE", "artifact": str(path), "payload": verify_candidate_artifact(path, candidate)}
    context = (context_loader or build_context)(candidate.dataset)
    runner = branch_runner or _make_branch_runner(candidate)
    results = formal_candidate_evaluator_r2._evaluate_with_runner(
        context, candidate_config, device, runner
    )
    bases = {
        seed: formal_base_reference.load_verified_base_reference(context, seed)
        for seed in b1_protocol.DEV_SEEDS
    }
    branches = {}
    deltas = {}
    optimizer_audits = {}
    for seed in b1_protocol.DEV_SEEDS:
        for fold in b1_protocol.FOLDS:
            result = results[(seed, fold)]
            base_j = float(bases[seed].J_BASE_A if fold == "A" else bases[seed].J_BASE_B)
            ours_j = float(result["J_OURS"])
            delta = objective_a1.paired_delta_j(base_j, ours_j)
            key = _branch_key(seed, fold)
            deltas[(seed, fold)] = delta
            branches[key] = {
                "J_BASE": base_j, "J_OURS": ours_j, "DeltaJ": delta,
                "supported_count": result["audit"]["supported_anchor_count"],
                "abstained_count": result["audit"]["abstained_anchor_count"],
                "coverage": result["audit"]["coverage"],
                "semantic_optimizer_step_count_total": result["semantic_optimizer_step_count_total"],
                "semantic_steps": result["semantic_steps"],
                "native_epochs": result["native_epochs"],
                "optimizer_audit": result["optimizer_audit"],
                "initial_state_sha256": result["initial_state_sha256"],
                "carrier_model_state_file_sha256": result["carrier_model_state_file_sha256"],
                "carrier_model_state_sha256": result["carrier_model_state_sha256"],
                "loaded_initial_model_state_sha256": result["loaded_initial_model_state_sha256"],
                "model_state_exact_parity": result["model_state_exact_parity"],
                "model_loading_path": result["model_loading_path"],
                "U_cycle_detached": result["U_cycle_detached"],
                "T_rel_detached": result["T_rel_detached"],
                "anchor_posterior_detached": result["anchor_posterior_detached"],
                "semantic_to_native_grad_leak": result["semantic_to_native_grad_leak"],
                "native_to_next_semantic_grad_leak": result["native_to_next_semantic_grad_leak"],
                "full_gt_used": False, "formal_metrics_read": False,
            }
            optimizer_audits[key] = result["optimizer_audit"]
    aggregation = objective_a1.aggregate_candidate_scores(deltas)
    payload = {
        "status": "COMPLETE", "completion_status": "REAL_CANDIDATE_COMPLETE",
        "dataset": candidate.dataset, "optimizer_name": candidate.optimizer_name,
        "semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
        "branches": branches,
        "mean_delta_J": aggregation["mean_delta_J"],
        "worst_delta_J": aggregation["worst_delta_J"],
        "positive_count": aggregation["positive_count"],
        "optimizer_audit": optimizer_audits,
        "full_gt_used": False, "formal_metrics_read": False, "GT_LOADED": False,
        "real_training_started": True, "real_training_completed": True,
        "branch_count": 4,
        "preflight_status": "AUTHORIZATION_PREFLIGHT_PASS",
        "protocol_fingerprint": context.protocol_fingerprint,
        "candidate_origin": "B5_A2_REAL_TRAINING",
    }
    artifact_hash = _write_complete_artifact(path, payload)
    verified = verify_candidate_artifact(path, candidate)
    return {
        "status": "COMPLETE", "completion_status": "REAL_CANDIDATE_COMPLETE",
        "artifact": str(path), "artifact_sha256": artifact_hash, "payload": verified,
    }


__all__ = [
    "BRANCH_KEYS", "canonical_lr_slug", "candidate_artifact_path",
    "verify_candidate_artifact", "run_real_candidate", "run_model_load_smoke",
]
