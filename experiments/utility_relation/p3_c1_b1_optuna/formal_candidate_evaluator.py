"""Four-branch A1 candidate evaluator with atomic, hash-checked caching."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Callable, Mapping

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0

from . import amendment_a1, objective_a1, protocol, trial_runner
from .formal_base_reference import BaseReference, load_verified_base_reference
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .study_schema import CandidateConfig, CandidateEvaluation, candidate_key
from .parent_integrity import sha256_file


def canonical_candidate_slug(candidate: CandidateConfig) -> str:
    return f"lr_{candidate.semantic_lr:.17g}__es_{candidate.semantic_exposure}"


def candidate_directory(formal_tuning_root: str | Path, dataset: str, candidate: CandidateConfig) -> Path:
    from .formal_carrier_materializer import DATASET_SLUGS
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    return Path(formal_tuning_root) / DATASET_SLUGS[dataset] / "candidates" / canonical_candidate_slug(candidate)


def _load_model(carrier: CarrierContext, device: torch.device):
    model = c0._new_model(carrier.views, carrier.k, carrier.spec, carrier.dev_seed, device)
    model.load_state_dict(c0._load_state(carrier.state_path, device), strict=True)
    return model


def _native_lr(carrier: CarrierContext) -> float:
    if carrier.dataset == "MSRC":
        from . import msrc_adapter
        return float(msrc_adapter.p2_carrier.LR)
    return float(c0_protocol.NATIVE_LR)


def _run_branch(carrier: CarrierContext, fold: str, candidate: CandidateConfig, device_name: str,
                payload: Mapping[str, object]) -> dict:
    if device_name != "cuda:0":
        raise ValueError("FORMAL_CANDIDATE_REQUIRES_CUDA_0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(device_name)
    model = _load_model(carrier, device)
    initial_hash = c0._state_hash(model)
    native_optimizer = optim.Adam(model.parameters(), lr=_native_lr(carrier))
    semantic_optimizer = optim.Adam(model.parameters(), lr=candidate.semantic_lr)
    loader = c0._native_loader(carrier.views, carrier.adapter, carrier.dev_seed, carrier.spec.batch_size)
    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in carrier.views]
    semantic_orders = [
        np.random.default_rng(carrier.dev_seed + epoch).permutation(payload["query_rows"].size)
        for epoch in range(candidate.semantic_exposure)
    ]
    native_order_hashes, semantic_count = [], 0
    for epoch in range(protocol.REFINEMENT_HORIZON):
        if epoch < candidate.semantic_exposure:
            semantic_loss = c0._semantic_epoch(
                model, semantic_optimizer, full_views,
                torch.as_tensor(payload["query_rows"], dtype=torch.long, device=device),
                torch.as_tensor(payload["train_anchor_rows"], dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(payload["T_rel"]).copy(), dtype=torch.float32, device=device).detach(),
                torch.as_tensor(np.asarray(payload["U_train"]).copy(), dtype=torch.float32, device=device).detach(),
                semantic_orders[epoch], carrier.k,
            )
            if not np.isfinite(semantic_loss):
                raise FloatingPointError("NONFINITE_SEMANTIC_LOSS")
            semantic_count += 1
        loss, order_hash = c0._native_epoch(model, native_optimizer, loader, device)
        if not np.isfinite(loss):
            raise FloatingPointError("NONFINITE_NATIVE_LOSS")
        native_order_hashes.append(order_hash)
    if len(native_order_hashes) != protocol.REFINEMENT_HORIZON:
        raise RuntimeError("NATIVE_EPOCH_COUNT_MISMATCH")
    pi = trial_runner._validation_pi(
        model, carrier.views, device, payload["validation_rows"], payload["class_signature"]
    )
    j_ours, audit = objective_a1.validation_j_with_audit(
        payload["U_validation"], pi, payload["validation_targets"]
    )
    return {
        "model": model, "initial_state_sha256": initial_hash,
        "J_OURS": float(j_ours), "audit": audit,
        "semantic_steps": semantic_count, "native_epochs": len(native_order_hashes),
        "native_order_hashes": native_order_hashes,
    }


def _evaluate_with_runner(context: FormalTuningContext, candidate: CandidateConfig, device: str,
                          runner: Callable[[CarrierContext, str, CandidateConfig, str, Mapping], dict] | None):
    results = {}
    for seed in protocol.DEV_SEEDS:
        carrier = context.carrier(seed)
        payloads = trial_runner_a1.build_fold_semantics_a1(carrier.arrays, carrier.adapter, carrier.folds)
        for fold in protocol.FOLDS:
            result = (runner or _run_branch)(carrier, fold, candidate, device, payloads[fold])
            if result.get("semantic_steps") != candidate.semantic_exposure or result.get("native_epochs") != protocol.REFINEMENT_HORIZON:
                raise RuntimeError("CANDIDATE_SCHEDULE_CONTRACT_FAILURE")
            results[(seed, fold)] = result
    return results


def _evaluation_payload(context: FormalTuningContext, candidate: CandidateConfig,
                        bases: Mapping[int, BaseReference], results: Mapping[tuple[int, str], dict]) -> dict:
    deltas = {}
    folds = {}
    for seed in protocol.DEV_SEEDS:
        base = bases[seed]
        for fold in protocol.FOLDS:
            base_j = float(base.J_BASE_A if fold == "A" else base.J_BASE_B)
            ours_j = float(results[(seed, fold)]["J_OURS"])
            delta = objective_a1.paired_delta_j(base_j, ours_j)
            deltas[(seed, fold)] = delta
            item = results[(seed, fold)]
            folds[f"seed{seed}_fold{fold}"] = {
                "J_BASE": base_j, "J_OURS": ours_j, "DeltaJ": delta,
                "supported_count": item["audit"]["supported_anchor_count"],
                "abstained_count": item["audit"]["abstained_anchor_count"],
                "coverage": item["audit"]["coverage"],
                "semantic_steps": item["semantic_steps"],
                "native_epochs": item["native_epochs"],
                "initial_state_sha256": item["initial_state_sha256"],
                "full_gt_loaded": False,
            }
    aggregation = objective_a1.aggregate_candidate_scores(deltas)
    values = {(seed, fold): deltas[(seed, fold)] for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS}
    payload = {
        "dataset": context.dataset, "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "delta_j_seed11_foldA": values[(protocol.DEV_SEEDS[0], protocol.FOLDS[0])],
        "delta_j_seed11_foldB": values[(protocol.DEV_SEEDS[0], protocol.FOLDS[1])],
        "delta_j_seed37_foldA": values[(protocol.DEV_SEEDS[1], protocol.FOLDS[0])],
        "delta_j_seed37_foldB": values[(protocol.DEV_SEEDS[1], protocol.FOLDS[1])],
        "mean_delta_j": aggregation["mean_delta_J"], "worst_delta_j": aggregation["worst_delta_J"],
        "positive_count": aggregation["positive_count"], "folds": folds,
        "protocol_fingerprint": context.protocol_fingerprint,
        "full_gt_used": False, "full_gt_used_for_selection": False,
        "candidate_key": candidate_key(candidate),
    }
    return payload


def _verify_candidate_payload(path: Path, context: FormalTuningContext, candidate: CandidateConfig) -> CandidateEvaluation:
    marker_path, eval_path, audit_path = path / "COMPLETE.json", path / "candidate_evaluation.json", path / "audit.json"
    if not marker_path.is_file() or not eval_path.is_file() or not audit_path.is_file():
        raise ValueError("CANDIDATE_INCOMPLETE")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE" or marker.get("evaluation_sha256") != sha256_file(eval_path) or marker.get("audit_sha256") != sha256_file(audit_path):
        raise ValueError("CANDIDATE_HASH_MISMATCH")
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != context.dataset or payload.get("candidate_key") != candidate_key(candidate):
        raise ValueError("CANDIDATE_IDENTITY_MISMATCH")
    if payload.get("protocol_fingerprint") != context.protocol_fingerprint or payload.get("full_gt_used") is not False:
        raise ValueError("CANDIDATE_PROTOCOL_MISMATCH")
    return CandidateEvaluation(
        context.dataset, candidate.semantic_lr, candidate.semantic_exposure,
        payload["delta_j_seed11_foldA"], payload["delta_j_seed11_foldB"],
        payload["delta_j_seed37_foldA"], payload["delta_j_seed37_foldB"],
    )


def load_verified_candidate(context: FormalTuningContext, candidate: CandidateConfig) -> CandidateEvaluation:
    return _verify_candidate_payload(candidate_directory(context.formal_tuning_root, context.dataset, candidate), context, candidate)


def evaluate_candidate(
    context: FormalTuningContext,
    candidate: CandidateConfig,
    device: str = "cuda:0",
    replace_incomplete: bool = False,
    runner: Callable[[CarrierContext, str, CandidateConfig, str, Mapping], dict] | None = None,
) -> CandidateEvaluation:
    """Evaluate four isolated branches or return an already verified cache."""
    candidate = CandidateConfig(candidate.semantic_lr, candidate.semantic_exposure)
    path = candidate_directory(context.formal_tuning_root, context.dataset, candidate)
    if (path / "COMPLETE.json").is_file():
        return load_verified_candidate(context, candidate)
    staging_paths = sorted(path.parent.glob(f".{path.name}.staging_*")) if path.parent.exists() else []
    if (path.exists() or staging_paths) and not replace_incomplete:
        raise RuntimeError("INCOMPLETE_CANDIDATE")
    if replace_incomplete:
        if path.exists():
            shutil.rmtree(path)
        for staging in staging_paths:
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()
    bases = {seed: load_verified_base_reference(context, seed) for seed in protocol.DEV_SEEDS}
    results = _evaluate_with_runner(context, candidate, device, runner)
    payload = _evaluation_payload(context, candidate, bases, results)
    audit = {
        "dataset": context.dataset, "candidate_key": candidate_key(candidate),
        "initial_state_hashes": {f"seed{seed}_fold{fold}": results[(seed, fold)]["initial_state_sha256"] for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS},
        "semantic_steps": {f"seed{seed}_fold{fold}": results[(seed, fold)]["semantic_steps"] for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS},
        "native_epochs": {f"seed{seed}_fold{fold}": results[(seed, fold)]["native_epochs"] for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS},
        "U_cycle_frozen": True, "T_rel_frozen": True,
        "a1_validation_objective": amendment_a1.AMENDMENT_ID,
        "full_gt_loaded": False, "full_gt_used": False,
        "refinement_run": True,
    }
    parent = path.parent; parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{path.name}.staging_{uuid.uuid4().hex}"
    staging.mkdir()
    evaluation_path, audit_path = staging / "candidate_evaluation.json", staging / "audit.json"
    evaluation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = {
        "status": "COMPLETE", "dataset": context.dataset,
        "semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
        "evaluation_sha256": sha256_file(evaluation_path), "audit_sha256": sha256_file(audit_path),
        "full_gt_used": False, "protocol_fingerprint": context.protocol_fingerprint,
    }
    (staging / "COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(staging, path)
    return load_verified_candidate(context, candidate)


# Explicit names used in reports/tests.
evaluate_formal_candidate = evaluate_candidate
verify_complete_candidate = load_verified_candidate
