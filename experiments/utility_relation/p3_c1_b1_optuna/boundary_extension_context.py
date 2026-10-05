"""A2 context and isolated cache routing for upper-LR extension candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Callable, Mapping

from . import amendment_a1, objective_a1, protocol
from . import formal_candidate_evaluator_r2 as frozen_evaluator
from .boundary_extension_protocol import (
    BOUNDARY_DIRECTORY_NAME,
    EXTENSION_ID,
    ExtensionCandidateConfig,
    ExtensionCandidateEvaluation,
    extension_candidate_key,
    verify_boundary_extension_preconditions,
)
from .formal_base_reference import BaseReference, load_verified_base_reference
from .formal_tuning_context_a2 import (
    FormalTuningContextA2,
    build_formal_tuning_context_a2,
)
from .parent_integrity import sha256_file


@dataclass(frozen=True)
class BoundaryExtensionContext:
    base_context: FormalTuningContextA2
    extension_root: Path
    precondition_audit: Mapping

    @property
    def dataset(self) -> str:
        return self.base_context.dataset

    @property
    def formal_tuning_root(self) -> Path:
        return self.base_context.formal_tuning_root

    @property
    def protocol_fingerprint(self) -> str:
        return self.base_context.protocol_fingerprint

    @property
    def feasibility(self):
        return self.base_context.feasibility

    def carrier(self, dev_seed: int):
        return self.base_context.carrier(dev_seed)


def build_boundary_extension_context(
    dataset: str,
    formal_tuning_root: str | Path,
    repository_root: str | Path,
    builder: Callable[..., FormalTuningContextA2] | None = None,
) -> BoundaryExtensionContext:
    precondition = verify_boundary_extension_preconditions(repository_root)
    base = (builder or build_formal_tuning_context_a2)(
        dataset, formal_tuning_root
    )
    from .run_formal_optuna_tuning import dataset_slug

    extension_root = (
        Path(formal_tuning_root)
        / dataset_slug(dataset)
        / BOUNDARY_DIRECTORY_NAME
    )
    return BoundaryExtensionContext(base, extension_root, precondition)


def extension_candidate_directory(
    context: BoundaryExtensionContext,
    candidate: ExtensionCandidateConfig,
) -> Path:
    candidate = ExtensionCandidateConfig(
        candidate.semantic_lr, candidate.semantic_exposure
    )
    return (
        context.extension_root
        / "candidates"
        / f"lr_{candidate.semantic_lr:.17g}__es_{candidate.semantic_exposure}"
    )


def _evaluation_payload(
    context: BoundaryExtensionContext,
    candidate: ExtensionCandidateConfig,
    bases: Mapping[int, BaseReference],
    results: Mapping[tuple[int, str], dict],
) -> dict:
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
                "J_BASE": base_j,
                "J_OURS": ours_j,
                "DeltaJ": delta,
                "supported_count": item["audit"]["supported_anchor_count"],
                "abstained_count": item["audit"]["abstained_anchor_count"],
                "coverage": item["audit"]["coverage"],
                "semantic_steps": item["semantic_steps"],
                "native_epochs": item["native_epochs"],
                "initial_state_sha256": item["initial_state_sha256"],
                "full_gt_loaded": False,
            }
    aggregation = objective_a1.aggregate_candidate_scores(deltas)
    return {
        "dataset": context.dataset,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "delta_j_seed11_foldA": deltas[(11, "A")],
        "delta_j_seed11_foldB": deltas[(11, "B")],
        "delta_j_seed37_foldA": deltas[(37, "A")],
        "delta_j_seed37_foldB": deltas[(37, "B")],
        "mean_delta_j": aggregation["mean_delta_J"],
        "worst_delta_j": aggregation["worst_delta_J"],
        "positive_count": aggregation["positive_count"],
        "folds": folds,
        "protocol_fingerprint": context.protocol_fingerprint,
        "extension_id": EXTENSION_ID,
        "full_gt_used": False,
        "full_gt_used_for_selection": False,
        "candidate_key": extension_candidate_key(candidate),
    }


def _verify_extension_candidate_payload(
    path: Path,
    context: BoundaryExtensionContext,
    candidate: ExtensionCandidateConfig,
) -> ExtensionCandidateEvaluation:
    marker_path = path / "COMPLETE.json"
    evaluation_path = path / "candidate_evaluation.json"
    audit_path = path / "audit.json"
    if not marker_path.is_file() or not evaluation_path.is_file() or not audit_path.is_file():
        raise ValueError("EXTENSION_CANDIDATE_INCOMPLETE")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "COMPLETE"
        or marker.get("evaluation_sha256") != sha256_file(evaluation_path)
        or marker.get("audit_sha256") != sha256_file(audit_path)
    ):
        raise ValueError("EXTENSION_CANDIDATE_HASH_MISMATCH")
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        payload.get("dataset") != context.dataset
        or payload.get("candidate_key") != extension_candidate_key(candidate)
        or payload.get("extension_id") != EXTENSION_ID
    ):
        raise ValueError("EXTENSION_CANDIDATE_IDENTITY_MISMATCH")
    if (
        payload.get("protocol_fingerprint") != context.protocol_fingerprint
        or payload.get("full_gt_used") is not False
    ):
        raise ValueError("EXTENSION_CANDIDATE_PROTOCOL_MISMATCH")
    return ExtensionCandidateEvaluation(
        context.dataset,
        candidate.semantic_lr,
        candidate.semantic_exposure,
        payload["delta_j_seed11_foldA"],
        payload["delta_j_seed11_foldB"],
        payload["delta_j_seed37_foldA"],
        payload["delta_j_seed37_foldB"],
    )


def load_verified_extension_candidate(
    context: BoundaryExtensionContext,
    candidate: ExtensionCandidateConfig,
) -> ExtensionCandidateEvaluation:
    return _verify_extension_candidate_payload(
        extension_candidate_directory(context, candidate), context, candidate
    )


def evaluate_extension_candidate(
    context: BoundaryExtensionContext,
    candidate: ExtensionCandidateConfig,
    device: str = "cuda:0",
    replace_incomplete: bool = False,
    runner=None,
) -> ExtensionCandidateEvaluation:
    """Reuse frozen four-branch science with isolated extension persistence."""
    candidate = ExtensionCandidateConfig(
        candidate.semantic_lr, candidate.semantic_exposure
    )
    path = extension_candidate_directory(context, candidate)
    if (path / "COMPLETE.json").is_file():
        return load_verified_extension_candidate(context, candidate)
    staging_paths = sorted(path.parent.glob(f".{path.name}.staging_*")) if path.parent.exists() else []
    if (path.exists() or staging_paths) and not replace_incomplete:
        raise RuntimeError("INCOMPLETE_EXTENSION_CANDIDATE")
    if replace_incomplete:
        if path.exists():
            shutil.rmtree(path)
        for staging in staging_paths:
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()

    bases = {
        seed: load_verified_base_reference(context.base_context, seed)
        for seed in protocol.DEV_SEEDS
    }
    results = frozen_evaluator._evaluate_with_runner(
        context.base_context, candidate, device, runner
    )
    payload = _evaluation_payload(context, candidate, bases, results)
    audit = {
        "dataset": context.dataset,
        "candidate_key": extension_candidate_key(candidate),
        "extension_id": EXTENSION_ID,
        "initial_state_hashes": {
            f"seed{seed}_fold{fold}": results[(seed, fold)]["initial_state_sha256"]
            for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS
        },
        "semantic_steps": {
            f"seed{seed}_fold{fold}": results[(seed, fold)]["semantic_steps"]
            for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS
        },
        "native_epochs": {
            f"seed{seed}_fold{fold}": results[(seed, fold)]["native_epochs"]
            for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS
        },
        "U_cycle_frozen": True,
        "T_rel_frozen": True,
        "a1_validation_objective": amendment_a1.AMENDMENT_ID,
        "a2_feasibility_active": True,
        "full_gt_loaded": False,
        "full_gt_used": False,
        "refinement_run": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.staging_{uuid.uuid4().hex}"
    staging.mkdir()
    evaluation_path = staging / "candidate_evaluation.json"
    audit_path = staging / "audit.json"
    evaluation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "status": "COMPLETE",
        "dataset": context.dataset,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "evaluation_sha256": sha256_file(evaluation_path),
        "audit_sha256": sha256_file(audit_path),
        "full_gt_used": False,
        "protocol_fingerprint": context.protocol_fingerprint,
        "extension_id": EXTENSION_ID,
    }
    (staging / "COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, path)
    return load_verified_extension_candidate(context, candidate)

