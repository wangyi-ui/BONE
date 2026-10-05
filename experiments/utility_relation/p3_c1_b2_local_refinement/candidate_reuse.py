"""Hash/provenance checking for B2 reads of immutable B1 candidate artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.utility_relation.p3_c1_b1_optuna import amendment_a1
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation.p3_c1_b1_optuna.formal_tuning_context_a2 import build_formal_tuning_context_a2
from experiments.utility_relation.p3_c1_b1_optuna.parent_integrity import read_manifest, sha256_file

from .protocol import Candidate, DEV_SEEDS, FOLDS

ROOT = Path(__file__).resolve().parents[3]
B1_ROOT = ROOT / "outputs/bone_p3/p3_c1_b1_optuna/formal_tuning"
B1_DEPENDENCY_MANIFEST = ROOT / "experiment_freeze/bone_p3_c1_b1_a2_coverage_aware_abstention_preregistered_20261003/frozen_dependency_sha256.txt"


def verify_frozen_dependencies() -> dict[str, str]:
    manifest = read_manifest(B1_DEPENDENCY_MANIFEST)
    for relative, expected in manifest.items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"B2_FROZEN_DEPENDENCY_MISMATCH:{relative}")
    return manifest


def _candidate_files(dataset: str) -> list[Path]:
    slug = dataset.lower()
    return sorted((B1_ROOT / slug).glob("**/candidates/*/candidate_evaluation.json"))


def _source_origin(path: Path) -> str:
    return "REUSED_BOUNDARY_EXTENSION" if "boundary_extension_upper_1e3" in path.parts else "REUSED_B1"


def _validated_match(dataset: str, candidate: Candidate, context) -> dict[str, Any] | None:
    # Candidate branch audits record BONE's deterministic parameter-state hash,
    # not the serialized checkpoint-file hash held by the context object.
    state_hashes = {}
    for seed in DEV_SEEDS:
        try:
            checkpoint = torch.load(context.carrier(seed).state_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch fallback
            checkpoint = torch.load(context.carrier(seed).state_path, map_location="cpu")
        state_hashes[seed] = c0_protocol.state_dict_sha256(checkpoint)
    expected_initial = {f"seed{seed}_fold{fold}": state_hashes[seed] for seed in DEV_SEEDS for fold in FOLDS}
    for evaluation_path in _candidate_files(dataset):
        directory = evaluation_path.parent
        marker_path, audit_path = directory / "COMPLETE.json", directory / "audit.json"
        if not marker_path.is_file() or not audit_path.is_file():
            continue
        evaluation, marker, audit = (json.loads(item.read_text()) for item in (evaluation_path, marker_path, audit_path))
        if (marker.get("status") != "COMPLETE" or marker.get("evaluation_sha256") != sha256_file(evaluation_path)
                or marker.get("audit_sha256") != sha256_file(audit_path)):
            continue
        if (evaluation.get("dataset") != dataset or evaluation.get("semantic_lr") != candidate.semantic_lr
                or evaluation.get("semantic_exposure") != candidate.semantic_exposure
                or evaluation.get("protocol_fingerprint") != context.protocol_fingerprint
                or evaluation.get("full_gt_used") is not False or evaluation.get("full_gt_used_for_selection") is not False):
            continue
        if (audit.get("a1_validation_objective") != amendment_a1.AMENDMENT_ID or audit.get("U_cycle_frozen") is not True
                or audit.get("T_rel_frozen") is not True or audit.get("initial_state_hashes") != expected_initial):
            continue
        if (set(audit.get("semantic_steps", {})) != set(expected_initial) or set(audit.get("native_epochs", {})) != set(expected_initial)
                or any(value != candidate.semantic_exposure for value in audit["semantic_steps"].values())
                or any(value != 20 for value in audit["native_epochs"].values())):
            continue
        return {"candidate_origin": _source_origin(evaluation_path), "candidate_path": str(evaluation_path.relative_to(ROOT)),
                "evaluation_sha256": sha256_file(evaluation_path), "audit_sha256": sha256_file(audit_path),
                "protocol_fingerprint": context.protocol_fingerprint,
                "carrier_state_sha256": {str(seed): state_hashes[seed] for seed in DEV_SEEDS},
                "carrier_pre_gt_sha256": {str(seed): context.carrier(seed).pre_gt_sha256 for seed in DEV_SEEDS},
                "evaluation": evaluation}
    return None


def preflight(dataset: str, local_candidates: tuple[Candidate, ...]) -> dict[str, Any]:
    verify_frozen_dependencies()
    context = build_formal_tuning_context_a2(dataset, B1_ROOT)
    entries = []
    for candidate in local_candidates:
        reusable = _validated_match(dataset, candidate, context)
        entries.append({"semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
                        "reusable": reusable is not None, "provenance": reusable})
    return {"dataset": dataset, "protocol_fingerprint": context.protocol_fingerprint, "candidate_count": len(entries),
            "reusable_count": sum(entry["reusable"] for entry in entries), "new_count": sum(not entry["reusable"] for entry in entries),
            "entries": entries}
