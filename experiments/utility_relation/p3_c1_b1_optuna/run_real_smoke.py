"""Opt-in single Caltech/seed11/lr1e-5/es5 GPU engineering smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utility_relation.p3_c1_b1_optuna import parent_integrity, protocol
from experiments.utility_relation.p3_c1_b1_optuna.carrier_adapter import (
    build_development_carrier, expected_carrier_dir, validate_carrier_request,
)
from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateConfig
from experiments.utility_relation.p3_c1_b1_optuna.trial_runner import run_single_candidate


REPO_ROOT = Path(__file__).resolve().parents[3]
PREREG_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_optuna_preregistered_20261003"
IMPLEMENTATION_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_optuna_implementation_20261003"
B0_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b0_transfer_validation_final_20261002"


def verify_parent_integrity_gate() -> None:
    for manifest in (
        PREREG_FREEZE / "b1_source_sha256.txt",
        PREREG_FREEZE / "parent_source_sha256.txt",
        IMPLEMENTATION_FREEZE / "preregistered_b1_sha256.txt",
        IMPLEMENTATION_FREEZE / "implementation_source_sha256.txt",
        IMPLEMENTATION_FREEZE / "parent_source_sha256.txt",
        B0_FREEZE / "final_source_sha256.txt",
        B0_FREEZE / "freeze_core_sha256.txt",
    ):
        parent_integrity.verify_manifest(manifest, REPO_ROOT)


def validate_smoke_args(dataset: str, dev_seed: int, semantic_lr: float, semantic_exposure: int,
                        device: str, output_root: str | Path) -> CandidateConfig:
    validate_carrier_request(dataset, dev_seed, device)
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    if (dataset, dev_seed, candidate.semantic_lr, candidate.semantic_exposure, device) != (
        "Caltech-5V", protocol.DEV_SEEDS[0], 1e-5, 5, "cuda:0"
    ):
        raise ValueError("ONLY_REGISTERED_SINGLE_ENGINEERING_SMOKE_IS_ALLOWED")
    root = Path(output_root)
    if root.parts[-4:] != ("bone_p3", "p3_c1_b1_optuna", "engineering_smoke")[-len(root.parts[-4:]):]:
        if not str(root).endswith("outputs/bone_p3/p3_c1_b1_optuna/engineering_smoke"):
            raise ValueError("INVALID_ENGINEERING_SMOKE_OUTPUT_ROOT")
    return candidate


def run_smoke(dataset: str, dev_seed: int, semantic_lr: float, semantic_exposure: int,
              device: str, output_root: str | Path) -> Path:
    candidate = validate_smoke_args(dataset, dev_seed, semantic_lr, semantic_exposure, device, output_root)
    verify_parent_integrity_gate()
    carrier_dir = expected_carrier_dir(output_root, dataset, dev_seed)
    carrier = build_development_carrier(dataset, dev_seed, device, carrier_dir)
    candidate_dir = carrier_dir.parent / "lr1e-5_es5"
    trial = run_single_candidate(carrier, candidate.semantic_lr, candidate.semantic_exposure, candidate_dir)
    schedule, scores = trial["schedule"], trial["scores"]
    summary = {
        "status": "ENGINEERING_SMOKE_PASS",
        "dataset": dataset,
        "dev_seed": dev_seed,
        "semantic_lr": candidate.semantic_lr,
        "semantic_exposure": candidate.semantic_exposure,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "base_native_epochs": schedule.native_counts["base"],
        "ours_foldA_native_epochs": schedule.native_counts["A"],
        "ours_foldA_semantic_steps": schedule.semantic_counts["A"],
        "ours_foldB_native_epochs": schedule.native_counts["B"],
        "ours_foldB_semantic_steps": schedule.semantic_counts["B"],
        "initial_base_state_sha256": schedule.initial_hashes["base"],
        "initial_ours_A_state_sha256": schedule.initial_hashes["A"],
        "initial_ours_B_state_sha256": schedule.initial_hashes["B"],
        "initial_state_parity": len(set(schedule.initial_hashes.values())) == 1,
        "U_carrier_frozen": True,
        "fold_A_zero_utility_mass": bool(np.any(trial["fold_payloads"]["A"]["validation_utility_mass"] <= 0)),
        "fold_B_zero_utility_mass": bool(np.any(trial["fold_payloads"]["B"]["validation_utility_mass"] <= 0)),
        "J_BASE_A": scores["A"]["J_BASE"],
        "J_OURS_A": scores["A"]["J_OURS"],
        "DeltaJ_A": scores["A"]["DeltaJ"],
        "J_BASE_B": scores["B"]["J_BASE"],
        "J_OURS_B": scores["B"]["J_OURS"],
        "DeltaJ_B": scores["B"]["DeltaJ"],
        "SMOKE_PARTIAL_OBJECTIVE_ONLY": True,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_selection": False,
        "full_gt_used_for_smoke_objective": False,
        "OPTUNA_STUDY_CREATED": False,
    }
    numeric = tuple(summary[name] for name in ("J_BASE_A", "J_OURS_A", "DeltaJ_A", "J_BASE_B", "J_OURS_B", "DeltaJ_B"))
    if not np.isfinite(numeric).all() or not summary["initial_state_parity"]:
        raise RuntimeError("SMOKE_FINITE_OR_PARITY_GATE_FAILED")
    summary_path = candidate_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_parent_integrity_gate()
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev-seed", required=True, type=int)
    parser.add_argument("--semantic-lr", required=True, type=float)
    parser.add_argument("--semantic-exposure", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = run_smoke(args.dataset, args.dev_seed, args.semantic_lr, args.semantic_exposure, args.device, args.output_root)
    print(json.dumps({"smoke_summary": str(path)}, indent=2))


if __name__ == "__main__":
    main()
