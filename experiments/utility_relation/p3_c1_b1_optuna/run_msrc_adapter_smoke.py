"""Opt-in single MSRC/dev-seed11/lr1e-5/es5 GPU adapter smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utility_relation.p3_c1_b1_optuna import objective, parent_integrity, protocol, trial_runner
from experiments.utility_relation.p3_c1_b1_optuna.msrc_adapter import (
    build_msrc_development_carrier, expected_msrc_carrier_dir, validate_msrc_request,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FREEZE_ROOT = REPO_ROOT / "experiment_freeze"


def verify_frozen_inputs() -> None:
    manifests = (
        FREEZE_ROOT / "bone_p3_c1_b1_optuna_preregistered_20261003/b1_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_optuna_preregistered_20261003/parent_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_optuna_implementation_20261003/implementation_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_gpu_bridge_smoke_20261003/bridge_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_gpu_bridge_smoke_20261003/smoke_result_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_real_evaluator_smoke_20261003/new_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b1_real_evaluator_smoke_20261003/result_sha256.txt",
        FREEZE_ROOT / "bone_p3_a3_sparse_label_protocol_20261001/FREEZE_MANIFEST.sha256",
        FREEZE_ROOT / "bone_p3_a4_corruption_protocol_20261001/FREEZE_MANIFEST.sha256",
        FREEZE_ROOT / "bone_p3_b0_generic_carrier_20261001/FREEZE_MANIFEST.sha256",
        FREEZE_ROOT / "bone_p3_c1_b0_transfer_validation_final_20261002/final_source_sha256.txt",
        FREEZE_ROOT / "bone_p3_c1_b0_transfer_validation_final_20261002/freeze_core_sha256.txt",
    )
    for manifest in manifests:
        parent_integrity.verify_manifest(manifest, REPO_ROOT)


def build_msrc_summary(carrier: dict, trial: dict, semantic_lr: float, semantic_exposure: int) -> dict:
    schedule, scores = trial["schedule"], trial["scores"]
    payloads = trial["fold_payloads"]
    summary = {
        "dataset": "MSRC", "dev_seed": protocol.DEV_SEEDS[0],
        "semantic_lr": semantic_lr, "semantic_exposure": semantic_exposure,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "view_count": int(carrier["arrays"]["q_views"].shape[1]),
        "cluster_count": int(carrier["arrays"]["q_views"].shape[2]),
        "action_count": int(carrier["arrays"]["U_cycle"].shape[1]),
        "base_native_epochs": schedule.native_counts["base"],
        "ours_foldA_semantic_steps": schedule.semantic_counts["A"],
        "ours_foldA_native_epochs": schedule.native_counts["A"],
        "ours_foldB_semantic_steps": schedule.semantic_counts["B"],
        "ours_foldB_native_epochs": schedule.native_counts["B"],
        "initial_base_state_sha256": schedule.initial_hashes["base"],
        "initial_ours_foldA_state_sha256": schedule.initial_hashes["A"],
        "initial_ours_foldB_state_sha256": schedule.initial_hashes["B"],
        "initial_state_parity": len(set(schedule.initial_hashes.values())) == 1,
        "U_carrier_frozen": True,
        "fold_A_zero_utility_mass": bool(np.any(payloads["A"]["validation_utility_mass"] <= 0)),
        "fold_B_zero_utility_mass": bool(np.any(payloads["B"]["validation_utility_mass"] <= 0)),
        "J_BASE_A": scores["A"]["J_BASE"], "J_OURS_A": scores["A"]["J_OURS"],
        "DeltaJ_A": scores["A"]["DeltaJ"],
        "J_BASE_B": scores["B"]["J_BASE"], "J_OURS_B": scores["B"]["J_OURS"],
        "DeltaJ_B": scores["B"]["DeltaJ"],
        "SMOKE_PARTIAL_OBJECTIVE_ONLY": True,
        "full_gt_loaded": False, "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False, "full_gt_used_for_selection": False,
        "optuna_study_created": False, "formal_tuning_run": False,
    }
    numeric = tuple(summary[name] for name in ("J_BASE_A", "J_OURS_A", "DeltaJ_A", "J_BASE_B", "J_OURS_B", "DeltaJ_B"))
    expected_counts = (20, 5, 20, 5, 20)
    observed_counts = (
        summary["base_native_epochs"], summary["ours_foldA_semantic_steps"],
        summary["ours_foldA_native_epochs"], summary["ours_foldB_semantic_steps"],
        summary["ours_foldB_native_epochs"],
    )
    if not np.isfinite(numeric).all() or observed_counts != expected_counts:
        raise RuntimeError("MSRC_SMOKE_FINITE_OR_COUNT_GATE_FAILURE")
    if not summary["initial_state_parity"] or summary["fold_A_zero_utility_mass"] or summary["fold_B_zero_utility_mass"]:
        raise RuntimeError("MSRC_SMOKE_PARITY_OR_UTILITY_GATE_FAILURE")
    return summary


def run(dataset: str, dev_seed: int, semantic_lr: float, semantic_exposure: int,
        device: str, output_root: str | Path) -> Path:
    candidate = validate_msrc_request(dataset, dev_seed, semantic_lr, semantic_exposure, device)
    verify_frozen_inputs()
    output_root = Path(output_root)
    if not str(output_root).endswith("outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke"):
        raise ValueError("INVALID_MSRC_SMOKE_OUTPUT_ROOT")
    carrier_dir = expected_msrc_carrier_dir(output_root)
    carrier = build_msrc_development_carrier(dev_seed, device, carrier_dir)
    candidate_dir = carrier_dir.parent / "lr1e-5_es5"
    trial = trial_runner.run_single_candidate(carrier, candidate.semantic_lr, candidate.semantic_exposure, candidate_dir)
    summary = build_msrc_summary(carrier, trial, candidate.semantic_lr, candidate.semantic_exposure)
    summary_path = candidate_dir / "msrc_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_frozen_inputs()
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
    path = run(args.dataset, args.dev_seed, args.semantic_lr, args.semantic_exposure, args.device, args.output_root)
    print(json.dumps({"msrc_smoke_summary": str(path)}, indent=2))


if __name__ == "__main__":
    main()
