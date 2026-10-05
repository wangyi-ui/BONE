"""Single amended MSRC seed11 refinement smoke reusing the frozen carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation.p3_c1_b1_optuna import (
    amendment_a1, parent_integrity, protocol, sparse_cv, trial_runner,
)
from experiments.utility_relation.p3_c1_b1_optuna.msrc_adapter import (
    MSRC_SPEC, _load_frozen_msrc_inputs, validate_msrc_request,
)
from experiments.utility_relation.p3_c1_b1_optuna.trial_runner_a1 import run_single_candidate_a1
from experiments.weak_quality import bone_p1_msrc_protocol as p1_protocol


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_OUTPUT_ROOT = Path("outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_a1_smoke")
EXPECTED_CARRIER_STATE = Path(
    "outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke/"
    "msrc/dev_seed11/carrier/carrier_model_state.pt"
)
EXPECTED_CARRIER_PRE_GT = EXPECTED_CARRIER_STATE.with_name("carrier_pre_gt.npz")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs() -> None:
    freeze = REPO_ROOT / "experiment_freeze"
    manifests = (
        freeze / "bone_p3_c1_b1_optuna_preregistered_20261003/b1_source_sha256.txt",
        freeze / "bone_p3_c1_b1_optuna_preregistered_20261003/parent_source_sha256.txt",
        freeze / "bone_p3_c1_b1_optuna_implementation_20261003/implementation_source_sha256.txt",
        freeze / "bone_p3_c1_b1_optuna_implementation_20261003/parent_source_sha256.txt",
        freeze / "bone_p3_c1_b1_gpu_bridge_smoke_20261003/bridge_source_sha256.txt",
        freeze / "bone_p3_c1_b1_gpu_bridge_smoke_20261003/smoke_result_sha256.txt",
        freeze / "bone_p3_c1_b1_real_evaluator_smoke_20261003/new_source_sha256.txt",
        freeze / "bone_p3_c1_b1_real_evaluator_smoke_20261003/result_sha256.txt",
        freeze / "bone_p3_c1_b1_msrc_zero_utility_diagnosis_20261003/source_sha256.txt",
        freeze / "bone_p3_c1_b1_msrc_zero_utility_diagnosis_20261003/diagnosis_sha256.txt",
        freeze / "bone_p3_c1_b1_a1_utility_abstention_preregistered_20261003/amendment_source_sha256.txt",
        freeze / "bone_p3_c1_b1_a1_utility_abstention_preregistered_20261003/base_b1_source_sha256.txt",
        freeze / "bone_p3_c1_b0_transfer_validation_final_20261002/final_source_sha256.txt",
        freeze / "bone_p3_c1_b0_transfer_validation_final_20261002/freeze_core_sha256.txt",
    )
    for manifest in manifests:
        parent_integrity.verify_manifest(manifest, REPO_ROOT)


def load_reused_carrier(state_path: Path, pre_gt_path: Path) -> tuple[dict, dict]:
    """Load the frozen MSRC carrier and reconstruct only its read-only runtime context."""
    if state_path != EXPECTED_CARRIER_STATE or pre_gt_path != EXPECTED_CARRIER_PRE_GT:
        raise ValueError("ONLY_FROZEN_MSRC_SEED11_CARRIER_IS_ALLOWED")
    before = {"state": _sha256(state_path), "pre_gt": _sha256(pre_gt_path)}
    required = (
        "q_views", "p_gen", "p_ver", "U_cycle", "generator_actions",
        "verifier_actions", "sample_ids", "labeled_ids", "labeled_targets",
    )
    with np.load(pre_gt_path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"MISSING_CARRIER_KEYS:{sorted(missing)}")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    views, frozen_ids, frozen_targets, _, _ = _load_frozen_msrc_inputs()
    if not np.array_equal(arrays["labeled_ids"], frozen_ids):
        raise RuntimeError("SPARSE_ID_MISMATCH")
    if not np.array_equal(arrays["labeled_targets"], frozen_targets):
        raise RuntimeError("SPARSE_TARGET_MISMATCH")
    adapter = b0.SampleIdAdapter(arrays["sample_ids"])
    folds = sparse_cv.build_sparse_label_folds(
        frozen_ids, frozen_targets, expected_class_ids=range(p1_protocol.K)
    )
    carrier = {
        "views": views,
        "adapter": adapter,
        "spec": MSRC_SPEC,
        "k": p1_protocol.K,
        "state_path": state_path,
        "carrier_path": pre_gt_path,
        "folds": folds,
        "arrays": arrays,
    }
    return carrier, before


def build_summary(carrier: dict, carrier_hashes: dict, feasibility: dict, trial: dict) -> dict:
    schedule, scores, coverage = trial["schedule"], trial["scores"], trial["coverage"]
    payloads = trial["fold_payloads"]
    fold_a_ids = payloads["A"]["validation_raw_ids"].tolist()
    row175_index = fold_a_ids.index(175)
    summary = {
        "dataset": "MSRC", "dev_seed": 11,
        "semantic_lr": 1e-5, "semantic_exposure": 5,
        "carrier_reused": True, "carrier_training_run": False,
        "carrier_epochs_already_completed": 100,
        "carrier_state_path": str(EXPECTED_CARRIER_STATE),
        "carrier_pre_gt_path": str(EXPECTED_CARRIER_PRE_GT),
        "carrier_state_sha256_before": carrier_hashes["state"],
        "carrier_pre_gt_sha256_before": carrier_hashes["pre_gt"],
        "carrier_state_sha256_after": _sha256(EXPECTED_CARRIER_STATE),
        "carrier_pre_gt_sha256_after": _sha256(EXPECTED_CARRIER_PRE_GT),
        "class_feasibility_pass": feasibility["all_classes_feasible"],
        "class_feasibility": feasibility,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "initial_base_state_sha256": schedule.initial_hashes["base"],
        "initial_ours_foldA_state_sha256": schedule.initial_hashes["A"],
        "initial_ours_foldB_state_sha256": schedule.initial_hashes["B"],
        "initial_state_parity": len(set(schedule.initial_hashes.values())) == 1,
        "base_native_epochs": schedule.native_counts["base"],
        "ours_foldA_semantic_steps": schedule.semantic_counts["A"],
        "ours_foldA_native_epochs": schedule.native_counts["A"],
        "ours_foldB_semantic_steps": schedule.semantic_counts["B"],
        "ours_foldB_native_epochs": schedule.native_counts["B"],
        "fold_A_supported_anchor_count": coverage["A"]["supported_anchor_count"],
        "fold_A_abstained_anchor_count": coverage["A"]["abstained_anchor_count"],
        "fold_A_coverage": coverage["A"]["coverage"],
        "fold_A_denominator": coverage["A"]["fold_denominator"],
        "fold_B_supported_anchor_count": coverage["B"]["supported_anchor_count"],
        "fold_B_abstained_anchor_count": coverage["B"]["abstained_anchor_count"],
        "fold_B_coverage": coverage["B"]["coverage"],
        "fold_B_denominator": coverage["B"]["fold_denominator"],
        "raw175_fold_A_role": "validation_anchor",
        "raw175_fold_A_utility_mass": float(payloads["A"]["validation_utility_mass"][row175_index]),
        "raw175_fold_A_J_BASE": scores["A"]["base_anchor_scores"][row175_index],
        "raw175_fold_A_J_OURS": scores["A"]["ours_anchor_scores"][row175_index],
        "raw175_fold_A_abstained": coverage["A"]["anchor_utility_masses"][row175_index] == 0,
        "raw175_fold_B_role": "training_anchor",
        "raw175_fold_B_removed": 175 not in payloads["B"]["train_anchor_raw_ids"],
        "J_BASE_A": scores["A"]["J_BASE"], "J_OURS_A": scores["A"]["J_OURS"],
        "DeltaJ_A": scores["A"]["DeltaJ"],
        "J_BASE_B": scores["B"]["J_BASE"], "J_OURS_B": scores["B"]["J_OURS"],
        "DeltaJ_B": scores["B"]["DeltaJ"],
        "U_carrier_array_sha256": trial_runner._array_hash(carrier["arrays"]["U_cycle"]),
        "U_carrier_frozen": True,
        "SMOKE_PARTIAL_OBJECTIVE_ONLY": True,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False,
        "full_gt_used_for_selection": False,
        "optuna_study_created": False, "formal_tuning_run": False,
    }
    finite = [summary[key] for key in (
        "J_BASE_A", "J_OURS_A", "DeltaJ_A", "J_BASE_B", "J_OURS_B", "DeltaJ_B"
    )]
    required_counts = (
        summary["base_native_epochs"], summary["ours_foldA_semantic_steps"],
        summary["ours_foldA_native_epochs"], summary["ours_foldB_semantic_steps"],
        summary["ours_foldB_native_epochs"],
    )
    if not np.isfinite(finite).all() or required_counts != (20, 5, 20, 5, 20):
        raise RuntimeError("A1_FINITE_OR_SCHEDULE_GATE_FAILURE")
    if not summary["initial_state_parity"] or not summary["class_feasibility_pass"]:
        raise RuntimeError("A1_PARITY_OR_FEASIBILITY_GATE_FAILURE")
    if (summary["fold_A_supported_anchor_count"], summary["fold_A_abstained_anchor_count"],
        summary["fold_B_supported_anchor_count"], summary["fold_B_abstained_anchor_count"]) != (6, 1, 7, 0):
        raise RuntimeError("A1_COVERAGE_GATE_FAILURE")
    if summary["fold_A_denominator"] != 7 or summary["fold_B_denominator"] != 7:
        raise RuntimeError("A1_FOLD_DENOMINATOR_FAILURE")
    if not (summary["raw175_fold_A_utility_mass"] == 0
            and summary["raw175_fold_A_J_BASE"] == 0
            and summary["raw175_fold_A_J_OURS"] == 0
            and summary["raw175_fold_A_abstained"]
            and not summary["raw175_fold_B_removed"]):
        raise RuntimeError("A1_RAW175_GATE_FAILURE")
    if not (summary["carrier_state_sha256_before"] == summary["carrier_state_sha256_after"]
            and summary["carrier_pre_gt_sha256_before"] == summary["carrier_pre_gt_sha256_after"]):
        raise RuntimeError("CARRIER_INPUT_MUTATION")
    return summary


def run(dataset: str, dev_seed: int, semantic_lr: float, semantic_exposure: int,
        device: str, carrier_state: Path, carrier_pre_gt: Path, output_root: Path) -> Path:
    validate_msrc_request(dataset, dev_seed, semantic_lr, semantic_exposure, device)
    if output_root != EXPECTED_OUTPUT_ROOT:
        raise ValueError("INVALID_A1_SMOKE_OUTPUT_ROOT")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    verify_frozen_inputs()
    carrier, carrier_hashes = load_reused_carrier(carrier_state, carrier_pre_gt)
    labeled_rows = carrier["adapter"].row_positions(carrier["arrays"]["labeled_ids"])
    labeled_masses = carrier["arrays"]["U_cycle"][labeled_rows].sum(axis=1)
    feasibility = amendment_a1.validate_class_feasibility(
        carrier["arrays"]["labeled_targets"], labeled_masses,
        expected_class_ids=range(carrier["k"]),
    )
    output_dir = output_root / "msrc" / "dev_seed11" / "lr1e-5_es5"
    trial = run_single_candidate_a1(carrier, semantic_lr, semantic_exposure, output_dir, device)
    summary = build_summary(carrier, carrier_hashes, feasibility, trial)
    summary_path = output_dir / "msrc_a1_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_frozen_inputs()
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev-seed", required=True, type=int)
    parser.add_argument("--semantic-lr", required=True, type=float)
    parser.add_argument("--semantic-exposure", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--carrier-state", required=True, type=Path)
    parser.add_argument("--carrier-pre-gt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    path = run(
        args.dataset, args.dev_seed, args.semantic_lr, args.semantic_exposure,
        args.device, args.carrier_state, args.carrier_pre_gt, args.output_root,
    )
    print(json.dumps({"msrc_a1_smoke_summary": str(path)}, indent=2))


if __name__ == "__main__":
    main()
