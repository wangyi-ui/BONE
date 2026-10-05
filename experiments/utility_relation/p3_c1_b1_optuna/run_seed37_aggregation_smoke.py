"""Run only Caltech dev-seed37, then aggregate it with frozen seed11 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import optim

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_optuna import objective, parent_integrity, protocol
from experiments.utility_relation.p3_c1_b1_optuna.carrier_adapter import (
    build_development_carrier, expected_carrier_dir,
)
from experiments.utility_relation.p3_c1_b1_optuna.real_evaluator import build_candidate_evaluation
from experiments.utility_relation.p3_c1_b1_optuna.study_schema import CandidateConfig
from experiments.utility_relation.p3_c1_b1_optuna import trial_runner


REPO_ROOT = Path(__file__).resolve().parents[3]
PREREG_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_optuna_preregistered_20261003"
IMPLEMENTATION_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_optuna_implementation_20261003"
BRIDGE_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b1_gpu_bridge_smoke_20261003"
B0_FREEZE = REPO_ROOT / "experiment_freeze/bone_p3_c1_b0_transfer_validation_final_20261002"


def verify_frozen_inputs() -> None:
    for manifest in (
        PREREG_FREEZE / "b1_source_sha256.txt",
        PREREG_FREEZE / "parent_source_sha256.txt",
        IMPLEMENTATION_FREEZE / "preregistered_b1_sha256.txt",
        IMPLEMENTATION_FREEZE / "implementation_source_sha256.txt",
        IMPLEMENTATION_FREEZE / "parent_source_sha256.txt",
        BRIDGE_FREEZE / "bridge_source_sha256.txt",
        BRIDGE_FREEZE / "smoke_result_sha256.txt",
        BRIDGE_FREEZE / "parent_source_sha256.txt",
        B0_FREEZE / "final_source_sha256.txt",
        B0_FREEZE / "freeze_core_sha256.txt",
    ):
        parent_integrity.verify_manifest(manifest, REPO_ROOT)


def _normalize_result(summary: dict) -> dict:
    return {
        "dev_seed": summary["dev_seed"],
        "dataset": summary["dataset"],
        "semantic_lr": summary["semantic_lr"],
        "semantic_exposure": summary["semantic_exposure"],
        "delta_j_foldA": summary["DeltaJ_A"],
        "delta_j_foldB": summary["DeltaJ_B"],
        "full_gt_loaded": summary["full_gt_loaded"],
        "carrier_epochs": summary["carrier_epochs"],
        "refinement_horizon": summary["refinement_horizon"],
        "base_native_epochs": summary["base_native_epochs"],
        "ours_foldA_semantic_steps": summary["ours_foldA_semantic_steps"],
        "ours_foldA_native_epochs": summary["ours_foldA_native_epochs"],
        "ours_foldB_semantic_steps": summary["ours_foldB_semantic_steps"],
        "ours_foldB_native_epochs": summary["ours_foldB_native_epochs"],
        "initial_state_parity": summary["initial_state_parity"],
        "zero_utility_foldA": summary["fold_A_zero_utility_mass"],
        "zero_utility_foldB": summary["fold_B_zero_utility_mass"],
    }


def load_frozen_seed11_result() -> dict:
    """Read seed11 only from the verified immutable bridge freeze."""
    verify_frozen_inputs()
    frozen_summary = BRIDGE_FREEZE / "smoke_summary.json"
    result_manifest = parent_integrity.read_manifest(BRIDGE_FREEZE / "smoke_result_sha256.txt")
    summary_entries = [(path, digest) for path, digest in result_manifest.items() if path.endswith("/smoke_summary.json")]
    if len(summary_entries) != 1 or parent_integrity.sha256_file(frozen_summary) != summary_entries[0][1]:
        raise RuntimeError("FROZEN_SEED11_SUMMARY_HASH_MISMATCH")
    summary = json.loads(frozen_summary.read_text(encoding="utf-8"))
    expected = {
        "J_BASE_A": 0.7259641190931416,
        "J_OURS_A": 0.5093962019591405,
        "DeltaJ_A": -0.21656791713400114,
        "J_BASE_B": 0.8288661980454906,
        "J_OURS_B": 0.9133838935436529,
        "DeltaJ_B": 0.08451769549816235,
    }
    if any(summary.get(name) != value for name, value in expected.items()):
        raise RuntimeError("FROZEN_SEED11_CONSISTENCY_FAILURE")
    return _normalize_result(summary)


def seed37_carrier_dir(output_root: str | Path) -> Path:
    return expected_carrier_dir(output_root, "Caltech-5V", protocol.DEV_SEEDS[1])


def _run_seed37_refinement(carrier: dict, candidate: CandidateConfig, output_dir: Path) -> dict:
    seed = protocol.DEV_SEEDS[1]
    views, adapter, spec, k = carrier["views"], carrier["adapter"], carrier["spec"], carrier["k"]
    device = torch.device("cuda:0")
    fold_payloads = trial_runner.build_fold_semantics(carrier["arrays"], adapter, carrier["folds"])
    frozen_before = {fold: payload["frozen_hash"] for fold, payload in fold_payloads.items()}

    def load_branch():
        model = c0._new_model(views, k, spec, seed, device)
        model.load_state_dict(c0._load_state(carrier["state_path"], device), strict=True)
        return model

    full_views = [torch.from_numpy(view).to(device=device, dtype=torch.float32) for view in views]
    semantic_orders = {
        fold: np.stack([
            np.random.default_rng(seed + epoch).permutation(payload["query_rows"].size)
            for epoch in range(candidate.semantic_exposure)
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

    schedule = trial_runner.execute_isolated_schedule(
        load_branch=load_branch,
        state_hash=c0._state_hash,
        make_native_optimizer=lambda model: optim.Adam(model.parameters(), lr=c0_protocol.NATIVE_LR),
        make_semantic_optimizer=lambda model, lr: optim.Adam(model.parameters(), lr=lr),
        make_native_loader=lambda: c0._native_loader(views, adapter, seed, spec.batch_size),
        native_epoch=lambda model, optimizer, loader: c0._native_epoch(model, optimizer, loader, device),
        semantic_epoch=semantic_step,
        semantic_lr=candidate.semantic_lr,
        semantic_exposure=candidate.semantic_exposure,
    )
    if any(payload["frozen_hash"] != frozen_before[fold] for fold, payload in fold_payloads.items()):
        raise RuntimeError("FROZEN_SEMANTIC_TENSOR_MUTATION")
    if any(value != carrier["state_sha256"] for value in schedule.initial_hashes.values()):
        raise RuntimeError("SEED37_INITIAL_STATE_PARITY_FAILURE")

    scores = {}
    for fold, payload in fold_payloads.items():
        base_pi = trial_runner._validation_pi(
            schedule.models["base"], views, device, payload["validation_rows"], payload["class_signature"]
        )
        ours_pi = trial_runner._validation_pi(
            schedule.models[fold], views, device, payload["validation_rows"], payload["class_signature"]
        )
        base_j = objective.validation_j(payload["U_validation"], base_pi, payload["validation_targets"])
        ours_j = objective.validation_j(payload["U_validation"], ours_pi, payload["validation_targets"])
        scores[fold] = {
            "J_BASE": base_j, "J_OURS": ours_j,
            "DeltaJ": objective.paired_delta_j(base_j, ours_j),
        }

    for name in ("base", "ours_foldA", "ours_foldB", "audit"):
        (output_dir / name).mkdir(parents=True, exist_ok=False)
    c0._save_state(schedule.models["base"], output_dir / "base/final_model_state.pt")
    c0._save_state(schedule.models["A"], output_dir / "ours_foldA/final_model_state.pt")
    c0._save_state(schedule.models["B"], output_dir / "ours_foldB/final_model_state.pt")
    summary = {
        "dataset": "Caltech-5V", "dev_seed": seed,
        "semantic_lr": candidate.semantic_lr, "semantic_exposure": candidate.semantic_exposure,
        "carrier_epochs": protocol.CARRIER_EPOCHS, "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "base_native_epochs": schedule.native_counts["base"],
        "ours_foldA_native_epochs": schedule.native_counts["A"],
        "ours_foldA_semantic_steps": schedule.semantic_counts["A"],
        "ours_foldB_native_epochs": schedule.native_counts["B"],
        "ours_foldB_semantic_steps": schedule.semantic_counts["B"],
        "initial_base_state_sha256": schedule.initial_hashes["base"],
        "initial_ours_A_state_sha256": schedule.initial_hashes["A"],
        "initial_ours_B_state_sha256": schedule.initial_hashes["B"],
        "initial_state_parity": len(set(schedule.initial_hashes.values())) == 1,
        "fold_A_zero_utility_mass": bool(np.any(fold_payloads["A"]["validation_utility_mass"] <= 0)),
        "fold_B_zero_utility_mass": bool(np.any(fold_payloads["B"]["validation_utility_mass"] <= 0)),
        "J_BASE_A": scores["A"]["J_BASE"], "J_OURS_A": scores["A"]["J_OURS"], "DeltaJ_A": scores["A"]["DeltaJ"],
        "J_BASE_B": scores["B"]["J_BASE"], "J_OURS_B": scores["B"]["J_OURS"], "DeltaJ_B": scores["B"]["DeltaJ"],
        "full_gt_loaded": False, "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False, "full_gt_used_for_selection": False,
        "optuna_study_created": False, "formal_tuning_run": False,
    }
    numeric = tuple(summary[name] for name in ("J_BASE_A", "J_OURS_A", "DeltaJ_A", "J_BASE_B", "J_OURS_B", "DeltaJ_B"))
    if not np.isfinite(numeric).all():
        raise RuntimeError("NONFINITE_SEED37_OBJECTIVE")
    summary_path = output_dir / "seed37_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "summary_path": summary_path}


def run(output_root: str | Path) -> Path:
    verify_frozen_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    output_root = Path(output_root)
    if not str(output_root).endswith("outputs/bone_p3/p3_c1_b1_optuna/engineering_evaluator_smoke"):
        raise ValueError("INVALID_EVALUATOR_SMOKE_OUTPUT_ROOT")
    candidate = CandidateConfig(1e-5, 5)
    seed11_result = load_frozen_seed11_result()
    carrier_dir = seed37_carrier_dir(output_root)
    carrier = build_development_carrier("Caltech-5V", protocol.DEV_SEEDS[1], "cuda:0", carrier_dir)
    seed37_output = carrier_dir.parent / "lr1e-5_es5"
    seed37_run = _run_seed37_refinement(carrier, candidate, seed37_output)
    seed37_result = _normalize_result(seed37_run["summary"])
    evaluation = build_candidate_evaluation(
        "Caltech-5V", candidate.semantic_lr, candidate.semantic_exposure, seed11_result, seed37_result
    )
    summary = {
        "dataset": evaluation.dataset,
        "semantic_lr": evaluation.semantic_lr,
        "semantic_exposure": evaluation.semantic_exposure,
        "seed11_source": "frozen_gpu_bridge_smoke",
        "delta_j_seed11_foldA": evaluation.delta_j_seed11_foldA,
        "delta_j_seed11_foldB": evaluation.delta_j_seed11_foldB,
        "delta_j_seed37_foldA": evaluation.delta_j_seed37_foldA,
        "delta_j_seed37_foldB": evaluation.delta_j_seed37_foldB,
        "mean_delta_j": evaluation.mean_delta_j,
        "worst_delta_j": evaluation.worst_delta_j,
        "positive_count": evaluation.positive_count,
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "dev_seeds_complete": list(protocol.DEV_SEEDS),
        "candidate_evaluation_complete": True,
        "REAL_CANDIDATE_EVALUATION_CONSTRUCTED": True,
        "full_gt_loaded": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_validation": False,
        "full_gt_used_for_selection": False,
        "optuna_study_created": False,
        "formal_tuning_run": False,
    }
    summary_path = output_root / "caltech_5v/real_candidate_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_frozen_inputs()
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps({"real_candidate_evaluation_summary": str(run(args.output_root))}, indent=2))


if __name__ == "__main__":
    main()
