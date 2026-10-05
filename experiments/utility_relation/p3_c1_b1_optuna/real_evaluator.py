"""Pure integration from two real dev-seed results to frozen CandidateEvaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from . import objective, protocol
from .study_schema import CandidateConfig, CandidateEvaluation


REQUIRED_RESULT_FIELDS = (
    "dev_seed", "dataset", "semantic_lr", "semantic_exposure",
    "delta_j_foldA", "delta_j_foldB", "full_gt_loaded",
    "carrier_epochs", "refinement_horizon", "base_native_epochs",
    "ours_foldA_semantic_steps", "ours_foldA_native_epochs",
    "ours_foldB_semantic_steps", "ours_foldB_native_epochs",
    "initial_state_parity", "zero_utility_foldA", "zero_utility_foldB",
)


def _validate_seed_result(result: Mapping, expected_seed: int, candidate: CandidateConfig) -> None:
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        raise ValueError(f"MISSING_REAL_RESULT_FIELDS:{','.join(missing)}")
    if result["dev_seed"] != expected_seed:
        raise ValueError("WRONG_DEV_SEED")
    if result["dataset"] != "Caltech-5V":
        raise ValueError("WRONG_DATASET")
    if (result["semantic_lr"], result["semantic_exposure"]) != (
        candidate.semantic_lr, candidate.semantic_exposure,
    ):
        raise ValueError("CANDIDATE_MISMATCH")
    expected_counts = {
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "base_native_epochs": protocol.REFINEMENT_HORIZON,
        "ours_foldA_semantic_steps": candidate.semantic_exposure,
        "ours_foldA_native_epochs": protocol.REFINEMENT_HORIZON,
        "ours_foldB_semantic_steps": candidate.semantic_exposure,
        "ours_foldB_native_epochs": protocol.REFINEMENT_HORIZON,
    }
    if any(result[name] != expected for name, expected in expected_counts.items()):
        raise ValueError("EPOCH_COUNT_MISMATCH")
    if result["initial_state_parity"] is not True:
        raise ValueError("INITIAL_STATE_PARITY_FAILURE")
    if result["zero_utility_foldA"] is not False or result["zero_utility_foldB"] is not False:
        raise ValueError("ZERO_UTILITY_MASS")
    if result["full_gt_loaded"] is not False:
        raise ValueError("FULL_GT_FIREWALL_FAILURE")
    if not np.isfinite((result["delta_j_foldA"], result["delta_j_foldB"])).all():
        raise ValueError("NONFINITE_DELTA_J")


def build_candidate_evaluation(
    dataset: str,
    semantic_lr: float,
    semantic_exposure: int,
    seed11_result: Mapping,
    seed37_result: Mapping,
) -> CandidateEvaluation:
    """Validate two real runs and delegate all four-score science to frozen code."""
    if dataset != "Caltech-5V" or semantic_lr != 1e-5 or semantic_exposure != 5:
        raise ValueError("ONLY_CALTECH_REGISTERED_INTEGRATION_SMOKE_IS_ALLOWED")
    candidate = CandidateConfig(semantic_lr, semantic_exposure)
    _validate_seed_result(seed11_result, protocol.DEV_SEEDS[0], candidate)
    _validate_seed_result(seed37_result, protocol.DEV_SEEDS[1], candidate)
    evaluation = CandidateEvaluation(
        dataset=dataset,
        semantic_lr=candidate.semantic_lr,
        semantic_exposure=candidate.semantic_exposure,
        delta_j_seed11_foldA=float(seed11_result["delta_j_foldA"]),
        delta_j_seed11_foldB=float(seed11_result["delta_j_foldB"]),
        delta_j_seed37_foldA=float(seed37_result["delta_j_foldA"]),
        delta_j_seed37_foldB=float(seed37_result["delta_j_foldB"]),
    )
    aggregated = objective.aggregate_candidate_scores(evaluation.delta_scores())
    if (
        evaluation.mean_delta_j != aggregated["mean_delta_J"]
        or evaluation.worst_delta_j != aggregated["worst_delta_J"]
        or evaluation.positive_count != aggregated["positive_count"]
    ):
        raise RuntimeError("FROZEN_AGGREGATION_MISMATCH")
    return evaluation
