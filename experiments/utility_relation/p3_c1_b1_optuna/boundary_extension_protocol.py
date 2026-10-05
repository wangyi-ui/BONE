"""Preregistered single-stage upper-LR boundary extension protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from . import objective_a1, protocol


EXTENSION_ID = "P3-C1-B1-A2-UPPER-LR-1E-3"
EXTENSION_LR = (1e-3,)
SEMANTIC_EXPOSURES = protocol.SEMANTIC_EXPOSURE_GRID
BOUNDARY_DIRECTORY_NAME = "boundary_extension_upper_1e3"
STUDY_DB_NAME = "study_boundary_ext.db"
SECOND_BOUNDARY_EXTENSION_ALLOWED = False

TRIGGER_FREEZE_RELATIVE = Path(
    "experiment_freeze/bone_p3_c1_b1_a2_eight_dataset_main_grid_provisional_selection_20261004"
)
A2_FREEZE_RELATIVE = Path(
    "experiment_freeze/bone_p3_c1_b1_a2_coverage_aware_abstention_preregistered_20261003"
)

REQUIRED_TRIGGER_STATUS = {
    "EIGHT_DATASET_MAIN_GRID_COMPLETE": "true",
    "DATASET_COUNT": "8",
    "CANDIDATES_PER_DATASET": "25",
    "LOWER_BOUNDARY_HIT_DATASETS": "NONE",
    "UPPER_BOUNDARY_HIT_DATASETS": "BDGP",
    "BOUNDARY_EXTENSION_REQUIRED": "true",
    "FINAL_HYPERPARAMETERS_FROZEN": "false",
    "NEXT_STAGE": "PREREGISTERED_LR_BOUNDARY_EXTENSION",
}
REQUIRED_A2_STATUS = {
    "STATUS": "P3_C1_B1_A2_COVERAGE_AWARE_ABSTENTION_FROZEN",
}


@dataclass(frozen=True, order=True)
class ExtensionCandidateConfig:
    semantic_lr: float
    semantic_exposure: int

    def __post_init__(self):
        if self.semantic_lr != EXTENSION_LR[0]:
            raise ValueError("UNREGISTERED_EXTENSION_LR")
        if self.semantic_exposure not in SEMANTIC_EXPOSURES:
            raise ValueError("UNREGISTERED_EXTENSION_EXPOSURE")


@dataclass(frozen=True)
class ExtensionCandidateEvaluation:
    dataset: str
    semantic_lr: float
    semantic_exposure: int
    delta_j_seed11_foldA: float
    delta_j_seed11_foldB: float
    delta_j_seed37_foldA: float
    delta_j_seed37_foldB: float
    mean_delta_j: float = field(init=False)
    worst_delta_j: float = field(init=False)
    positive_count: int = field(init=False)

    def __post_init__(self):
        if self.dataset not in protocol.DATASETS:
            raise ValueError("UNREGISTERED_DATASET")
        config = ExtensionCandidateConfig(
            self.semantic_lr, self.semantic_exposure
        )
        del config
        aggregate = objective_a1.aggregate_candidate_scores(self.delta_scores())
        object.__setattr__(self, "mean_delta_j", aggregate["mean_delta_J"])
        object.__setattr__(self, "worst_delta_j", aggregate["worst_delta_J"])
        object.__setattr__(self, "positive_count", aggregate["positive_count"])

    def candidate_config(self) -> ExtensionCandidateConfig:
        return ExtensionCandidateConfig(
            self.semantic_lr, self.semantic_exposure
        )

    def delta_scores(self) -> dict[tuple[int, str], float]:
        values = (
            self.delta_j_seed11_foldA,
            self.delta_j_seed11_foldB,
            self.delta_j_seed37_foldA,
            self.delta_j_seed37_foldB,
        )
        if not np.isfinite(values).all():
            raise ValueError("NONFINITE_EXTENSION_DELTA_SCORE")
        keys = tuple(
            (seed, fold) for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS
        )
        return dict(zip(keys, map(float, values)))


def extension_candidate_key(candidate: ExtensionCandidateConfig) -> str:
    candidate = ExtensionCandidateConfig(
        candidate.semantic_lr, candidate.semantic_exposure
    )
    return (
        f"semantic_lr={candidate.semantic_lr:.17g};"
        f"semantic_exposure={candidate.semantic_exposure}"
    )


def extension_candidates() -> tuple[ExtensionCandidateConfig, ...]:
    candidates = tuple(
        ExtensionCandidateConfig(lr, exposure)
        for lr in EXTENSION_LR
        for exposure in SEMANTIC_EXPOSURES
    )
    if len(candidates) != 5 or len({extension_candidate_key(c) for c in candidates}) != 5:
        raise RuntimeError("INVALID_EXTENSION_GRID")
    return candidates


def _parse_status(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"MISSING_BOUNDARY_PRECONDITION:{path}")
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"INVALID_STATUS_LINE:{line}")
        key, value = line.split("=", 1)
        result[key] = value
    return result


def _require_status(
    path: Path,
    required: Mapping[str, str],
    error: str,
) -> dict[str, str]:
    status = _parse_status(path)
    if any(status.get(key) != value for key, value in required.items()):
        raise RuntimeError(error)
    return status


def verify_boundary_extension_preconditions(
    repository_root: str | Path,
) -> dict:
    root = Path(repository_root)
    trigger_path = root / TRIGGER_FREEZE_RELATIVE / "STATUS.txt"
    a2_path = root / A2_FREEZE_RELATIVE / "STATUS.txt"
    trigger = _require_status(
        trigger_path,
        REQUIRED_TRIGGER_STATUS,
        "BOUNDARY_TRIGGER_PRECONDITION_FAILED",
    )
    a2 = _require_status(
        a2_path,
        REQUIRED_A2_STATUS,
        "A2_FREEZE_PRECONDITION_FAILED",
    )
    return {
        "extension_id": EXTENSION_ID,
        "trigger_status_path": str(trigger_path),
        "a2_status_path": str(a2_path),
        "lower_boundary_triggered": False,
        "upper_boundary_triggered": True,
        "trigger_datasets": ["BDGP"],
        "extension_grid": [
            {
                "semantic_lr": candidate.semantic_lr,
                "semantic_exposure": candidate.semantic_exposure,
            }
            for candidate in extension_candidates()
        ],
        "dataset_count": len(protocol.DATASETS),
        "candidates_per_dataset": 5,
        "total_extension_candidates": 40,
        "second_boundary_extension_allowed": False,
        "a2_feasibility_active": a2["STATUS"] == REQUIRED_A2_STATUS["STATUS"],
        "full_gt_used": False,
        "trigger_status": trigger,
    }


def extension_protocol_fingerprint() -> str:
    import hashlib

    payload = {
        "extension_id": EXTENSION_ID,
        "extension_lr": EXTENSION_LR,
        "semantic_exposures": SEMANTIC_EXPOSURES,
        "datasets": protocol.DATASETS,
        "dev_seeds": protocol.DEV_SEEDS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "second_boundary_extension_allowed": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

