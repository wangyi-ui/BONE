"""Data-only schema for B1 Optuna plumbing; scientific rules stay frozen upstream."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from . import objective, protocol


@dataclass(frozen=True)
class CandidateConfig:
    semantic_lr: float
    semantic_exposure: int

    def __post_init__(self):
        validate_candidate_config(self)


def validate_candidate_config(candidate: CandidateConfig) -> CandidateConfig:
    """Accept exactly one preregistered LR/exposure pair."""
    if candidate.semantic_lr not in protocol.SEMANTIC_LR_GRID:
        raise ValueError("UNREGISTERED_SEMANTIC_LR")
    protocol.validate_semantic_exposure(candidate.semantic_exposure)
    return candidate


def candidate_key(candidate: CandidateConfig) -> str:
    """Canonical stable identity for one registered grid point."""
    validate_candidate_config(candidate)
    return f"semantic_lr={candidate.semantic_lr:.17g};semantic_exposure={candidate.semantic_exposure}"


@dataclass(frozen=True)
class CandidateEvaluation:
    """The four paired sparse-label validation gains and frozen aggregation."""

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
        config = CandidateConfig(self.semantic_lr, self.semantic_exposure)
        deltas = self.delta_scores()
        aggregated = objective.aggregate_candidate_scores(deltas)
        object.__setattr__(self, "mean_delta_j", aggregated["mean_delta_J"])
        object.__setattr__(self, "worst_delta_j", aggregated["worst_delta_J"])
        object.__setattr__(self, "positive_count", aggregated["positive_count"])

    def candidate_config(self) -> CandidateConfig:
        return CandidateConfig(self.semantic_lr, self.semantic_exposure)

    def delta_scores(self) -> dict[tuple[int, str], float]:
        values = (
            self.delta_j_seed11_foldA,
            self.delta_j_seed11_foldB,
            self.delta_j_seed37_foldA,
            self.delta_j_seed37_foldB,
        )
        keys = tuple((seed, fold) for seed in protocol.DEV_SEEDS for fold in protocol.FOLDS)
        if len(values) != len(keys) or not np.isfinite(values).all():
            raise ValueError("NONFINITE_OR_MALFORMED_DELTA_SCORES")
        return dict(zip(keys, map(float, values)))
