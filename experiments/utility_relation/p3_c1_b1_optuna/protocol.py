"""Frozen, dataset-invariant scientific protocol for P3-C1-B1.

This module deliberately contains metadata and pure validation helpers only.  It
does not load data, run models, access ground truth, or import Optuna.
"""

from dataclasses import dataclass
from typing import Mapping


DATASETS = (
    "HandWrite", "BDGP", "MSRC", "COIL20", "Caltech-5V", "MNIST-USPS",
    "ALOI", "CCV",
)
CARRIER_EPOCHS = 100
REFINEMENT_HORIZON = 20
SEMANTIC_LR_GRID = (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
SEMANTIC_EXPOSURE_GRID = (1, 3, 5, 10, 20)
DEV_SEEDS = (11, 37)
FORMAL_SEEDS = (20, 30, 50)
FOLDS = ("A", "B")
TIE_ROUND_DECIMALS = 12
FUTURE_OPTUNA_SAMPLER = "GridSampler"
ZERO_UTILITY_POLICY = "FAIL_CLOSED"
SEED_IS_A_HYPERPARAMETER = False
SEMANTIC_EXPOSURE_UPPER_BOUND = REFINEMENT_HORIZON


@dataclass(frozen=True, order=True)
class Candidate:
    """One registered scientific intervention; seeds are never candidates."""

    semantic_lr: float
    semantic_exposure: int


def validate_semantic_exposure(exposure: int) -> int:
    """Reject nonpositive, over-horizon, and unregistered exposures."""
    if isinstance(exposure, bool) or not isinstance(exposure, int):
        raise ValueError("INVALID_SEMANTIC_EXPOSURE")
    if exposure <= 0 or exposure > REFINEMENT_HORIZON:
        raise ValueError("INVALID_SEMANTIC_EXPOSURE")
    if exposure not in SEMANTIC_EXPOSURE_GRID:
        raise ValueError("UNREGISTERED_SEMANTIC_EXPOSURE")
    return exposure


def main_search_candidates() -> tuple[Candidate, ...]:
    """Return the common 5 x 5 candidate grid for every dataset."""
    candidates = tuple(
        Candidate(lr, exposure)
        for lr in SEMANTIC_LR_GRID
        for exposure in SEMANTIC_EXPOSURE_GRID
    )
    if len(candidates) != 25 or len(set(candidates)) != 25:
        raise RuntimeError("INVALID_MAIN_SEARCH_GRID")
    return candidates


def refinement_schedule(exposure: int) -> tuple[int, int]:
    """Return ``(semantic_then_native_epochs, native_only_epochs)``."""
    exposure = validate_semantic_exposure(exposure)
    return exposure, REFINEMENT_HORIZON - exposure


def required_lr_extension(best_lrs_by_dataset: Mapping[str, float]) -> tuple[float, ...]:
    """Return the one globally shared boundary extension, if one is required."""
    if set(best_lrs_by_dataset) != set(DATASETS):
        raise ValueError("BEST_LRS_MUST_COVER_EXACTLY_ALL_DATASETS")
    values = tuple(best_lrs_by_dataset.values())
    if any(lr not in SEMANTIC_LR_GRID for lr in values):
        raise ValueError("UNREGISTERED_PROVISIONAL_LR")
    extensions = []
    if SEMANTIC_LR_GRID[0] in values:
        extensions.append(1e-6)
    if SEMANTIC_LR_GRID[-1] in values:
        extensions.append(1e-3)
    return tuple(extensions)


def exposure_upper_bound_hit(exposure: int) -> bool:
    """Record-only rule: B1 never extends exposure above 20."""
    return validate_semantic_exposure(exposure) == SEMANTIC_EXPOSURE_UPPER_BOUND
