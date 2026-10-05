"""Frozen B2 local-grid and selection metadata; no data/model loading."""
from __future__ import annotations

from dataclasses import dataclass

DATASETS = ("CCV", "MSRC", "ALOI")
DEV_SEEDS = (11, 37)
FOLDS = ("A", "B")
REFINEMENT_EPOCHS = 20
TIE_DECIMALS = 12
LOCAL_GRIDS = {
    "CCV": {"semantic_lr": (3e-6, 5e-6, 1e-5, 2e-5, 3e-5), "semantic_exposure": (1, 2, 3, 4, 5)},
    "MSRC": {"semantic_lr": (3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3), "semantic_exposure": (1, 2, 3)},
    "ALOI": {"semantic_lr": (3e-5, 5e-5, 1e-4, 2e-4, 3e-4), "semantic_exposure": (1, 2, 3, 5, 10, 20)},
}
B1_SELECTED = {"CCV": (1e-5, 3), "MSRC": (1e-3, 1), "ALOI": (1e-4, 1)}


@dataclass(frozen=True)
class Candidate:
    semantic_lr: float
    semantic_exposure: int

    def key(self) -> str:
        return f"semantic_lr={self.semantic_lr:.17g};semantic_exposure={self.semantic_exposure}"


def candidates(dataset: str) -> tuple[Candidate, ...]:
    if dataset not in DATASETS:
        raise ValueError("B2_UNREGISTERED_DATASET")
    grid = LOCAL_GRIDS[dataset]
    result = tuple(Candidate(lr, exposure) for lr in grid["semantic_lr"] for exposure in grid["semantic_exposure"])
    if len(result) != len(set(result)):
        raise RuntimeError("B2_DUPLICATE_CANDIDATE")
    return result


def selection_key(candidate: Candidate, score: dict) -> tuple:
    """Exact B1 lexicographic ordering, including 12-decimal canonicalization."""
    return (round(float(score["mean_delta_J"]), TIE_DECIMALS), round(float(score["worst_delta_J"]), TIE_DECIMALS),
            int(score["positive_count"]), -candidate.semantic_exposure, -candidate.semantic_lr)


def select_best(records: list[dict]) -> dict:
    if not records:
        raise ValueError("B2_EMPTY_SELECTION")
    return max(records, key=lambda item: selection_key(item["candidate"], item))


def validate_protocol() -> None:
    expected = {"CCV": 25, "MSRC": 18, "ALOI": 30}
    if {name: len(candidates(name)) for name in DATASETS} != expected or sum(expected.values()) != 73:
        raise RuntimeError("B2_GRID_CARDINALITY_MISMATCH")
    for dataset, config in B1_SELECTED.items():
        if Candidate(*config) not in candidates(dataset):
            raise RuntimeError("B1_SELECTED_NOT_IN_B2_GRID")
    if Candidate(1e-4, 20) not in candidates("ALOI"):
        raise RuntimeError("ALOI_REFERENCE_NOT_IN_B2_GRID")
