"""Fixed pre-GT protocol for the BONE-P2-A0 utility/relation carrier."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


N = 210
V = 4
K = 7
L = 14
NU = N - L
S = 6
TRAINING_SEED = 20
CORRUPTION_SEED = 20
TARGET_SNR_DB = 2.5
FEATURE_DIMS = (24, 512, 256, 254)
GENERATOR_SUBSETS = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
)
VERIFIER_SUBSETS = np.asarray(
    ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)), dtype=np.int64
)
LABELED_IDS = np.asarray((17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201), dtype=np.int64)
EXPECTED_TARGETS = np.asarray((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), dtype=np.int64)
PARENT_ARTIFACT = Path("outputs/bone_p1/msrc_snr2p5_exact2of4_seed20/bone_p1_msrc_corruption.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/bone_p2/p2_a0_msrc_seed20")


def action_subsets() -> Tuple[np.ndarray, np.ndarray]:
    """Return copies of the six pre-registered directional actions."""
    return GENERATOR_SUBSETS.copy(), VERIFIER_SUBSETS.copy()


def validate_action_subsets(generator_subsets: np.ndarray, verifier_subsets: np.ndarray) -> None:
    """Fail closed unless the six two-view actions are strict complements."""
    if generator_subsets.shape != (S, 2) or verifier_subsets.shape != (S, 2):
        raise ValueError("generator and verifier subsets must both have shape (6, 2)")
    if generator_subsets.dtype.kind not in "iu" or verifier_subsets.dtype.kind not in "iu":
        raise ValueError("action subsets must be integer arrays")
    universe = set(range(V))
    for index, (generator, verifier) in enumerate(zip(generator_subsets, verifier_subsets)):
        generator_set, verifier_set = set(map(int, generator)), set(map(int, verifier))
        if len(generator_set) != 2 or len(verifier_set) != 2:
            raise ValueError(f"action {index} must contain two distinct views per side")
        if generator_set & verifier_set or generator_set | verifier_set != universe:
            raise ValueError(f"action {index} generator/verifier sides are not strict complements")
    if not np.array_equal(generator_subsets, GENERATOR_SUBSETS):
        raise ValueError("generator actions do not match the pre-registered enumeration")
    if not np.array_equal(verifier_subsets, VERIFIER_SUBSETS):
        raise ValueError("verifier actions do not match the pre-registered complements")


def unlabeled_complement(labeled_ids: np.ndarray = LABELED_IDS) -> np.ndarray:
    """Return canonical-row complement of the fixed sparse anchor IDs."""
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    if labeled_ids.shape != (L,) or not np.array_equal(labeled_ids, LABELED_IDS):
        raise ValueError("labeled IDs must equal the frozen 14-row sparse split")
    if len(np.unique(labeled_ids)) != L or np.any(labeled_ids < 0) or np.any(labeled_ids >= N):
        raise ValueError("invalid sparse anchor IDs")
    return np.setdiff1d(np.arange(N, dtype=np.int64), labeled_ids, assume_unique=True)


def validate_sparse_targets(targets: np.ndarray) -> None:
    """Fail closed unless the historical two-labels-per-class split matches exactly."""
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape != (L,) or not np.array_equal(targets, EXPECTED_TARGETS):
        raise ValueError(f"sparse split parity failure; actual targets={targets.tolist()}")
    if not np.array_equal(np.bincount(targets, minlength=K), np.full(K, 2)):
        raise ValueError("every class must have exactly two sparse anchors")
