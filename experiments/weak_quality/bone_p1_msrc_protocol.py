"""Fixed, label-independent MSRC weak-quality corruption protocol for BONE-P1."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np


N = 210
V = 4
K = 7
FEATURE_DIMS = (24, 512, 256, 254)
CORRUPTION_SEED = 20
TARGET_SNR_DB = 2.5
TRAINING_SEED = 20
PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
)
PAIR_REPETITIONS = 35


def pair_key(pair: Tuple[int, int]) -> str:
    """Return the stable JSON key used for a corrupted-view pair."""
    return f"{pair[0]}-{pair[1]}"


def build_corruption_mask(seed: int = CORRUPTION_SEED) -> np.ndarray:
    """Build [N, V] bool mask, using RNG only to permute fixed pair assignments.

    The construction does not receive or inspect labels.
    """
    assigned_pairs = np.repeat(np.asarray(PAIRS, dtype=np.int8), PAIR_REPETITIONS, axis=0)
    if assigned_pairs.shape != (N, 2):
        raise AssertionError("Fixed pair schedule has the wrong number of rows")
    row_order = np.random.default_rng(seed).permutation(N)
    mask = np.zeros((N, V), dtype=bool)
    mask[row_order, assigned_pairs[:, 0]] = True
    mask[row_order, assigned_pairs[:, 1]] = True
    validate_mask(mask)
    return mask


def pair_counts(mask: np.ndarray) -> Dict[str, int]:
    """Count the six possible pairs represented by a validated mask."""
    validate_mask(mask)
    observed = Counter(tuple(np.flatnonzero(row).tolist()) for row in mask)
    return {pair_key(pair): int(observed[pair]) for pair in PAIRS}


def validate_mask(mask: np.ndarray) -> None:
    """Fail closed unless every exact fixed-mask invariant holds."""
    if not isinstance(mask, np.ndarray) or mask.shape != (N, V):
        raise ValueError(f"mask must have shape {(N, V)}, got {getattr(mask, 'shape', None)}")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask must be boolean or integer")
    if not np.array_equal(mask, mask.astype(bool)):
        raise ValueError("mask values must be binary")
    boolean_mask = mask.astype(bool, copy=False)
    if not np.all(boolean_mask.sum(axis=1) == 2):
        raise ValueError("every row must corrupt exactly two views")
    if not np.all(boolean_mask.sum(axis=0) == 105):
        raise ValueError("every view must have exactly 105 corrupted samples")
    if int(boolean_mask.sum()) != 420:
        raise ValueError("total corrupted sample-view pairs must be 420")
    counts = Counter(tuple(np.flatnonzero(row).tolist()) for row in boolean_mask)
    expected = {pair: PAIR_REPETITIONS for pair in PAIRS}
    if dict(counts) != expected:
        raise ValueError(f"pair counts must be {expected}, got {dict(counts)}")
