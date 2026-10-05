"""Deterministic balanced half-view corruption masks for P3-A4.

Mask construction depends only on dataset name, N, V, raw row IDs, the fixed
corruption seed, and this namespace.  It does not accept labels, features,
training seeds, DataLoader positions, or global RNG state.  Masks are indexed
by RAW_FILE_ROW_ID; a future experiment-layer adapter must transport that ID
through native preprocessing permutations before applying a mask to tensors.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
from typing import Iterable

import numpy as np


NAMESPACE = "BONE_P3_A4_V1"
CORRUPTION_SEED = 20
TARGET_SNR_DB = 2.5
HISTORICAL_NOISE_FORMULA_SOURCE = (
    "experiments.weak_quality.materialize_bone_p1_msrc_corruption."
    "corrupt_standardized_views: per-view RandomState(seed).normal, "
    "alpha=sqrt(sum(clean[mask]^2)/(sum(noise[mask]^2)*10**(snr_db/10))), "
    "corrupt[mask]=clean[mask]+alpha*noise[mask]"
)


def sha256_file(path: str) -> str:
    """Hash a file without introducing protocol state."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(*parts: object) -> bytes:
    return (NAMESPACE + "|" + "|".join(str(part) for part in parts)).encode("utf-8")


def _digest(*parts: object) -> str:
    return hashlib.sha256(_payload(*parts)).hexdigest()


def corruption_cardinality_groups(n: int, v: int) -> dict[int, int]:
    """Return the required row count for each permitted corruption cardinality."""
    if isinstance(n, bool) or isinstance(v, bool) or not isinstance(n, int) or not isinstance(v, int) or n < 1 or v < 2:
        raise ValueError("n must be positive and v must be an integer at least two")
    low = v // 2
    if v % 2 == 0:
        return {low: n}
    total = (n * v) // 2
    high_count = total - n * low
    return {low: n - high_count, low + 1: high_count}


def pattern_tuples(v: int, cardinality: int) -> tuple[tuple[int, ...], ...]:
    """Return lexicographically enumerated corruption subsets."""
    if cardinality < 1 or cardinality >= v:
        raise ValueError("corruption cardinality must be within 1..V-1")
    return tuple(itertools.combinations(range(v), cardinality))


def _rank_rows(dataset_name: str, cardinality: int, raw_row_ids: Iterable[int]) -> tuple[int, ...]:
    rows = tuple(int(row_id) for row_id in raw_row_ids)
    if len(set(rows)) != len(rows) or any(row_id < 0 for row_id in rows):
        raise ValueError("raw row IDs must be unique nonnegative integers")
    return tuple(sorted(rows, key=lambda row_id: (_digest("row", dataset_name, CORRUPTION_SEED, cardinality, row_id), row_id)))


def _extra_patterns(dataset_name: str, v: int, cardinality: int, count: int) -> tuple[int, ...]:
    """Choose the ``count`` extra patterns with deterministic low-exposure greedy balance."""
    patterns = pattern_tuples(v, cardinality)
    if not 0 <= count < len(patterns):
        raise ValueError("extra pattern count is outside the valid range")
    selected: list[int] = []
    exposure = np.zeros(v, dtype=np.int64)
    remaining = set(range(len(patterns)))
    for step in range(count):
        candidates = []
        for index in remaining:
            next_exposure = exposure.copy()
            next_exposure[list(patterns[index])] += 1
            # First minimize exposure imbalance; then its sorted profile; ties
            # are resolved solely by a documented SHA key and lexicographic tuple.
            score = (
                int(next_exposure.max() - next_exposure.min()),
                tuple(sorted((int(value) for value in next_exposure), reverse=True)),
                _digest("extra-pattern", dataset_name, CORRUPTION_SEED, cardinality, step, ",".join(map(str, patterns[index]))),
                patterns[index],
            )
            candidates.append((score, index, next_exposure))
        _score, chosen, exposure = min(candidates, key=lambda item: item[0])
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def _pattern_slots(dataset_name: str, v: int, cardinality: int, row_count: int) -> tuple[tuple[int, ...], ...]:
    """Allocate patterns evenly, then deterministically rank repeated pattern slots."""
    patterns = pattern_tuples(v, cardinality)
    quotient, remainder = divmod(row_count, len(patterns))
    counts = np.full(len(patterns), quotient, dtype=np.int64)
    if remainder:
        counts[list(_extra_patterns(dataset_name, v, cardinality, remainder))] += 1
    if int(counts.max() - counts.min()) > 1:
        raise RuntimeError("pattern allocation is not balanced")
    slots = []
    for index, pattern in enumerate(patterns):
        for slot in range(int(counts[index])):
            slots.append((
                _digest("pattern-slot", dataset_name, CORRUPTION_SEED, cardinality, ",".join(map(str, pattern)), slot),
                pattern,
                slot,
            ))
    slots.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(pattern for _digest_value, pattern, _slot in slots)


def build_balanced_half_view_mask(dataset_name: str, n: int, v: int, corruption_seed: int = CORRUPTION_SEED) -> np.ndarray:
    """Construct a deterministic raw-row indexed boolean mask.

    ``corruption_seed`` is intentionally fixed to 20.  The explicit argument
    provides fail-closed validation and is not a training-seed input.
    """
    if corruption_seed != CORRUPTION_SEED:
        raise ValueError(f"corruption_seed must remain frozen at {CORRUPTION_SEED}")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a nonempty string")
    groups = corruption_cardinality_groups(n, v)
    raw_rows = set(range(n))
    mask = np.zeros((n, v), dtype=bool)
    for cardinality, count in groups.items():
        ranked = _rank_rows(dataset_name, cardinality, raw_rows)
        group_rows = ranked[:count]
        raw_rows.difference_update(group_rows)
        slots = _pattern_slots(dataset_name, v, cardinality, count)
        if len(group_rows) != len(slots):
            raise RuntimeError("cardinality group row/pattern assignment length mismatch")
        for row_id, pattern in zip(group_rows, slots):
            mask[row_id, list(pattern)] = True
    if raw_rows:
        raise RuntimeError("not every raw row received a cardinality assignment")
    validate_balanced_half_view_mask(mask, dataset_name, corruption_seed)
    return mask


def pattern_counts(mask: np.ndarray) -> dict[str, int]:
    """Count observed patterns in stable comma-separated view-index notation."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional ndarray")
    observed = Counter(tuple(np.flatnonzero(row).tolist()) for row in mask.astype(bool, copy=False))
    return {",".join(map(str, pattern)): int(count) for pattern, count in sorted(observed.items())}


def _group_pattern_counts(mask: np.ndarray, cardinality: int) -> list[int]:
    v = mask.shape[1]
    patterns = pattern_tuples(v, cardinality)
    observed = Counter(tuple(np.flatnonzero(row).tolist()) for row in mask.astype(bool, copy=False) if int(row.sum()) == cardinality)
    return [int(observed[pattern]) for pattern in patterns]


def validate_balanced_half_view_mask(mask: np.ndarray, dataset_name: str, corruption_seed: int = CORRUPTION_SEED) -> None:
    """Fail closed on row-cardinality, global, pattern, or exposure imbalance."""
    if corruption_seed != CORRUPTION_SEED:
        raise ValueError("corruption seed mismatch")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional bool ndarray")
    n, v = mask.shape
    groups = corruption_cardinality_groups(n, v)
    row_sums = mask.sum(axis=1)
    if sum(int(np.count_nonzero(row_sums == cardinality)) for cardinality in groups) != n:
        raise ValueError("rows contain forbidden corruption cardinalities")
    for cardinality, expected_rows in groups.items():
        if int(np.count_nonzero(row_sums == cardinality)) != expected_rows:
            raise ValueError("cardinality group count mismatch")
        usages = _group_pattern_counts(mask, cardinality)
        if usages and max(usages) - min(usages) > 1:
            raise ValueError("same-cardinality pattern usage differs by more than one")
    if int(mask.sum()) != (n * v) // 2:
        raise ValueError("global corruption cell count is incorrect")
    per_view = mask.sum(axis=0)
    # Exact equality is required whenever the target count per view is integral;
    # otherwise a one-cell spread is the strongest possible balance.
    if (n * v // 2) % v == 0:
        if not np.all(per_view == per_view[0]):
            raise ValueError("exact per-view balance was achievable but not reached")
    elif int(per_view.max() - per_view.min()) > 1:
        raise ValueError("per-view exposure differs by more than one")
    if not dataset_name:
        raise ValueError("dataset name is required for protocol validation")


def mask_sha256(mask: np.ndarray) -> str:
    """Hash canonical boolean mask bytes in C order."""
    if mask.dtype != np.bool_ or not mask.flags.c_contiguous:
        mask = np.ascontiguousarray(mask, dtype=bool)
    return hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def apply_historical_msrc_noise(clean_views: list[np.ndarray], mask: np.ndarray):
    """Reuse the frozen P1 noise helper directly for any future MSRC feature use."""
    from experiments.weak_quality.materialize_bone_p1_msrc_corruption import corrupt_standardized_views

    return corrupt_standardized_views(clean_views, mask, corruption_seed=CORRUPTION_SEED)
