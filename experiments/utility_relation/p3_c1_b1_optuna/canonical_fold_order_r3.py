"""Numeric class-order canonicalization by within-fold permutation only."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Iterable, Mapping

from .sparse_cv import SparseFold


CLASS_COVERAGE_ERROR = "NONCANONICAL_SPARSE_FOLD_CLASS_COVERAGE"


def _numeric_targets(values: Iterable, cluster_count: int) -> tuple[int, ...]:
    numeric = []
    try:
        for value in values:
            converted = int(value)
            if isinstance(value, bool) or value != converted:
                raise ValueError(CLASS_COVERAGE_ERROR)
            numeric.append(converted)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(CLASS_COVERAGE_ERROR) from error
    if tuple(sorted(numeric)) != tuple(range(cluster_count)):
        raise ValueError(CLASS_COVERAGE_ERROR)
    return tuple(numeric)


def _pair_payload(fold_name: str, train_pairs: Iterable, validation_pairs: Iterable) -> dict:
    def normalized(pairs):
        return sorted(
            ([int(target), int(raw_row_id)] for target, raw_row_id in pairs),
            key=lambda pair: (pair[0], pair[1]),
        )

    return {
        "fold": fold_name,
        "train": normalized(train_pairs),
        "validation": normalized(validation_pairs),
    }


def _pair_set_hash(fold_name: str, train_pairs: Iterable, validation_pairs: Iterable) -> str:
    payload = _pair_payload(fold_name, train_pairs, validation_pairs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_permutation(before: SparseFold, after: SparseFold) -> None:
    before_train = list(zip(before.train_targets, before.train_raw_row_ids))
    after_train = list(zip(after.train_targets, after.train_raw_row_ids))
    before_validation = list(zip(before.val_targets, before.val_raw_row_ids))
    after_validation = list(zip(after.val_targets, after.val_raw_row_ids))
    if before.fold_name != after.fold_name:
        raise RuntimeError("FOLD_MEMBERSHIP_CHANGED")
    if Counter(before_train) != Counter(after_train):
        raise RuntimeError("TRAIN_PAIR_PRESERVATION_FAILURE")
    if Counter(before_validation) != Counter(after_validation):
        raise RuntimeError("VALIDATION_PAIR_PRESERVATION_FAILURE")
    for before_values, after_values in (
        (before.train_raw_row_ids, after.train_raw_row_ids),
        (before.val_raw_row_ids, after.val_raw_row_ids),
        (before.train_targets, after.train_targets),
        (before.val_targets, after.val_targets),
    ):
        if Counter(before_values) != Counter(after_values):
            raise RuntimeError("FOLD_MULTISET_PRESERVATION_FAILURE")


def canonicalize_sparse_fold(
    fold: SparseFold,
    cluster_count: int,
) -> tuple[SparseFold, dict]:
    """Return a numerically ordered fold and a proof of pure permutation."""
    if isinstance(cluster_count, bool) or not isinstance(cluster_count, int) or cluster_count <= 0:
        raise ValueError(CLASS_COVERAGE_ERROR)
    if len(fold.train_targets) != len(fold.train_raw_row_ids):
        raise ValueError("TRAIN_PAIR_SHAPE_MISMATCH")
    if len(fold.val_targets) != len(fold.val_raw_row_ids):
        raise ValueError("VALIDATION_PAIR_SHAPE_MISMATCH")
    _numeric_targets(fold.train_targets, cluster_count)
    _numeric_targets(fold.val_targets, cluster_count)

    train_pairs = list(zip(fold.train_targets, fold.train_raw_row_ids))
    validation_pairs = list(zip(fold.val_targets, fold.val_raw_row_ids))
    sorted_train = sorted(train_pairs, key=lambda pair: int(pair[0]))
    sorted_validation = sorted(validation_pairs, key=lambda pair: int(pair[0]))
    candidate = SparseFold(
        fold_name=fold.fold_name,
        train_raw_row_ids=tuple(raw_row_id for target, raw_row_id in sorted_train),
        train_targets=tuple(target for target, raw_row_id in sorted_train),
        val_raw_row_ids=tuple(raw_row_id for target, raw_row_id in sorted_validation),
        val_targets=tuple(target for target, raw_row_id in sorted_validation),
    )
    _assert_permutation(fold, candidate)
    before_hash = _pair_set_hash(
        fold.fold_name, train_pairs, validation_pairs
    )
    after_hash = _pair_set_hash(
        candidate.fold_name,
        zip(candidate.train_targets, candidate.train_raw_row_ids),
        zip(candidate.val_targets, candidate.val_raw_row_ids),
    )
    if before_hash != after_hash:
        raise RuntimeError("FOLD_PAIR_SET_HASH_MISMATCH")
    expected = tuple(range(cluster_count))
    if candidate.train_targets != expected or candidate.val_targets != expected:
        raise RuntimeError("NUMERIC_CANONICALIZATION_FAILURE")

    # Preserve exact identity when no permutation is necessary.
    canonical = fold if candidate == fold else candidate
    audit = {
        "fold": fold.fold_name,
        "K": cluster_count,
        "ordering_changed": canonical is not fold,
        "before_order": {
            "train": [int(value) for value in fold.train_targets],
            "validation": [int(value) for value in fold.val_targets],
        },
        "after_order": {
            "train": [int(value) for value in canonical.train_targets],
            "validation": [int(value) for value in canonical.val_targets],
        },
        "pair_set_hash_before": before_hash,
        "pair_set_hash_after": after_hash,
        "pair_preservation": True,
        "label_values_changed": False,
        "anchor_membership_changed": False,
        "class_coverage_pass": True,
    }
    return canonical, audit


def canonicalize_sparse_folds(
    folds: Iterable[SparseFold],
    cluster_count: int,
) -> tuple[tuple[SparseFold, ...], Mapping[str, dict]]:
    canonical = []
    audits = {}
    for fold in folds:
        if fold.fold_name in audits:
            raise ValueError("DUPLICATE_FOLD_NAME")
        ordered, audit = canonicalize_sparse_fold(fold, cluster_count)
        canonical.append(ordered)
        audits[fold.fold_name] = audit
    return tuple(canonical), audits

