"""Deterministic two-fold CV over frozen sparse labels only."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SparseFold:
    fold_name: str
    train_raw_row_ids: tuple
    train_targets: tuple
    val_raw_row_ids: tuple
    val_targets: tuple


def _as_one_dimensional(name: str, values: Iterable) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name.upper()}_MUST_BE_1D")
    return array


def build_sparse_label_folds(
    labeled_raw_row_ids: Iterable,
    labeled_targets: Iterable,
    expected_class_ids: Iterable | None = None,
) -> tuple[SparseFold, SparseFold]:
    """Construct A/B folds without reading any unlabeled label or full GT.

    Every observed class must have exactly two sparse labels.  Within a class,
    the smaller raw file row id trains fold A and validates fold B.
    """
    row_ids = _as_one_dimensional("labeled_raw_row_ids", labeled_raw_row_ids)
    targets = _as_one_dimensional("labeled_targets", labeled_targets)
    if len(row_ids) != len(targets):
        raise ValueError("ROW_CLASS_SHAPE_MISMATCH")
    if len(set(row_ids.tolist())) != len(row_ids):
        raise ValueError("DUPLICATE_RAW_ROW_ID")

    observed = set(targets.tolist())
    if expected_class_ids is not None and observed != set(expected_class_ids):
        raise ValueError("MISSING_OR_UNEXPECTED_CLASS")
    if not observed:
        raise ValueError("NO_SPARSE_CLASSES")

    groups = {}
    for row_id, target in zip(row_ids.tolist(), targets.tolist()):
        groups.setdefault(target, []).append(row_id)
    for rows in groups.values():
        if len(rows) != 2:
            raise ValueError("CLASS_MUST_HAVE_EXACTLY_TWO_SPARSE_LABELS")

    # repr gives a deterministic total ordering even for nonnumeric class IDs.
    ordered_classes = sorted(groups, key=repr)
    smaller = {target: tuple(sorted(groups[target])) for target in ordered_classes}
    a_train_ids = tuple(smaller[target][0] for target in ordered_classes)
    a_val_ids = tuple(smaller[target][1] for target in ordered_classes)
    class_order = tuple(ordered_classes)
    fold_a = SparseFold("A", a_train_ids, class_order, a_val_ids, class_order)
    fold_b = SparseFold("B", a_val_ids, class_order, a_train_ids, class_order)
    return fold_a, fold_b


def split_audit_hash(folds: Iterable[SparseFold]) -> str:
    """Return a canonical SHA256 provenance identifier for generated splits."""
    payload = [
        {
            "fold_name": fold.fold_name,
            "train_raw_row_ids": list(fold.train_raw_row_ids),
            "train_targets": list(fold.train_targets),
            "val_raw_row_ids": list(fold.val_raw_row_ids),
            "val_targets": list(fold.val_targets),
        }
        for fold in folds
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
