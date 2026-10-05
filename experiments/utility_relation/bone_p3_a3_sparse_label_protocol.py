"""Frozen P3-A3 deterministic raw-row sparse-label protocol.

The selector is intentionally independent of features, models, optimizers,
training seeds, corruption seeds, DataLoader order, and all RNGs.  Full labels
are permitted only to materialize an offline frozen supervision artifact; a
future training runner receives only ``labeled_ids`` and ``labeled_targets``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


NAMESPACE = "BONE_P3_A3_V1"
LABELS_PER_CLASS = 2
LEGACY_MSRC_IDS = np.asarray((17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201), dtype=np.int64)
LEGACY_MSRC_TARGETS = np.asarray((0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6), dtype=np.int64)
LEGACY_MSRC_PROVENANCE_STATUS = "PARTIAL_PROVENANCE"


@dataclass(frozen=True)
class SparseLabelSplit:
    """The only supervision values a future training runner may receive."""

    dataset_name: str
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    raw_classes: np.ndarray
    split_mode: str
    provenance_status: str | None

    @property
    def labels_per_class(self) -> int:
        return LABELS_PER_CLASS

    @property
    def k(self) -> int:
        return int(self.raw_classes.size)

    @property
    def labeled_count(self) -> int:
        return int(self.labeled_ids.size)


def canonicalize_raw_labels(raw_labels: np.ndarray | Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Map sorted unique raw classes deterministically to ``0,...,K-1``."""
    labels = np.asarray(raw_labels).reshape(-1)
    if labels.size == 0:
        raise ValueError("raw labels must be nonempty")
    raw_classes, canonical_targets = np.unique(labels, return_inverse=True)
    return raw_classes, canonical_targets.astype(np.int64, copy=False)


def raw_to_canonical_label_mapping(raw_classes: np.ndarray) -> dict[str, int]:
    """Return the JSON-safe sorted-raw-label mapping required by the protocol."""
    return {str(int(raw_class)): int(index) for index, raw_class in enumerate(np.asarray(raw_classes).reshape(-1))}


def selector_payload(dataset_name: str, raw_row_id: int) -> bytes:
    """Return the exact UTF-8 selector payload mandated by P3-A3."""
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be a nonempty string")
    if isinstance(raw_row_id, bool) or int(raw_row_id) != raw_row_id or raw_row_id < 0:
        raise ValueError("raw_row_id must be a nonnegative integer")
    return f"{NAMESPACE}|{dataset_name}|{int(raw_row_id)}".encode("utf-8")


def selector_digest(dataset_name: str, raw_row_id: int) -> str:
    """Return the lowercase hexadecimal SHA256 rank key for one raw row ID."""
    return hashlib.sha256(selector_payload(dataset_name, raw_row_id)).hexdigest()


def rank_candidate_raw_rows(dataset_name: str, raw_row_ids: Iterable[int]) -> tuple[int, ...]:
    """Sort candidates only by (digest, raw_row_id), independent of input order."""
    rows = tuple(int(row_id) for row_id in raw_row_ids)
    if len(set(rows)) != len(rows) or any(row_id < 0 for row_id in rows):
        raise ValueError("candidate raw row IDs must be unique nonnegative integers")
    return tuple(sorted(rows, key=lambda row_id: (selector_digest(dataset_name, row_id), row_id)))


def _validate_split(split: SparseLabelSplit, canonical_targets: np.ndarray) -> None:
    expected_count = LABELS_PER_CLASS * split.k
    if split.labeled_ids.shape != (expected_count,) or split.labeled_targets.shape != (expected_count,):
        raise ValueError("split has the wrong labeled array shape")
    if np.unique(split.labeled_ids).size != split.labeled_ids.size:
        raise ValueError("split contains duplicate raw row IDs")
    if np.any(split.labeled_ids < 0) or np.any(split.labeled_ids >= canonical_targets.size):
        raise ValueError("split contains an invalid raw row ID")
    if not np.array_equal(canonical_targets[split.labeled_ids], split.labeled_targets):
        raise ValueError("selected raw labels do not equal saved canonical targets")
    if not np.array_equal(np.bincount(split.labeled_targets, minlength=split.k), np.full(split.k, LABELS_PER_CLASS)):
        raise ValueError("every canonical class must contribute exactly two labels")
    order = np.lexsort((split.labeled_ids, split.labeled_targets))
    if not np.array_equal(order, np.arange(split.labeled_ids.size)):
        raise ValueError("split output must be sorted by canonical target then raw row ID")


def build_sparse_label_split(dataset_name: str, raw_labels: np.ndarray | Iterable[int]) -> SparseLabelSplit:
    """Build the MSRC legacy split or the universal deterministic hash split."""
    raw_classes, canonical_targets = canonicalize_raw_labels(raw_labels)
    k = int(raw_classes.size)
    if dataset_name == "MSRC":
        if k != 7 or canonical_targets.size != 210:
            raise ValueError("MSRC legacy split requires N=210 and K=7")
        split = SparseLabelSplit(
            dataset_name=dataset_name,
            labeled_ids=LEGACY_MSRC_IDS.copy(),
            labeled_targets=LEGACY_MSRC_TARGETS.copy(),
            raw_classes=raw_classes.copy(),
            split_mode="LEGACY_FROZEN",
            provenance_status=LEGACY_MSRC_PROVENANCE_STATUS,
        )
        _validate_split(split, canonical_targets)
        return split

    selected: list[tuple[int, int]] = []
    for canonical_class in range(k):
        candidates = np.flatnonzero(canonical_targets == canonical_class).astype(np.int64, copy=False)
        if candidates.size < LABELS_PER_CLASS:
            raise ValueError(f"canonical class {canonical_class} has fewer than {LABELS_PER_CLASS} rows")
        selected.extend((canonical_class, row_id) for row_id in rank_candidate_raw_rows(dataset_name, candidates)[:LABELS_PER_CLASS])
    selected.sort(key=lambda item: (item[0], item[1]))
    split = SparseLabelSplit(
        dataset_name=dataset_name,
        labeled_ids=np.asarray([row_id for _target, row_id in selected], dtype=np.int64),
        labeled_targets=np.asarray([target for target, _row_id in selected], dtype=np.int64),
        raw_classes=raw_classes.copy(),
        split_mode="P3_A3_HASH_STRATIFIED_V1",
        provenance_status=None,
    )
    _validate_split(split, canonical_targets)
    return split


def canonical_split_payload(split: SparseLabelSplit, n: int, dataset_file_sha256: str) -> dict[str, object]:
    """Return canonical, JSON-serializable split evidence without full labels."""
    if n <= 0 or split.labeled_count != LABELS_PER_CLASS * split.k:
        raise ValueError("invalid split payload dimensions")
    return {
        "dataset_name": split.dataset_name,
        "N": int(n),
        "K": split.k,
        "labels_per_class": LABELS_PER_CLASS,
        "labeled_count": split.labeled_count,
        "label_fraction": split.labeled_count / n,
        "row_id_semantics": "RAW_FILE_ROW_ID",
        "split_mode": split.split_mode,
        "provenance_status": split.provenance_status,
        "namespace": "NOT_APPLICABLE_LEGACY_FROZEN" if split.split_mode == "LEGACY_FROZEN" else NAMESPACE,
        "raw_to_canonical_label_mapping": raw_to_canonical_label_mapping(split.raw_classes),
        "labeled_ids": split.labeled_ids.tolist(),
        "labeled_targets": split.labeled_targets.tolist(),
        "dataset_file_sha256": dataset_file_sha256,
        "full_gt_leakage_boundary": "Full GT is read only for offline split materialization. Frozen training receives labeled_ids/labeled_targets only; full GT is post-seal evaluator only.",
    }


def split_content_sha256(split: SparseLabelSplit) -> str:
    """Hash only canonical split content, independent of NPZ container metadata."""
    payload = {
        "dataset_name": split.dataset_name,
        "labeled_ids": split.labeled_ids.tolist(),
        "labeled_targets": split.labeled_targets.tolist(),
        "split_mode": split.split_mode,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
