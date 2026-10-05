"""P3-C1-B1-A2 coverage-aware utility-abstention feasibility protocol."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from . import protocol


AMENDMENT_ID = "P3-C1-B1-A2"
AMENDMENT_NAME = "Coverage-Aware Utility Abstention"
APPLIES_TO_ALL_DATASETS = True
AMENDED_DATASETS = protocol.DATASETS
FEASIBILITY_POLICY = "AT_LEAST_ONE_SUPPORTED_SPARSE_ANCHOR_PER_DEV_SEED"
FAIL_CLOSED_ERROR = "NO_UTILITY_SUPPORTED_VALIDATION_EVIDENCE"
UTILITY_THRESHOLD_ADDED = False
COVERAGE_THRESHOLD_ADDED = False


def validate_coverage_aware_feasibility(
    labeled_targets: Iterable[int],
    labeled_raw_ids: Iterable[int],
    utility_masses: Iterable[float],
    expected_class_ids: Iterable[int],
) -> dict:
    """Validate A2 feasibility and return coverage information for audit only."""
    targets = np.asarray(labeled_targets)
    raw_ids = np.asarray(labeled_raw_ids)
    masses = np.asarray(utility_masses, dtype=float)
    if (
        targets.ndim != 1
        or raw_ids.ndim != 1
        or masses.ndim != 1
        or len(targets) != len(raw_ids)
        or len(targets) != len(masses)
    ):
        raise ValueError("INVALID_COVERAGE_FEASIBILITY_SHAPE")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("INVALID_CLASS_TARGET")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("INVALID_SPARSE_ANCHOR_ID")
    if not np.isfinite(masses).all():
        raise ValueError("NONFINITE_UTILITY_MASS")
    if np.any(masses < 0):
        raise ValueError("NEGATIVE_UTILITY_MASS")
    if len(set(int(value) for value in raw_ids.tolist())) != len(raw_ids):
        raise ValueError("DUPLICATE_SPARSE_ANCHOR_ID")

    observed = set(int(value) for value in targets.tolist())
    expected = set(int(value) for value in expected_class_ids)
    if observed != expected:
        raise ValueError("MISSING_OR_UNEXPECTED_CLASS")
    if not observed:
        raise ValueError("NO_SPARSE_CLASSES")

    per_class = []
    for class_id in sorted(observed):
        mask = targets == class_id
        class_masses = masses[mask]
        class_raw_ids = raw_ids[mask]
        if len(class_masses) != 2:
            raise ValueError("CLASS_MUST_HAVE_EXACTLY_TWO_SPARSE_ANCHORS")
        supported_count = int(np.count_nonzero(class_masses > 0))
        per_class.append({
            "class_id": int(class_id),
            "anchor_raw_ids": [int(value) for value in class_raw_ids],
            "anchor_utility_masses": [float(value) for value in class_masses],
            "supported_anchor_count": supported_count,
            "utility_abstained_class": supported_count == 0,
        })

    supported_anchor_count = int(np.count_nonzero(masses > 0))
    if supported_anchor_count == 0:
        raise ValueError(FAIL_CLOSED_ERROR)
    total_anchor_count = int(len(masses))
    supported_class_count = sum(
        not item["utility_abstained_class"] for item in per_class
    )
    class_count = len(per_class)
    return {
        "amendment_id": AMENDMENT_ID,
        "policy": FEASIBILITY_POLICY,
        "A2_FEASIBILITY": "PASS",
        "total_anchor_count": total_anchor_count,
        "supported_anchor_count": supported_anchor_count,
        "abstained_anchor_count": total_anchor_count - supported_anchor_count,
        "anchor_coverage": float(supported_anchor_count / total_anchor_count),
        "class_count": class_count,
        "supported_class_count": int(supported_class_count),
        "abstained_class_count": int(class_count - supported_class_count),
        "supported_class_coverage": float(supported_class_count / class_count),
        "per_class": per_class,
        "coverage_audit_only": True,
        "coverage_threshold_added": False,
        "full_gt_used": False,
    }


def aggregate_dev_seed_coverage_audits(
    audits: Mapping[int, Mapping],
) -> dict:
    """Summarize abstained classes across seeds without gating any dataset."""
    if not audits:
        raise ValueError("NO_DEV_SEED_COVERAGE_AUDITS")
    abstained_by_seed = {}
    for seed, audit in audits.items():
        if audit.get("A2_FEASIBILITY") != "PASS":
            raise ValueError("INVALID_DEV_SEED_COVERAGE_AUDIT")
        abstained_by_seed[int(seed)] = {
            int(item["class_id"])
            for item in audit["per_class"]
            if item["utility_abstained_class"]
        }
    sets = tuple(abstained_by_seed.values())
    persistent = set.intersection(*sets)
    union = set.union(*sets)
    return {
        "persistent_abstained_classes": sorted(persistent),
        "union_abstained_classes": sorted(union),
        "seed_specific_abstained_classes": {
            str(seed): sorted(classes - persistent)
            for seed, classes in sorted(abstained_by_seed.items())
        },
        "coverage_audit_only": True,
        "coverage_threshold_added": False,
        "full_gt_used": False,
    }

