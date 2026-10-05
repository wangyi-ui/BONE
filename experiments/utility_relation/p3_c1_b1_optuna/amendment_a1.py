"""Preregistered utility-abstention amendment to the frozen B1 protocol.

This module changes only the interpretation of an exactly zero utility mass at
validation time.  It imports all scientific search metadata from the immutable
base protocol rather than restating it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import protocol


AMENDMENT_ID = "P3-C1-B1-A1"
AMENDMENT_NAME = "Utility-Abstention Validation Amendment"
APPLIES_TO_ALL_DATASETS = True
AMENDED_DATASETS = protocol.DATASETS
ZERO_UTILITY_VALIDATION_POLICY = "ABSTAIN_ZERO_CONTRIBUTION"
CLASS_FEASIBILITY_POLICY = "AT_LEAST_ONE_SUPPORTED_ANCHOR_PER_CLASS_PER_DEV_SEED"


def validate_class_feasibility(
    labeled_targets: Iterable[int],
    utility_masses: Iterable[float],
    expected_class_ids: Iterable[int] | None = None,
) -> dict:
    """Require at least one of each class's two frozen anchors to be supported.

    The returned information is audit-only.  It is not an objective, a
    threshold, a parameter filter, or a tie-break input.
    """
    targets = np.asarray(labeled_targets)
    masses = np.asarray(utility_masses, dtype=float)
    if targets.ndim != 1 or masses.ndim != 1 or len(targets) != len(masses):
        raise ValueError("INVALID_CLASS_FEASIBILITY_SHAPE")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("INVALID_CLASS_TARGET")
    if not np.isfinite(masses).all():
        raise ValueError("NONFINITE_UTILITY_MASS")
    if np.any(masses < 0):
        raise ValueError("NEGATIVE_UTILITY_MASS")

    observed = set(targets.tolist())
    if expected_class_ids is not None and observed != set(expected_class_ids):
        raise ValueError("MISSING_OR_UNEXPECTED_CLASS")
    if not observed:
        raise ValueError("NO_SPARSE_CLASSES")

    per_class = {}
    for class_id in sorted(observed):
        class_masses = masses[targets == class_id]
        if len(class_masses) != 2:
            raise ValueError("CLASS_MUST_HAVE_EXACTLY_TWO_SPARSE_ANCHORS")
        supported = class_masses > 0
        per_class[str(int(class_id))] = {
            "anchor_utility_masses": [float(value) for value in class_masses],
            "supported_anchor_count": int(np.count_nonzero(supported)),
            "feasible": bool(np.any(supported)),
        }
        if not np.any(supported):
            raise ValueError("CLASS_ZERO_UTILITY_EVIDENCE")

    return {
        "policy": CLASS_FEASIBILITY_POLICY,
        "class_count": len(per_class),
        "all_classes_feasible": True,
        "per_class": per_class,
    }
