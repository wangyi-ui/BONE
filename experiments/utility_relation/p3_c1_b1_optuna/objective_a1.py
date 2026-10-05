"""Versioned B1-A1 utility-conditioned abstention objective."""

from __future__ import annotations

import numpy as np

from .objective import aggregate_candidate_scores, paired_delta_j, select_best_candidate


def validation_j_with_audit(U_carrier_val, Pi_val, val_targets) -> tuple[float, dict]:
    """Compute J_UCSV, assigning exactly zero contribution when utility abstains.

    Every validation anchor remains in the fold denominator.  Positive-utility
    actions retain the exact frozen B1 equation and validation behavior.
    """
    utility = np.asarray(U_carrier_val, dtype=float)
    probabilities = np.asarray(Pi_val, dtype=float)
    targets = np.asarray(val_targets)
    if utility.ndim != 2 or probabilities.ndim != 3 or targets.ndim != 1:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    k_val, actions = utility.shape
    if probabilities.shape[0] != k_val or probabilities.shape[1] != actions:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    if len(targets) != k_val or k_val == 0 or probabilities.shape[2] <= 0:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    if not (np.isfinite(utility).all() and np.isfinite(probabilities).all()):
        raise ValueError("NONFINITE_INPUT")
    if np.any(utility < 0):
        raise ValueError("NEGATIVE_UTILITY")
    classes = probabilities.shape[2]
    if not np.issubdtype(targets.dtype, np.integer) or np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("INVALID_VALIDATION_TARGET")

    anchor_scores = []
    masses = utility.sum(axis=1)
    if not np.isfinite(masses).all():
        raise ValueError("NONFINITE_INPUT")
    for index, target in enumerate(targets):
        weights = utility[index]
        mass = masses[index]
        if mass == 0:
            anchor_scores.append(0.0)
            continue
        active = weights > 0
        true_probabilities = probabilities[index, :, int(target)][active]
        if np.any(true_probabilities <= 0):
            raise ValueError("NONPOSITIVE_TRUE_RELATION_PROBABILITY")
        anchor_scores.append(float(np.sum(weights[active] * np.log(classes * true_probabilities)) / mass))

    result = float(np.sum(anchor_scores) / k_val)
    if not np.isfinite(result):
        raise ValueError("NONFINITE_OUTPUT")
    supported = masses > 0
    audit = {
        "validation_anchor_count": int(k_val),
        "supported_anchor_count": int(np.count_nonzero(supported)),
        "abstained_anchor_count": int(np.count_nonzero(~supported)),
        "coverage": float(np.count_nonzero(supported) / k_val),
        "anchor_utility_masses": [float(value) for value in masses],
        "anchor_scores": anchor_scores,
        "fold_denominator": int(k_val),
    }
    return result, audit


def validation_j(U_carrier_val, Pi_val, val_targets) -> float:
    """Return the A1 fold score while retaining all anchors in the denominator."""
    return validation_j_with_audit(U_carrier_val, Pi_val, val_targets)[0]
