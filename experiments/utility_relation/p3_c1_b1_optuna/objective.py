"""Pure NumPy implementation of the preregistered UCSV objective."""

from typing import Iterable, Mapping

import numpy as np

from .protocol import Candidate, DEV_SEEDS, FOLDS, TIE_ROUND_DECIMALS


def validation_j(U_carrier_val, Pi_val, val_targets) -> float:
    """Compute J_UCSV from frozen carrier utility and relation probabilities.

    ``U_carrier_val`` has shape ``[K_val, S]``; ``Pi_val`` has shape
    ``[K_val, S, K]``; and ``val_targets`` has shape ``[K_val]``.  Only
    positive-utility actions participate in the logarithm, avoiding 0*log(0).
    No epsilon, temperature, or additional softmax is applied.
    """
    utility = np.asarray(U_carrier_val, dtype=float)
    probabilities = np.asarray(Pi_val, dtype=float)
    targets = np.asarray(val_targets)
    if utility.ndim != 2 or probabilities.ndim != 3 or targets.ndim != 1:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    k_val, actions = utility.shape
    if probabilities.shape[0] != k_val or probabilities.shape[1] != actions:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    if len(targets) != k_val or probabilities.shape[2] <= 0:
        raise ValueError("INVALID_OBJECTIVE_SHAPE")
    if not (np.isfinite(utility).all() and np.isfinite(probabilities).all()):
        raise ValueError("NONFINITE_INPUT")
    if np.any(utility < 0):
        raise ValueError("NEGATIVE_UTILITY")
    classes = probabilities.shape[2]
    if not np.issubdtype(targets.dtype, np.integer) or np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("INVALID_VALIDATION_TARGET")

    scores = []
    for index, target in enumerate(targets):
        weights = utility[index]
        mass = weights.sum()
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("ZERO_UTILITY_MASS")
        active = weights > 0
        true_probabilities = probabilities[index, :, int(target)][active]
        if np.any(true_probabilities <= 0):
            raise ValueError("NONPOSITIVE_TRUE_RELATION_PROBABILITY")
        scores.append(float(np.sum(weights[active] * np.log(classes * true_probabilities)) / mass))
    result = float(np.mean(scores))
    if not np.isfinite(result):
        raise ValueError("NONFINITE_OUTPUT")
    return result


def paired_delta_j(base_j: float, ours_j: float) -> float:
    """Return the paired semantic validation gain for one seed/fold."""
    if not (np.isfinite(base_j) and np.isfinite(ours_j)):
        raise ValueError("NONFINITE_PAIRED_SCORE")
    return float(ours_j - base_j)


def aggregate_candidate_scores(delta_scores: Mapping) -> dict:
    """Aggregate exactly the two dev seeds times two folds for one candidate."""
    expected = {(seed, fold) for seed in DEV_SEEDS for fold in FOLDS}
    if set(delta_scores) != expected:
        raise ValueError("DELTA_SCORES_MUST_COVER_EXACTLY_FOUR_DEV_FOLDS")
    raw = tuple(float(delta_scores[(seed, fold)]) for seed in DEV_SEEDS for fold in FOLDS)
    if not np.isfinite(raw).all():
        raise ValueError("NONFINITE_DELTA_SCORE")
    return {
        "mean_delta_J": float(np.mean(raw)),
        "worst_delta_J": float(np.min(raw)),
        "positive_count": int(sum(value > 0 for value in raw)),
        "raw_fold_seed_scores": {
            f"{seed}:{fold}": float(delta_scores[(seed, fold)])
            for seed in DEV_SEEDS for fold in FOLDS
        },
    }


def _candidate_fields(candidate) -> tuple[float, int]:
    if isinstance(candidate, Candidate):
        return candidate.semantic_lr, candidate.semantic_exposure
    if isinstance(candidate, Mapping):
        return float(candidate["semantic_lr"]), int(candidate["semantic_exposure"])
    return float(candidate.semantic_lr), int(candidate.semantic_exposure)


def select_best_candidate(candidate_scores: Mapping) -> object:
    """Select by preregistered rounded lexicographic tie-break, without GT."""
    if not candidate_scores:
        raise ValueError("NO_CANDIDATE_SCORES")

    def ranking(item):
        candidate, score = item
        lr, exposure = _candidate_fields(candidate)
        return (
            round(float(score["mean_delta_J"]), TIE_ROUND_DECIMALS),
            round(float(score["worst_delta_J"]), TIE_ROUND_DECIMALS),
            int(score["positive_count"]),
            -exposure,
            -lr,
        )

    return max(candidate_scores.items(), key=ranking)[0]
