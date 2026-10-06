"""
P4-B0 Utility-Aligned Shared Early Stopping (UASES).

This module is deliberately arm-agnostic:
the identical score/state-machine implementation is used for BASE and OURS.

No GT.
No sparse labels.
No T_rel.
No semantic loss.

Frozen carrier U_cycle is used only as a detached reliability weight.

Shapes
------
q_views:
    [N, V, K]

frozen_u:
    [N, S]

p_gen / p_ver:
    [N, S, K]

consensus_q:
    [N, K]

hard_prediction:
    [N]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)


CHECKPOINT_INTERVAL = 5
H_MAX = 100
PATIENCE_CHECKPOINTS = 4
IMPROVEMENT_EPSILON = 1e-6

PROB_EPSILON = 1e-12


def _finite_scalar(
    value: float,
    name: str,
) -> float:
    out = float(value)

    if not np.isfinite(out):
        raise ValueError(
            f"{name} must be finite"
        )

    return out


def freeze_utility(
    u_cycle: np.ndarray,
) -> np.ndarray:
    """
    Freeze carrier full-sample U_cycle.

    Input:
        u_cycle [N, S]

    Output:
        frozen_u [N, S], float64, read-only

    This is the NumPy equivalent of stop-gradient:
    the returned array owns an immutable detached copy.
    """

    values = np.asarray(
        u_cycle,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "U_cycle must have shape [N,S]"
        )

    if (
        values.shape[0] < 1
        or values.shape[1] < 1
    ):
        raise ValueError(
            "U_cycle dimensions must be non-empty"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "U_cycle must be finite"
        )

    if np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ValueError(
            "U_cycle must be in [0,1]"
        )

    out = np.array(
        values,
        dtype=np.float64,
        copy=True,
        order="C",
    )

    if float(out.sum()) <= 0.0:
        raise ValueError(
            "sum(U_cycle) must be positive"
        )

    out.setflags(
        write=False
    )

    return out


def validate_q_views(
    q_views: np.ndarray,
) -> np.ndarray:
    """
    Validate per-view BONE posteriors.

    q_views:
        [N, V, K]

    Returns a float64 view/copy with the same [N,V,K] shape.
    """

    q = np.asarray(
        q_views,
        dtype=np.float64,
    )

    if q.ndim != 3:
        raise ValueError(
            "q_views must have shape [N,V,K]"
        )

    n, v, k = q.shape

    if n < 1 or v < 2 or k < 2:
        raise ValueError(
            "q_views requires N>=1, V>=2, K>=2"
        )

    if not np.isfinite(q).all():
        raise ValueError(
            "q_views must be finite"
        )

    if np.any(q < 0.0):
        raise ValueError(
            "q_views must be nonnegative"
        )

    if not np.allclose(
        q.sum(axis=2),
        1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "q_views rows must sum to one over K"
        )

    return q


def _normalize_probabilities(
    p: np.ndarray,
) -> np.ndarray:
    """
    p:
        [..., K]

    Returns normalized probability tensors with identical shape.
    """

    x = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        PROB_EPSILON,
        None,
    )

    denominator = x.sum(
        axis=-1,
        keepdims=True,
    )

    if (
        not np.isfinite(
            denominator
        ).all()
        or np.any(
            denominator <= 0.0
        )
    ):
        raise ValueError(
            "invalid probability normalization"
        )

    return (
        x / denominator
    )


def action_posteriors(
    q_views: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build generator/verifier distributions using the existing
    frozen BONE/P3 action ordering.

    q_views:
        [N, V, K]

    Returns:
        p_gen [N, S, K]
        p_ver [N, S, K]
    """

    q = validate_q_views(
        q_views
    )

    _, v, _ = q.shape

    actions, _, _ = (
        b0.action_arrays(v)
    )

    p_gen = np.stack(
        [
            q[
                :,
                action.generator_views,
                :,
            ].mean(axis=1)
            for action in actions
        ],
        axis=1,
    )

    p_ver = np.stack(
        [
            q[
                :,
                action.verifier_views,
                :,
            ].mean(axis=1)
            for action in actions
        ],
        axis=1,
    )

    # [N,S,K]
    if (
        p_gen.shape
        != p_ver.shape
        or p_gen.ndim != 3
    ):
        raise RuntimeError(
            "action posterior shape failure"
        )

    return (
        _normalize_probabilities(
            p_gen
        ),
        _normalize_probabilities(
            p_ver
        ),
    )


def utility_weighted_cross_view_score(
    q_views: np.ndarray,
    frozen_u: np.ndarray,
) -> float:
    """
    Compute S_cv(h).

    q_views:
        [N,V,K]

    frozen_u:
        [N,S]

    p_gen / p_ver:
        [N,S,K]

    JS:
        [N,S]

    agreement:
        [N,S]
    """

    p_gen, p_ver = (
        action_posteriors(
            q_views
        )
    )

    u = np.asarray(
        frozen_u,
        dtype=np.float64,
    )

    expected_shape = (
        p_gen.shape[0],
        p_gen.shape[1],
    )

    if u.shape != expected_shape:
        raise ValueError(
            "frozen_u shape must match [N,S]"
        )

    if not np.isfinite(u).all():
        raise ValueError(
            "frozen_u must be finite"
        )

    if np.any(u < 0.0) or np.any(
        u > 1.0
    ):
        raise ValueError(
            "frozen_u must be in [0,1]"
        )

    denominator = float(
        u.sum()
    )

    if (
        not np.isfinite(
            denominator
        )
        or denominator <= 0.0
    ):
        raise ValueError(
            "sum(frozen_u) must be positive"
        )

    midpoint = (
        0.5
        * (
            p_gen
            + p_ver
        )
    )

    midpoint = (
        _normalize_probabilities(
            midpoint
        )
    )

    # Each KL tensor: [N,S]
    kl_gen = np.sum(
        p_gen
        * (
            np.log(p_gen)
            - np.log(midpoint)
        ),
        axis=2,
    )

    kl_ver = np.sum(
        p_ver
        * (
            np.log(p_ver)
            - np.log(midpoint)
        ),
        axis=2,
    )

    js = (
        0.5
        * (
            kl_gen
            + kl_ver
        )
    )

    agreement = (
        1.0
        - js
        / np.log(2.0)
    )

    agreement = np.clip(
        agreement,
        0.0,
        1.0,
    )

    score = float(
        np.sum(
            u * agreement
        )
        / denominator
    )

    score = _finite_scalar(
        score,
        "S_cv",
    )

    if (
        score < -1e-12
        or score > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "S_cv outside [0,1]"
        )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def consensus_posterior(
    q_views: np.ndarray,
) -> np.ndarray:
    """
    q_views:
        [N,V,K]

    Returns:
        q_bar [N,K]
    """

    q = validate_q_views(
        q_views
    )

    q_bar = q.mean(
        axis=1
    )

    return _normalize_probabilities(
        q_bar
    )


def cluster_balance_score(
    q_views: np.ndarray,
) -> float:
    """
    Compute S_bal(h).

    consensus_q:
        [N,K]

    pi:
        [K]
    """

    q_bar = (
        consensus_posterior(
            q_views
        )
    )

    _, k = q_bar.shape

    pi = q_bar.mean(
        axis=0
    )

    pi = _normalize_probabilities(
        pi[None, :]
    )[0]

    entropy = float(
        -np.sum(
            pi * np.log(pi)
        )
    )

    score = (
        entropy
        / np.log(k)
    )

    score = _finite_scalar(
        score,
        "S_bal",
    )

    if (
        score < -1e-12
        or score > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "S_bal outside [0,1]"
        )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def hard_prediction(
    q_views: np.ndarray,
) -> np.ndarray:
    """
    q_views:
        [N,V,K]

    Returns:
        y [N]
    """

    q_bar = (
        consensus_posterior(
            q_views
        )
    )

    return np.argmax(
        q_bar,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )


def temporal_stability_score(
    current_prediction: np.ndarray,
    previous_prediction: np.ndarray,
) -> float:
    """
    Prediction-to-prediction NMI only.

    current_prediction:
        [N]

    previous_prediction:
        [N]

    No GT is accepted by this API.
    """

    current = np.asarray(
        current_prediction,
        dtype=np.int64,
    )

    previous = np.asarray(
        previous_prediction,
        dtype=np.int64,
    )

    if (
        current.ndim != 1
        or previous.ndim != 1
        or current.shape
        != previous.shape
        or current.size < 1
    ):
        raise ValueError(
            "predictions must have equal shape [N]"
        )

    score = float(
        normalized_mutual_info_score(
            previous,
            current,
            average_method="arithmetic",
        )
    )

    score = _finite_scalar(
        score,
        "S_stab",
    )

    if (
        score < -1e-12
        or score > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "S_stab outside [0,1]"
        )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def q_stop_score(
    s_cv: float,
    s_bal: float,
    s_stab: float,
) -> float:
    """
    Q_stop = geometric mean(S_cv, S_bal, S_stab).
    """

    components = np.asarray(
        [
            s_cv,
            s_bal,
            s_stab,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        components
    ).all():
        raise ValueError(
            "Q_stop components must be finite"
        )

    if (
        np.any(
            components < 0.0
        )
        or np.any(
            components > 1.0
        )
    ):
        raise ValueError(
            "Q_stop components must lie in [0,1]"
        )

    score = float(
        np.prod(
            components
        )
        ** (
            1.0
            / 3.0
        )
    )

    return _finite_scalar(
        score,
        "Q_stop",
    )


def evaluate_checkpoint(
    q_views: np.ndarray,
    frozen_u: np.ndarray,
    previous_prediction: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate one scored checkpoint h>=5.

    q_views:
        [N,V,K]

    frozen_u:
        [N,S]

    previous_prediction:
        [N]

    Returns an arm-independent GT-free score record.
    """

    prediction = (
        hard_prediction(
            q_views
        )
    )

    s_cv = (
        utility_weighted_cross_view_score(
            q_views,
            frozen_u,
        )
    )

    s_bal = (
        cluster_balance_score(
            q_views
        )
    )

    s_stab = (
        temporal_stability_score(
            prediction,
            previous_prediction,
        )
    )

    score = q_stop_score(
        s_cv,
        s_bal,
        s_stab,
    )

    return {
        "S_cv":
            s_cv,
        "S_bal":
            s_bal,
        "S_stab":
            s_stab,
        "Q_stop":
            score,
        # [N]
        "prediction":
            prediction,
    }


@dataclass
class EarlyStoppingState:
    """
    Shared arm-agnostic UASES state machine.

    Same class and update rule must be used for BASE and OURS.
    """

    patience_checkpoints: int = (
        PATIENCE_CHECKPOINTS
    )

    improvement_epsilon: float = (
        IMPROVEMENT_EPSILON
    )

    checkpoint_interval: int = (
        CHECKPOINT_INTERVAL
    )

    h_max: int = H_MAX

    best_score: float = float(
        "-inf"
    )

    best_h: int | None = None

    bad_count: int = 0

    last_h: int | None = None

    stopped: bool = False

    stop_h: int | None = None

    def __post_init__(
        self,
    ) -> None:

        if (
            self.patience_checkpoints
            < 1
        ):
            raise ValueError(
                "patience_checkpoints must be >=1"
            )

        if (
            self.checkpoint_interval
            < 1
        ):
            raise ValueError(
                "checkpoint_interval must be >=1"
            )

        if (
            self.h_max
            < self.checkpoint_interval
            or self.h_max
            % self.checkpoint_interval
            != 0
        ):
            raise ValueError(
                "h_max must be a positive multiple of checkpoint_interval"
            )

        if (
            not np.isfinite(
                self.improvement_epsilon
            )
            or self.improvement_epsilon
            < 0.0
        ):
            raise ValueError(
                "improvement_epsilon must be finite and >=0"
            )

    def update(
        self,
        h: int,
        score: float,
    ) -> dict[str, Any]:

        if self.stopped:
            raise RuntimeError(
                "cannot update a stopped early-stopping state"
            )

        if (
            not isinstance(
                h,
                (int, np.integer),
            )
            or int(h) < self.checkpoint_interval
            or int(h) > self.h_max
            or int(h)
            % self.checkpoint_interval
            != 0
        ):
            raise ValueError(
                "h must be a scored checkpoint"
            )

        h = int(h)

        if (
            self.last_h is not None
            and h
            != self.last_h
            + self.checkpoint_interval
        ):
            raise ValueError(
                "checkpoints must be supplied consecutively"
            )

        value = _finite_scalar(
            score,
            "Q_stop",
        )

        if (
            value < 0.0
            or value > 1.0
        ):
            raise ValueError(
                "Q_stop must lie in [0,1]"
            )

        improved = bool(
            self.best_h is None
            or value
            > self.best_score
            + self.improvement_epsilon
        )

        if improved:
            self.best_score = value
            self.best_h = h
            self.bad_count = 0
        else:
            # Equal / within epsilon does NOT replace the earlier best.
            self.bad_count += 1

        self.last_h = h

        patience_triggered = bool(
            self.bad_count
            >= self.patience_checkpoints
        )

        hmax_reached = bool(
            h == self.h_max
        )

        should_stop = bool(
            patience_triggered
            or hmax_reached
        )

        if should_stop:
            self.stopped = True
            self.stop_h = h

        return {
            "h":
                h,
            "score":
                value,
            "improved":
                improved,
            "bad_count":
                self.bad_count,
            "best_h":
                self.best_h,
            "best_score":
                self.best_score,
            "patience_triggered":
                patience_triggered,
            "hmax_reached":
                hmax_reached,
            "should_stop":
                should_stop,
            "return_h":
                self.best_h
                if should_stop
                else None,
        }


def protocol_constants() -> dict[str, Any]:
    """
    Small audit helper; contains no tunable runtime behavior.
    """

    return {
        "checkpoint_interval":
            CHECKPOINT_INTERVAL,
        "h_max":
            H_MAX,
        "patience_checkpoints":
            PATIENCE_CHECKPOINTS,
        "patience_native_epochs":
            PATIENCE_CHECKPOINTS
            * CHECKPOINT_INTERVAL,
        "improvement_epsilon":
            IMPROVEMENT_EPSILON,
        "score_components":
            (
                "S_cv",
                "S_bal",
                "S_stab",
            ),
        "score_aggregation":
            "geometric_mean",
        "uses_gt":
            False,
        "uses_sparse_labels":
            False,
        "uses_t_rel":
            False,
        "uses_semantic_loss":
            False,
        "arm_specific_formula":
            False,
    }
