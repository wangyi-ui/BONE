"""
P3-D0-B1 WHAT Diagnostic
========================

Question:
    Given that U_cycle decides WHETHER an action is trusted,
    does T_rel / Pi provide the correct semantic direction (WHAT)?

Post-seal GT-only diagnostic.
No training.
No hyperparameter tuning.
No model modification.

Important:
    Pi is already indexed by semantic class through labeled_targets.
    Therefore no Hungarian mapping is required here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)


FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_d0_b1_what/aloi"
)


def safe_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:

    denominator = float(
        weights.sum()
    )

    if denominator <= 0:
        return float("nan")

    return float(
        (values * weights).sum()
        / denominator
    )


def analyze(seed: int):

    root = (
        FORMAL_ROOT
        / f"seed{seed}"
    )

    carrier_path = (
        root
        / "carrier/carrier_pre_gt.npz"
    )

    audit_path = (
        root
        / "seal/pre_gt_audit.json"
    )

    metrics_path = (
        root
        / "formal_metrics.json"
    )

    for p in (
        carrier_path,
        audit_path,
        metrics_path,
    ):
        if not p.exists():
            raise FileNotFoundError(p)

    # ========================================================
    # 1. Verify post-seal diagnostic contract
    # ========================================================

    audit = json.loads(
        audit_path.read_text()
    )

    if audit.get(
        "GT_loaded_before_seal"
    ) is not False:
        raise RuntimeError(
            "invalid pre-GT isolation"
        )

    # ========================================================
    # 2. Load frozen semantic carrier
    # ========================================================

    with np.load(
        carrier_path,
        allow_pickle=False,
    ) as d:

        sample_ids = np.asarray(
            d["sample_ids"],
            dtype=np.int64,
        )

        query_rows = np.asarray(
            d["query_rows"],
            dtype=np.int64,
        )

        labeled_targets = np.asarray(
            d["labeled_targets"],
            dtype=np.int64,
        )

        Pi = np.asarray(
            d["Pi"],
            dtype=np.float32,
        )

        T_rel = np.asarray(
            d["T_rel"],
            dtype=np.float32,
        )

        U = np.asarray(
            d["query_U_cycle"],
            dtype=np.float32,
        )

    # Pi:
    # [N_query, S, K]
    #
    # T_rel:
    # [N_query, L, S]
    #
    # U:
    # [N_query, S]

    if Pi.ndim != 3:
        raise ValueError(
            "Pi must be [N_query,S,K]"
        )

    nq, action_num, k = Pi.shape

    if U.shape != (
        nq,
        action_num,
    ):
        raise ValueError(
            "U/Pi shape mismatch"
        )

    if (
        labeled_targets.ndim != 1
        or T_rel.shape
        != (
            nq,
            labeled_targets.size,
            action_num,
        )
    ):
        raise ValueError(
            "T_rel shape mismatch"
        )

    if query_rows.size != nq:
        raise ValueError(
            "query_rows/Pi mismatch"
        )

    if not np.allclose(
        Pi.sum(axis=2),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Pi is not a probability distribution"
        )

    # ========================================================
    # 3. Verify T_rel is exactly Pi indexed by anchor targets
    # ========================================================

    expected_T = np.take(
        Pi,
        labeled_targets,
        axis=2,
    ).transpose(
        0,
        2,
        1,
    )

    max_t_error = float(
        np.max(
            np.abs(
                expected_T
                - T_rel
            )
        )
    )

    print()
    print(
        "T_rel / Pi max_abs_error =",
        max_t_error,
    )

    if max_t_error > 1e-6:
        raise RuntimeError(
            "T_rel != Pi indexed by labeled_targets"
        )

    # ========================================================
    # 4. Post-seal GT load
    # ========================================================

    raw_y = np.asarray(
        scipy.io.loadmat(
            protocol.DATA_ROOT / "ALOI.mat",
            variable_names=("Y",),
        )["Y"]
    ).reshape(-1).astype(np.int64)

    if raw_y.size != sample_ids.size:
        raise RuntimeError(
            "GT/sample-ID size mismatch"
        )

    y_training = raw_y[
        sample_ids
    ]

    unique_gt = np.unique(
        y_training
    )

    # Normalize potential MATLAB 1..K labels.
    if np.array_equal(
        unique_gt,
        np.arange(
            1,
            k + 1,
        ),
    ):
        y_training = (
            y_training - 1
        )

    if not np.array_equal(
        np.unique(y_training),
        np.arange(k),
    ):
        raise RuntimeError(
            "GT does not map to 0..K-1"
        )

    y_query = y_training[
        query_rows
    ]

    # ========================================================
    # 5. Core WHAT diagnostics
    # ========================================================

    # Predicted semantic class:
    #
    # [N_query, S]
    semantic_pred = np.argmax(
        Pi,
        axis=2,
    )

    semantic_correct = (
        semantic_pred
        == y_query[:, None]
    )

    # True-class semantic probability:
    #
    # [N_query, S]
    true_prob = np.take_along_axis(
        Pi,
        y_query[
            :,
            None,
            None,
        ].repeat(
            action_num,
            axis=1,
        ),
        axis=2,
    )[
        :,
        :,
        0,
    ]

    # Highest incorrect semantic class probability.
    Pi_other = Pi.copy()

    rows = np.arange(
        nq
    )[:, None]

    actions = np.arange(
        action_num
    )[None, :]

    Pi_other[
        rows,
        actions,
        y_query[:, None],
    ] = -np.inf

    max_other = np.max(
        Pi_other,
        axis=2,
    )

    margin = (
        true_prob
        - max_other
    )

    # Utility-weighted WHAT:
    #
    # This is especially important because actual semantic
    # action is controlled by U_cycle.
    weighted_top1 = safe_weighted_mean(
        semantic_correct.astype(
            np.float32
        ),
        U,
    )

    weighted_true_prob = (
        safe_weighted_mean(
            true_prob,
            U,
        )
    )

    weighted_margin = (
        safe_weighted_mean(
            margin,
            U,
        )
    )

    # ========================================================
    # 6. Per-action relation discrimination
    #
    # Positive class = GT semantic class.
    #
    # Use Pi directly at class level. Since every class has
    # exactly two labeled anchors, T_rel is a duplicated
    # relation representation of these same semantic scores.
    # ========================================================

    one_hot = np.zeros(
        (
            nq,
            k,
        ),
        dtype=np.int8,
    )

    one_hot[
        np.arange(nq),
        y_query,
    ] = 1

    action_metrics = []

    for s in range(
        action_num
    ):

        scores = Pi[
            :,
            s,
            :,
        ].reshape(-1)

        truth = one_hot.reshape(-1)

        auc = float(
            roc_auc_score(
                truth,
                scores,
            )
        )

        ap = float(
            average_precision_score(
                truth,
                scores,
            )
        )

        action_metrics.append(
            {
                "action":
                    s,

                "semantic_top1_accuracy":
                    float(
                        semantic_correct[
                            :,
                            s,
                        ].mean()
                    ),

                "mean_true_probability":
                    float(
                        true_prob[
                            :,
                            s,
                        ].mean()
                    ),

                "mean_margin":
                    float(
                        margin[
                            :,
                            s,
                        ].mean()
                    ),

                "relation_AUC":
                    auc,

                "relation_AP":
                    ap,
            }
        )

    # ========================================================
    # 7. High-U conditional WHAT
    # ========================================================

    positive_u = (
        U > 0
    )

    positive_values = U[
        positive_u
    ]

    high_u = {}

    for q in (
        0.50,
        0.75,
        0.90,
        0.95,
    ):

        threshold = float(
            np.quantile(
                positive_values,
                q,
            )
        )

        selected = (
            positive_u
            & (U >= threshold)
        )

        if not np.any(
            selected
        ):
            continue

        high_u[
            f"q{int(q*100)}"
        ] = {
            "threshold":
                threshold,

            "pair_count":
                int(
                    selected.sum()
                ),

            "semantic_top1_accuracy":
                float(
                    semantic_correct[
                        selected
                    ].mean()
                ),

            "mean_true_probability":
                float(
                    true_prob[
                        selected
                    ].mean()
                ),

            "mean_margin":
                float(
                    margin[
                        selected
                    ].mean()
                ),
        }

    # ========================================================
    # 8. Low-U vs high-U monotonicity
    # ========================================================

    deciles = []

    edges = np.quantile(
        positive_values,
        np.linspace(
            0,
            1,
            11,
        ),
    )

    for b in range(
        10
    ):

        lo = float(
            edges[b]
        )

        hi = float(
            edges[
                b + 1
            ]
        )

        if b < 9:
            selected = (
                positive_u
                & (U >= lo)
                & (U < hi)
            )
        else:
            selected = (
                positive_u
                & (U >= lo)
                & (U <= hi)
            )

        if not np.any(
            selected
        ):
            continue

        deciles.append(
            {
                "bin":
                    b + 1,

                "u_min":
                    lo,

                "u_max":
                    hi,

                "count":
                    int(
                        selected.sum()
                    ),

                "semantic_top1_accuracy":
                    float(
                        semantic_correct[
                            selected
                        ].mean()
                    ),

                "mean_true_probability":
                    float(
                        true_prob[
                            selected
                        ].mean()
                    ),

                "mean_margin":
                    float(
                        margin[
                            selected
                        ].mean()
                    ),
            }
        )

    formal = json.loads(
        metrics_path.read_text()
    )

    method_delta = formal[
        "delta_OURS_minus_BASE"
    ]

    result = {
        "experiment":
            "P3-D0-B1_WHAT_diagnostic",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "T_rel_Pi_parity_max_abs_error":
            max_t_error,

        "overall_WHAT": {
            "semantic_top1_accuracy":
                float(
                    semantic_correct.mean()
                ),

            "mean_true_probability":
                float(
                    true_prob.mean()
                ),

            "mean_margin":
                float(
                    margin.mean()
                ),

            "utility_weighted_semantic_top1_accuracy":
                weighted_top1,

            "utility_weighted_true_probability":
                weighted_true_prob,

            "utility_weighted_margin":
                weighted_margin,

            "mean_relation_AUC":
                float(
                    np.mean(
                        [
                            x[
                                "relation_AUC"
                            ]
                            for x
                            in action_metrics
                        ]
                    )
                ),

            "mean_relation_AP":
                float(
                    np.mean(
                        [
                            x[
                                "relation_AP"
                            ]
                            for x
                            in action_metrics
                        ]
                    )
                ),
        },

        "per_action":
            action_metrics,

        "high_U_WHAT":
            high_u,

        "utility_deciles":
            deciles,

        "formal_method_delta":
            method_delta,
    }

    out_dir = (
        OUTPUT_ROOT
        / f"seed{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        out_dir
        / "what_diagnostic.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 96)
    print(
        f"ALOI SEED{seed} WHAT DIAGNOSTIC"
    )
    print("=" * 96)

    for key, value in (
        result[
            "overall_WHAT"
        ].items()
    ):
        print(
            f"{key:45s} = {value}"
        )

    print()
    print("HIGH-U WHAT:")

    for name, row in (
        high_u.items()
    ):
        print(
            name,
            row,
        )

    print()
    print(
        "FORMAL METHOD DELTA ="
    )

    print(
        json.dumps(
            method_delta,
            indent=2,
        )
    )

    print()
    print(
        "P3_D0_B1_WHAT_DIAGNOSTIC_PASS"
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(20, 30, 50),
        required=True,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.execute:
        raise SystemExit(
            "Add --execute."
        )

    analyze(
        args.training_seed
    )


if __name__ == "__main__":
    main()
