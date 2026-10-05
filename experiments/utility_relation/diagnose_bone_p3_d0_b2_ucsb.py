"""
P3-D0-B2 UCSB Counterfactual Semantic Diagnostic
=================================================

UCSB:
    Utility-Conditioned Semantic Bridge

Purpose:
    Test whether U_cycle should bridge global semantic signatures
    and action-specific semantic signatures before constructing T_rel.

NO TRAINING.
NO MODEL MODIFICATION.
NO NEW HYPERPARAMETERS.
POST-SEAL GT-ONLY DIAGNOSTIC.

Arms:
    CURRENT
        Pi_global

    ACTION_ONLY
        Pi_action

    TRUE_U_BRIDGE
        (1-U) * Pi_global + U * Pi_action

    SHUFFLE_U_BRIDGE
        (1-U_shuffle) * Pi_global + U_shuffle * Pi_action

Evaluation is always weighted by the TRUE frozen U_cycle,
because the actual semantic action mass remains controlled by U_cycle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)


FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_d0_b2_ucsb/aloi"
)

SHUFFLE_SEED = 20261002


def normalize_energy(
    energy: np.ndarray,
) -> np.ndarray:
    """
    energy:
        [N_query, S, C]

    return:
        Pi [N_query, S, C]
    """

    if (
        energy.ndim != 3
        or not np.isfinite(energy).all()
        or np.any(energy < 0)
    ):
        raise ValueError(
            "invalid semantic energy"
        )

    denominator = energy.sum(
        axis=2,
        keepdims=True,
    )

    if (
        not np.isfinite(
            denominator
        ).all()
        or np.any(
            denominator <= 0
        )
    ):
        raise ValueError(
            "semantic energy denominator "
            "must be positive"
        )

    Pi = (
        energy
        / denominator
    )

    if not np.allclose(
        Pi.sum(axis=2),
        1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Pi normalization failure"
        )

    return Pi


def weighted_mean(
    x: np.ndarray,
    w: np.ndarray,
) -> float:

    denominator = float(
        w.sum()
    )

    if denominator <= 0:
        raise ValueError(
            "zero utility mass"
        )

    return float(
        (x * w).sum()
        / denominator
    )


def evaluate_pi(
    Pi: np.ndarray,
    y_query: np.ndarray,
    true_u: np.ndarray,
) -> dict:
    """
    Pi:
        [N_query, S, C]

    y_query:
        [N_query]

    true_u:
        [N_query, S]
    """

    nq, actions, classes = (
        Pi.shape
    )

    if y_query.shape != (
        nq,
    ):
        raise ValueError(
            "GT/Pi shape mismatch"
        )

    if true_u.shape != (
        nq,
        actions,
    ):
        raise ValueError(
            "U/Pi shape mismatch"
        )

    pred = np.argmax(
        Pi,
        axis=2,
    )

    correct = (
        pred
        == y_query[:, None]
    )

    true_prob = np.take_along_axis(
        Pi,
        np.broadcast_to(
            y_query[
                :,
                None,
                None,
            ],
            (
                nq,
                actions,
                1,
            ),
        ),
        axis=2,
    )[
        :,
        :,
        0,
    ]

    other = Pi.copy()

    rows = np.arange(
        nq
    )[:, None]

    action_ids = np.arange(
        actions
    )[None, :]

    other[
        rows,
        action_ids,
        y_query[:, None],
    ] = -np.inf

    strongest_wrong = np.max(
        other,
        axis=2,
    )

    margin = (
        true_prob
        - strongest_wrong
    )

    positive_u = (
        true_u > 0
    )

    return {
        "semantic_top1_accuracy":
            float(
                correct.mean()
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
            weighted_mean(
                correct.astype(
                    np.float64
                ),
                true_u,
            ),

        "utility_weighted_true_probability":
            weighted_mean(
                true_prob,
                true_u,
            ),

        "utility_weighted_margin":
            weighted_mean(
                margin,
                true_u,
            ),

        "positive_U_semantic_top1_accuracy":
            float(
                correct[
                    positive_u
                ].mean()
            ),

        "positive_U_mean_margin":
            float(
                margin[
                    positive_u
                ].mean()
            ),
    }


def analyze_seed(
    seed: int,
) -> dict:

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

    formal_metrics_path = (
        root
        / "formal_metrics.json"
    )

    for p in (
        carrier_path,
        audit_path,
        formal_metrics_path,
    ):
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    # --------------------------------------------------------
    # Pre-GT isolation check
    # --------------------------------------------------------

    audit = json.loads(
        audit_path.read_text()
    )

    if audit.get(
        "GT_loaded_before_seal"
    ) is not False:
        raise RuntimeError(
            "invalid GT isolation"
        )

    # --------------------------------------------------------
    # Load frozen carrier tensors
    # --------------------------------------------------------

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

        anchor_rows = np.asarray(
            d["anchor_rows"],
            dtype=np.int64,
        )

        labeled_targets = np.asarray(
            d["labeled_targets"],
            dtype=np.int64,
        )

        p_gen = np.asarray(
            d["p_gen"],
            dtype=np.float32,
        )

        saved_Pi = np.asarray(
            d["Pi"],
            dtype=np.float32,
        )

        U = np.asarray(
            d["query_U_cycle"],
            dtype=np.float32,
        )

    # p_gen:
    # [N, S, K_cluster]
    #
    # saved Pi:
    # [N_query, S, K_semantic]
    #
    # U:
    # [N_query, S]

    if p_gen.ndim != 3:
        raise ValueError(
            "p_gen must be [N,S,K]"
        )

    n, actions, k = (
        p_gen.shape
    )

    nq = query_rows.size

    if saved_Pi.shape != (
        nq,
        actions,
        k,
    ):
        raise ValueError(
            "saved Pi shape mismatch"
        )

    if U.shape != (
        nq,
        actions,
    ):
        raise ValueError(
            "query U shape mismatch"
        )

    if (
        anchor_rows.shape
        != labeled_targets.shape
    ):
        raise ValueError(
            "anchor target mismatch"
        )

    # --------------------------------------------------------
    # CURRENT:
    # reproduce frozen global class signature exactly.
    #
    # anchor_signature:
    # [L, K_cluster]
    #
    # class_global:
    # [K_semantic, K_cluster]
    # --------------------------------------------------------

    anchor_signature = (
        p_gen[
            anchor_rows
        ].mean(
            axis=1
        )
    )

    class_global = np.empty(
        (
            k,
            k,
        ),
        dtype=np.float32,
    )

    for c in range(
        k
    ):

        members = (
            anchor_signature[
                labeled_targets == c
            ]
        )

        if members.shape != (
            2,
            k,
        ):
            raise RuntimeError(
                f"class {c}: expected "
                "exactly 2 anchors"
            )

        class_global[
            c
        ] = members.mean(
            axis=0
        )

    p_query = p_gen[
        query_rows
    ]

    energy_global = np.einsum(
        "usk,ck->usc",
        p_query,
        class_global,
    )

    Pi_global = normalize_energy(
        energy_global
    ).astype(
        np.float32
    )

    parity_error = float(
        np.max(
            np.abs(
                Pi_global
                - saved_Pi
            )
        )
    )

    print()
    print(
        f"seed{seed} CURRENT Pi parity "
        f"max_abs_error = {parity_error}"
    )

    if parity_error > 1e-6:
        raise RuntimeError(
            "CURRENT Pi reconstruction "
            "does not match frozen carrier"
        )

    # --------------------------------------------------------
    # ACTION-SPECIFIC CLASS SIGNATURE
    #
    # class_action:
    # [S, K_semantic, K_cluster]
    # --------------------------------------------------------

    class_action = np.empty(
        (
            actions,
            k,
            k,
        ),
        dtype=np.float32,
    )

    for s in range(
        actions
    ):

        for c in range(
            k
        ):

            members = p_gen[
                anchor_rows[
                    labeled_targets
                    == c
                ],
                s,
                :,
            ]

            if members.shape != (
                2,
                k,
            ):
                raise RuntimeError(
                    f"action{s}/class{c}: "
                    "expected exactly 2 anchors"
                )

            class_action[
                s,
                c,
            ] = members.mean(
                axis=0
            )

    energy_action = np.einsum(
        "usk,sck->usc",
        p_query,
        class_action,
    )

    Pi_action = normalize_energy(
        energy_action
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # TRUE-U UCSB
    #
    # lambda:
    # [N_query, S, 1]
    #
    # U is frozen / detached artifact.
    # --------------------------------------------------------

    lam_true = U[
        :,
        :,
        None,
    ].astype(
        np.float32
    )

    Pi_true_bridge = (
        (
            1.0
            - lam_true
        )
        * Pi_global
        +
        lam_true
        * Pi_action
    )

    if not np.allclose(
        Pi_true_bridge.sum(
            axis=2
        ),
        1.0,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "TRUE-U bridge normalization "
            "failure"
        )

    # --------------------------------------------------------
    # SHUFFLED-U NEGATIVE CONTROL
    #
    # Permute query rows only.
    # Preserve each row's full action-wise U vector.
    # --------------------------------------------------------

    rng = np.random.RandomState(
        SHUFFLE_SEED
    )

    permutation = rng.permutation(
        nq
    )

    U_shuffle = U[
        permutation
    ]

    if np.array_equal(
        U_shuffle,
        U,
    ):
        raise RuntimeError(
            "shuffle-U control is identical"
        )

    # Distribution preservation audit
    if not np.array_equal(
        np.sort(
            U_shuffle.reshape(-1)
        ),
        np.sort(
            U.reshape(-1)
        ),
    ):
        raise RuntimeError(
            "shuffle-U distribution changed"
        )

    lam_shuffle = (
        U_shuffle[
            :,
            :,
            None,
        ]
    )

    Pi_shuffle_bridge = (
        (
            1.0
            - lam_shuffle
        )
        * Pi_global
        +
        lam_shuffle
        * Pi_action
    )

    # --------------------------------------------------------
    # Post-seal GT
    # --------------------------------------------------------

    raw_y = np.asarray(
        scipy.io.loadmat(
            protocol.DATA_ROOT
            / "ALOI.mat",
            variable_names=(
                "Y",
            ),
        )["Y"]
    ).reshape(
        -1
    ).astype(
        np.int64
    )

    if raw_y.size != n:
        raise RuntimeError(
            "GT length mismatch"
        )

    y_training = raw_y[
        sample_ids
    ]

    unique_y = np.unique(
        y_training
    )

    if np.array_equal(
        unique_y,
        np.arange(
            1,
            k + 1,
        ),
    ):
        y_training = (
            y_training
            - 1
        )

    if not np.array_equal(
        np.unique(
            y_training
        ),
        np.arange(k),
    ):
        raise RuntimeError(
            "GT semantic IDs "
            "are not 0..K-1"
        )

    y_query = y_training[
        query_rows
    ]

    # --------------------------------------------------------
    # Evaluate all counterfactual arms
    #
    # NOTE:
    # All arms are evaluated with TRUE U as weighting mass.
    # --------------------------------------------------------

    arms = {
        "CURRENT":
            evaluate_pi(
                Pi_global,
                y_query,
                U,
            ),

        "ACTION_ONLY":
            evaluate_pi(
                Pi_action,
                y_query,
                U,
            ),

        "TRUE_U_BRIDGE":
            evaluate_pi(
                Pi_true_bridge,
                y_query,
                U,
            ),

        "SHUFFLE_U_BRIDGE":
            evaluate_pi(
                Pi_shuffle_bridge,
                y_query,
                U,
            ),
    }

    formal = json.loads(
        formal_metrics_path.read_text()
    )

    result = {
        "experiment":
            "P3-D0-B2_UCSB_counterfactual",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "NO_TRAINING":
            True,

        "NO_HYPERPARAMETER_SELECTION":
            True,

        "CURRENT_Pi_parity_max_abs_error":
            parity_error,

        "bridge_definition":
            "Pi*=((1-U)*Pi_global)+(U*Pi_action)",

        "shuffle_seed":
            SHUFFLE_SEED,

        "arms":
            arms,

        "formal_method_delta":
            formal[
                "delta_OURS_minus_BASE"
            ],
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
        / "ucsb_diagnostic.json"
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
    print("=" * 100)
    print(
        f"ALOI SEED{seed} UCSB DIAGNOSTIC"
    )
    print("=" * 100)

    for arm, metrics in (
        arms.items()
    ):

        print()
        print(
            f"[{arm}]"
        )

        print(
            "U-weighted top1 =",
            metrics[
                "utility_weighted_semantic_top1_accuracy"
            ],
        )

        print(
            "U-weighted true probability =",
            metrics[
                "utility_weighted_true_probability"
            ],
        )

        print(
            "U-weighted margin =",
            metrics[
                "utility_weighted_margin"
            ],
        )

    print()
    print(
        "FORMAL ACC DELTA =",
        formal[
            "delta_OURS_minus_BASE"
        ]["ACC"],
    )

    print()
    print(
        "P3_D0_B2_UCSB_SEED_PASS"
    )

    return result


def summarize():

    rows = {}

    for seed in (
        20,
        30,
        50,
    ):

        path = (
            OUTPUT_ROOT
            / f"seed{seed}"
            / "ucsb_diagnostic.json"
        )

        if not path.exists():
            continue

        rows[
            seed
        ] = json.loads(
            path.read_text()
        )

    if set(
        rows
    ) != {
        20,
        30,
        50,
    }:
        raise RuntimeError(
            "missing three-seed UCSB outputs"
        )

    print()
    print("=" * 110)
    print(
        "P3-D0-B2 UCSB THREE-SEED SUMMARY"
    )
    print("=" * 110)

    primary = {}

    for seed in (
        20,
        30,
        50,
    ):

        d = rows[
            seed
        ]

        print()
        print(
            f"================ seed{seed} ================"
        )

        current = (
            d["arms"][
                "CURRENT"
            ]
        )

        action = (
            d["arms"][
                "ACTION_ONLY"
            ]
        )

        bridge = (
            d["arms"][
                "TRUE_U_BRIDGE"
            ]
        )

        shuffle = (
            d["arms"][
                "SHUFFLE_U_BRIDGE"
            ]
        )

        current_margin = (
            current[
                "utility_weighted_margin"
            ]
        )

        bridge_margin = (
            bridge[
                "utility_weighted_margin"
            ]
        )

        primary[
            seed
        ] = {
            "current_margin":
                current_margin,

            "bridge_margin":
                bridge_margin,

            "delta_margin":
                (
                    bridge_margin
                    - current_margin
                ),

            "action_margin":
                action[
                    "utility_weighted_margin"
                ],

            "shuffle_margin":
                shuffle[
                    "utility_weighted_margin"
                ],

            "current_top1":
                current[
                    "utility_weighted_semantic_top1_accuracy"
                ],

            "bridge_top1":
                bridge[
                    "utility_weighted_semantic_top1_accuracy"
                ],
        }

        for key, value in (
            primary[
                seed
            ].items()
        ):
            print(
                f"{key:24s} = {value}"
            )

    mean_true_margin = float(
        np.mean(
            [
                primary[s][
                    "bridge_margin"
                ]
                for s in (
                    20,
                    30,
                    50,
                )
            ]
        )
    )

    mean_shuffle_margin = float(
        np.mean(
            [
                primary[s][
                    "shuffle_margin"
                ]
                for s in (
                    20,
                    30,
                    50,
                )
            ]
        )
    )

    # --------------------------------------------------------
    # PRE-REGISTERED MECHANISM GATE
    # --------------------------------------------------------

    failing_seed_gain = (
        primary[30][
            "delta_margin"
        ] > 0.0
        and
        primary[50][
            "delta_margin"
        ] > 0.0
    )

    positive_seed_safe = (
        primary[20][
            "delta_margin"
        ] >= 0.0
    )

    utility_specificity = (
        mean_true_margin
        > mean_shuffle_margin
    )

    mean_bridge_top1 = float(
        np.mean(
            [
                primary[s][
                    "bridge_top1"
                ]
                for s in (
                    20,
                    30,
                    50,
                )
            ]
        )
    )

    mean_current_top1 = float(
        np.mean(
            [
                primary[s][
                    "current_top1"
                ]
                for s in (
                    20,
                    30,
                    50,
                )
            ]
        )
    )

    top1_noninferior = (
        mean_bridge_top1
        >= mean_current_top1
    )

    final_pass = bool(
        failing_seed_gain
        and positive_seed_safe
        and utility_specificity
        and top1_noninferior
    )

    print()
    print("=" * 110)
    print(
        "PRE-REGISTERED UCSB MECHANISM GATE"
    )
    print("=" * 110)

    print(
        "seed30 margin improves =",
        primary[30][
            "delta_margin"
        ] > 0.0,
    )

    print(
        "seed50 margin improves =",
        primary[50][
            "delta_margin"
        ] > 0.0,
    )

    print(
        "seed20 non-degradation =",
        positive_seed_safe,
    )

    print(
        "TRUE-U mean margin > SHUFFLE-U =",
        utility_specificity,
    )

    print(
        "bridge mean top1 >= current mean top1 =",
        top1_noninferior,
    )

    print(
        "mean TRUE-U margin =",
        mean_true_margin,
    )

    print(
        "mean SHUFFLE-U margin =",
        mean_shuffle_margin,
    )

    print(
        "mean CURRENT top1 =",
        mean_current_top1,
    )

    print(
        "mean TRUE-U top1 =",
        mean_bridge_top1,
    )

    print()
    print(
        "FINAL_UCSB_DIAGNOSTIC_DECISION =",
        (
            "PASS"
            if final_pass
            else "FAIL-CLOSED"
        ),
    )

    print()
    print(
        "P3_D0_B2_UCSB_THREE_SEED_COMPLETE"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(
            20,
            30,
            50,
        ),
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
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

    if args.summarize:

        summarize()

    else:

        if (
            args.training_seed
            is None
        ):
            raise SystemExit(
                "--training-seed required"
            )

        analyze_seed(
            args.training_seed
        )


if __name__ == "__main__":
    main()
