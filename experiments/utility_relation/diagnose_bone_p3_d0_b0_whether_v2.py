"""
P3-D0-B0-v2 WHETHER Diagnostic
===============================

Post-seal GT-only diagnostic.

Goal:
    Evaluate whether frozen U_cycle really identifies valid
    semantic actions.

Important:
    Cluster IDs are permutation-invariant.
    Therefore we first derive ONE canonical carrier-level
    Hungarian mapping and then freeze it for ALL actions.

No training.
No hyperparameter selection.
No model modification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)


FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_d0_b0_whether_v2/aloi"
)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def build_canonical_mapping(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
):
    """
    Build ONE carrier-level Hungarian permutation.

    y_true : [N]
    y_pred : [N]

    Returns:
        mapping : [K]
            mapping[pred_cluster] = GT class
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "canonical mapping shape mismatch"
        )

    if (
        y_true.min() < 0
        or y_true.max() >= k
        or y_pred.min() < 0
        or y_pred.max() >= k
    ):
        raise ValueError(
            "labels outside 0..K-1"
        )

    counts = np.zeros(
        (k, k),
        dtype=np.int64,
    )

    np.add.at(
        counts,
        (y_pred, y_true),
        1,
    )

    rows, cols = linear_sum_assignment(
        counts.max() - counts
    )

    if rows.size != k or cols.size != k:
        raise RuntimeError(
            "Hungarian mapping is not a full K-way permutation"
        )

    mapping = np.full(
        k,
        -1,
        dtype=np.int64,
    )

    mapping[rows] = cols

    if np.any(mapping < 0):
        raise RuntimeError(
            "incomplete carrier mapping"
        )

    matched_acc = float(
        counts[rows, cols].sum()
        / y_true.size
    )

    return mapping, matched_acc, counts


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

    # --------------------------------------------------------
    # Pre-GT isolation verification
    # --------------------------------------------------------

    audit = json.loads(
        audit_path.read_text()
    )

    if audit.get(
        "GT_loaded_before_seal"
    ) is not False:
        raise RuntimeError(
            "invalid pre-GT isolation"
        )

    # --------------------------------------------------------
    # Load frozen carrier
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

        q_views = np.asarray(
            d["q_views"],
            dtype=np.float64,
        )

        p_gen_full = np.asarray(
            d["p_gen"],
            dtype=np.float64,
        )

        p_ver_full = np.asarray(
            d["p_ver"],
            dtype=np.float64,
        )

        y_gen_saved = np.asarray(
            d["y_gen"],
            dtype=np.int64,
        )

        U = np.asarray(
            d["query_U_cycle"],
            dtype=np.float64,
        )

    # q_views:
    # [N, V, K]
    #
    # p_gen/p_ver:
    # [N, S, K]
    #
    # query U:
    # [N_query, S]

    if q_views.ndim != 3:
        raise ValueError(
            "q_views must be [N,V,K]"
        )

    n, v, k = q_views.shape

    if p_gen_full.shape[0] != n:
        raise ValueError(
            "p_gen N mismatch"
        )

    if p_gen_full.shape != p_ver_full.shape:
        raise ValueError(
            "p_gen/p_ver mismatch"
        )

    if not np.array_equal(
        np.argmax(
            p_gen_full,
            axis=-1,
        ),
        y_gen_saved,
    ):
        raise RuntimeError(
            "saved y_gen parity failure"
        )

    # --------------------------------------------------------
    # Canonical carrier prediction.
    #
    # This exactly follows BONE.forward:
    #
    # label_fc per view
    # -> mean across V
    # -> argmax.
    # --------------------------------------------------------

    carrier_mean_posterior = (
        q_views.mean(axis=1)
    )

    carrier_prediction = np.argmax(
        carrier_mean_posterior,
        axis=1,
    ).astype(np.int64)

    used_clusters = np.unique(
        carrier_prediction
    )

    print()
    print("=" * 96)
    print(
        f"ALOI SEED{seed} CANONICAL CARRIER"
    )
    print("=" * 96)

    print(
        "N / V / K =",
        n,
        v,
        k,
    )

    print(
        "carrier used clusters =",
        used_clusters.size,
        "/",
        k,
    )

    # Hard gate:
    #
    # If carrier fused prediction does not use all K cluster
    # coordinates, a unique full-axis interpretation is not
    # established. Fail closed rather than assign arbitrary
    # semantics to unused dimensions.
    if used_clusters.size != k:
        raise RuntimeError(
            "FAIL-CLOSED: carrier fused prediction "
            f"uses only {used_clusters.size}/{k} clusters; "
            "canonical full-axis mapping is not identifiable."
        )

    # --------------------------------------------------------
    # Post-seal GT load
    # --------------------------------------------------------

    raw_y = np.asarray(
        scipy.io.loadmat(
            protocol.DATA_ROOT / "ALOI.mat",
            variable_names=("Y",),
        )["Y"]
    ).reshape(-1).astype(np.int64)

    if raw_y.size != n:
        raise RuntimeError(
            "ALOI GT length mismatch"
        )

    # sample_ids maps training tensor order -> raw row ID.
    y_training = raw_y[
        sample_ids
    ]

    # Normalize GT IDs if the MAT file is 1-based.
    unique_gt = np.unique(
        y_training
    )

    if (
        unique_gt.size == k
        and np.array_equal(
            unique_gt,
            np.arange(1, k + 1),
        )
    ):
        y_training = (
            y_training - 1
        )

    if not np.array_equal(
        np.unique(y_training),
        np.arange(k),
    ):
        raise RuntimeError(
            "GT IDs are not canonical 0..K-1"
        )

    # --------------------------------------------------------
    # ONE Hungarian map on the full carrier prediction.
    # --------------------------------------------------------

    mapping, carrier_acc, counts = (
        build_canonical_mapping(
            y_training,
            carrier_prediction,
            k,
        )
    )

    print(
        "canonical carrier Hungarian ACC =",
        carrier_acc,
    )

    print(
        "mapping is permutation =",
        np.array_equal(
            np.sort(mapping),
            np.arange(k),
        ),
    )

    # --------------------------------------------------------
    # Query-aligned action predictions
    # --------------------------------------------------------

    p_gen = p_gen_full[
        query_rows
    ]

    p_ver = p_ver_full[
        query_rows
    ]

    if (
        U.shape
        != p_gen.shape[:2]
    ):
        raise RuntimeError(
            f"query U mismatch: "
            f"U={U.shape}, "
            f"p_gen={p_gen.shape}"
        )

    y_gt = y_training[
        query_rows
    ]

    y_gen_raw = np.argmax(
        p_gen,
        axis=-1,
    )

    y_ver_raw = np.argmax(
        p_ver,
        axis=-1,
    )

    # Apply SAME fixed carrier-level permutation
    # to every query and every action.
    y_gen = mapping[
        y_gen_raw
    ]

    y_ver = mapping[
        y_ver_raw
    ]

    # --------------------------------------------------------
    # True / false consensus
    # --------------------------------------------------------

    closure = (
        y_gen_raw
        == y_ver_raw
    )

    # Permutation preserves equality, so this must agree.
    if not np.array_equal(
        closure,
        y_gen == y_ver,
    ):
        raise RuntimeError(
            "permutation closure invariance failed"
        )

    correct_gen = (
        y_gen
        == y_gt[:, None]
    )

    correct_ver = (
        y_ver
        == y_gt[:, None]
    )

    true_consensus = (
        closure
        & correct_gen
    )

    false_consensus = (
        closure
        & (~correct_gen)
    )

    positive_u = (
        U > 0
    )

    # Frozen U definition:
    # U must be zero when closure is false.
    max_open_u = (
        float(
            U[~closure].max()
        )
        if np.any(~closure)
        else 0.0
    )

    if max_open_u > 1e-8:
        raise RuntimeError(
            "U_cycle positive outside closure"
        )

    utility_mass = float(
        U.sum()
    )

    if utility_mass <= 0:
        raise RuntimeError(
            "zero utility mass"
        )

    true_mass = float(
        (
            U
            * true_consensus
        ).sum()
    )

    false_mass = float(
        (
            U
            * false_consensus
        ).sum()
    )

    weighted_true = safe_ratio(
        true_mass,
        utility_mass,
    )

    hfcr = safe_ratio(
        false_mass,
        utility_mass,
    )

    # --------------------------------------------------------
    # High-U diagnostics
    # --------------------------------------------------------

    positive_values = U[
        positive_u
    ]

    quantile_result = {}

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

        t = int(
            (
                selected
                & true_consensus
            ).sum()
        )

        f = int(
            (
                selected
                & false_consensus
            ).sum()
        )

        selected_mass = float(
            U[selected].sum()
        )

        false_selected_mass = float(
            U[
                selected
                & false_consensus
            ].sum()
        )

        quantile_result[
            f"q{int(q*100)}"
        ] = {
            "threshold":
                threshold,

            "pair_count":
                int(
                    selected.sum()
                ),

            "correct_rate":
                safe_ratio(
                    t,
                    t + f,
                ),

            "false_rate":
                safe_ratio(
                    f,
                    t + f,
                ),

            "weighted_false_ratio":
                safe_ratio(
                    false_selected_mass,
                    selected_mass,
                ),
        }

    formal = json.loads(
        metrics_path.read_text()
    )

    method_delta = formal[
        "delta_OURS_minus_BASE"
    ]

    result = {
        "experiment":
            "P3-D0-B0-v2_WHETHER",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "canonical_mapping": {
            "source":
                "carrier_mean_q_views",

            "single_global_mapping":
                True,

            "per_action_matching":
                False,

            "carrier_hungarian_ACC":
                carrier_acc,

            "used_clusters":
                int(
                    used_clusters.size
                ),

            "K":
                int(k),
        },

        "utility": {
            "positive_fraction":
                float(
                    positive_u.mean()
                ),

            "closure_rate":
                float(
                    closure.mean()
                ),

            "total_mass":
                utility_mass,
        },

        "primary_WHETHER": {
            "utility_weighted_true_consensus_ratio":
                weighted_true,

            "HFCR":
                hfcr,

            "partition_sum":
                weighted_true
                + hfcr,
        },

        "high_U":
            quantile_result,

        "formal_method_delta":
            method_delta,
    }

    out = (
        OUTPUT_ROOT
        / f"seed{seed}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        out
        / "whether_v2.json"
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
        f"ALOI SEED{seed} CORRECTED WHETHER"
    )
    print("=" * 96)

    print(
        "positive U fraction =",
        float(
            positive_u.mean()
        ),
    )

    print(
        "closure rate =",
        float(
            closure.mean()
        ),
    )

    print(
        "weighted TRUE consensus =",
        weighted_true,
    )

    print(
        "HFCR =",
        hfcr,
    )

    print(
        "partition sum =",
        weighted_true
        + hfcr,
    )

    print()
    print("HIGH-U:")

    for name, row in (
        quantile_result.items()
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
        "P3_D0_B0_V2_WHETHER_PASS"
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
