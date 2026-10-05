"""
P3-D0-B0 WHETHER Diagnostic
============================

Question:
    Does U_cycle reliably indicate semantic action validity?

This is a POST-SEAL GT-only diagnostic.
It performs NO training and NO model selection.

Primary contrast:
    ALOI seed20 : method-positive
    ALOI seed30 : method-negative

Main diagnostic:
    High-utility False-Consensus Ratio (HFCR)

For query i and action s:

    y_gen = argmax p_gen[i,s]
    y_ver = argmax p_ver[i,s]

    closure =
        1[y_gen == y_ver]

    true_consensus =
        1[y_gen == y_ver == y_gt]

    false_consensus =
        1[y_gen == y_ver != y_gt]

U_cycle is evaluated only as a diagnostic variable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)
from experiments.utility_relation import (
    run_bone_p3_c0_cross_dataset_formal as c0,
)
from utils.data_util import get_dataset, D_ALOI


FORMAL_ROOT = Path(
    "outputs/bone_p3/p3_c0_formal/aloi"
)

OUTPUT_ROOT = Path(
    "outputs/bone_p3/p3_d0_b0_whether/aloi"
)


def pick_key(npz, candidates):
    for key in candidates:
        if key in npz.files:
            return key
    return None


def describe_array(name, x):
    print(
        f"{name:24s}",
        "shape=",
        tuple(x.shape),
        "dtype=",
        x.dtype,
    )


def safe_ratio(num, den):
    if den <= 0:
        return float("nan")
    return float(num / den)


def analyze_seed(seed: int) -> dict:

    root = FORMAL_ROOT / f"seed{seed}"

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

    audit = json.loads(
        audit_path.read_text()
    )

    assert (
        audit["GT_loaded_before_seal"]
        is False
    )

    # --------------------------------------------------------
    # 1. Load frozen pre-GT carrier
    # --------------------------------------------------------

    carrier = np.load(
        carrier_path,
        allow_pickle=False,
    )

    print()
    print("=" * 96)
    print(
        f"ALOI SEED{seed} FROZEN CARRIER INVENTORY"
    )
    print("=" * 96)

    for key in carrier.files:
        arr = np.asarray(carrier[key])
        describe_array(key, arr)

    # --------------------------------------------------------
    # 2. Resolve carrier keys robustly
    # --------------------------------------------------------

    p_gen_key = pick_key(
        carrier,
        (
            "query_p_gen",
            "p_gen",
            "query_pgen",
        ),
    )

    p_ver_key = pick_key(
        carrier,
        (
            "query_p_ver",
            "p_ver",
            "query_pver",
        ),
    )

    u_key = pick_key(
        carrier,
        (
            "query_U_cycle",
            "U_cycle",
            "query_u_cycle",
            "u_cycle",
        ),
    )

    query_rows_key = pick_key(
        carrier,
        (
            "query_rows",
            "query_row_positions",
            "query_indices",
            "unlabeled_rows",
        ),
    )

    sample_ids_key = pick_key(
        carrier,
        (
            "sample_ids",
            "training_sample_ids",
            "raw_sample_ids",
        ),
    )

    print()
    print("resolved p_gen key =", p_gen_key)
    print("resolved p_ver key =", p_ver_key)
    print("resolved U key     =", u_key)
    print("resolved rows key  =", query_rows_key)
    print("resolved IDs key   =", sample_ids_key)

    if p_gen_key is None:
        raise KeyError(
            "Could not resolve p_gen carrier key."
        )

    if p_ver_key is None:
        raise KeyError(
            "Could not resolve p_ver carrier key."
        )

    if u_key is None:
        raise KeyError(
            "Could not resolve U_cycle carrier key."
        )

    # --------------------------------------------------------
    # Align full-sample carrier posteriors with query-only U.
    #
    # p_gen / p_ver:
    #     [N, S, K]
    #
    # query_U_cycle:
    #     [N_query, S]
    #
    # query_rows:
    #     [N_query]
    #
    # Semantic relation actions operate on QUERY rows only.
    # --------------------------------------------------------

    p_gen_full = np.asarray(
        carrier[p_gen_key],
        dtype=np.float64,
    )

    p_ver_full = np.asarray(
        carrier[p_ver_key],
        dtype=np.float64,
    )

    U = np.asarray(
        carrier[u_key],
        dtype=np.float64,
    )

    if p_gen_full.ndim != 3:
        raise ValueError(
            f"p_gen must be [N,S,K], got {p_gen_full.shape}"
        )

    if p_ver_full.shape != p_gen_full.shape:
        raise ValueError(
            "p_gen/p_ver full-sample shape mismatch"
        )

    if query_rows_key is not None:

        query_rows_for_alignment = np.asarray(
            carrier[query_rows_key]
        ).reshape(-1).astype(np.int64)

    else:

        query_rows_for_alignment = None

    # Case 1:
    # carrier arrays are already query-level.
    if p_gen_full.shape[:2] == U.shape:

        p_gen = p_gen_full
        p_ver = p_ver_full

        print(
            "p_gen/p_ver already query-aligned"
        )

    # Case 2:
    # p_gen/p_ver contain all N samples, while U is query-only.
    elif (
        query_rows_for_alignment is not None
        and U.shape[0] == query_rows_for_alignment.size
        and U.shape[1] == p_gen_full.shape[1]
    ):

        if (
            query_rows_for_alignment.min() < 0
            or query_rows_for_alignment.max() >= p_gen_full.shape[0]
        ):
            raise ValueError(
                "query_rows are outside full carrier range"
            )

        p_gen = p_gen_full[
            query_rows_for_alignment
        ]

        p_ver = p_ver_full[
            query_rows_for_alignment
        ]

        print(
            "aligned full p_gen/p_ver to query_rows:",
            p_gen_full.shape,
            "->",
            p_gen.shape,
        )

    else:

        raise ValueError(
            "Cannot align p_gen/p_ver with query_U_cycle: "
            f"p_gen={p_gen_full.shape}, "
            f"U={U.shape}, "
            f"query_rows="
            f"{None if query_rows_for_alignment is None else query_rows_for_alignment.shape}"
        )

    if p_ver.shape != p_gen.shape:
        raise ValueError(
            "aligned p_gen/p_ver shape mismatch"
        )

    if U.shape != p_gen.shape[:2]:
        raise ValueError(
            f"U shape mismatch after query alignment: "
            f"U={U.shape}, expected={p_gen.shape[:2]}"
        )

    n_query, action_num, k = p_gen.shape

    # --------------------------------------------------------
    # 3. Recover exact training-order GT
    #
    # This happens post-seal only.
    # --------------------------------------------------------

    _, _, mask, _ = (
        c0._load_frozen_inputs("ALOI")
    )

    raw_views = c0.load_raw_views(
        "ALOI"
    )

    _, adapter = (
        b0.preprocess_with_raw_ids(
            raw_views,
            mask,
        )
    )

    dataset = get_dataset(
        data_name=D_ALOI
    )

    y_raw = np.asarray(
        dataset.label
    ).reshape(-1)

    y_training = y_raw[
        adapter.sample_ids
    ]

    # --------------------------------------------------------
    # 4. Resolve query rows
    # --------------------------------------------------------

    if query_rows_key is not None:

        query_rows = np.asarray(
            carrier[query_rows_key]
        ).reshape(-1).astype(np.int64)

        if query_rows.size != n_query:
            raise ValueError(
                "query_rows length mismatch"
            )

    else:
        # Fallback:
        # Generic carrier queries are unlabeled rows.
        # Recover labeled training positions from the frozen sparse split.

        labels, targets, _, _ = (
            c0._load_frozen_inputs(
                "ALOI"
            )
        )

        anchor_rows = adapter.row_positions(
            labels
        )

        all_rows = np.arange(
            y_training.shape[0],
            dtype=np.int64,
        )

        anchor_mask = np.zeros(
            y_training.shape[0],
            dtype=bool,
        )

        anchor_mask[
            np.asarray(anchor_rows, dtype=np.int64)
        ] = True

        query_rows = all_rows[
            ~anchor_mask
        ]

        if query_rows.size != n_query:
            raise RuntimeError(
                "Could not infer query rows: "
                f"inferred={query_rows.size}, "
                f"carrier={n_query}"
            )

        print(
            "query rows inferred from frozen sparse anchors"
        )

    y_gt = y_training[
        query_rows
    ]

    if y_gt.shape[0] != n_query:
        raise RuntimeError(
            "GT/query alignment failure"
        )

    # --------------------------------------------------------
    # 5. Cross-view consensus
    # --------------------------------------------------------

    y_gen = np.argmax(
        p_gen,
        axis=-1,
    )

    y_ver = np.argmax(
        p_ver,
        axis=-1,
    )

    # [N_query, S]
    closure = (
        y_gen == y_ver
    )

    generator_correct = (
        y_gen
        == y_gt[:, None]
    )

    verifier_correct = (
        y_ver
        == y_gt[:, None]
    )

    true_consensus = (
        closure
        & generator_correct
    )

    false_consensus = (
        closure
        & (~generator_correct)
    )

    # --------------------------------------------------------
    # 6. Basic sanity:
    #
    # Frozen definition has U=0 if closure=0.
    # --------------------------------------------------------

    max_u_when_open = (
        float(U[~closure].max())
        if np.any(~closure)
        else 0.0
    )

    positive_u = U > 0

    positive_count = int(
        positive_u.sum()
    )

    total_pairs = int(
        U.size
    )

    utility_mass = float(
        U.sum()
    )

    if utility_mass <= 0:
        raise RuntimeError(
            "zero total U_cycle mass"
        )

    # --------------------------------------------------------
    # 7. Primary utility-weighted correctness
    # --------------------------------------------------------

    true_mass = float(
        (
            U
            * true_consensus.astype(np.float64)
        ).sum()
    )

    false_mass = float(
        (
            U
            * false_consensus.astype(np.float64)
        ).sum()
    )

    weighted_true_ratio = safe_ratio(
        true_mass,
        utility_mass,
    )

    hfcr = safe_ratio(
        false_mass,
        utility_mass,
    )

    # They should almost sum to 1 because
    # non-closed pairs carry zero utility.
    weighted_partition_sum = (
        weighted_true_ratio
        + hfcr
    )

    # --------------------------------------------------------
    # 8. U quantile diagnostics
    # --------------------------------------------------------

    positive_values = U[
        positive_u
    ]

    quantiles = {}

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

        high = (
            positive_u
            & (U >= threshold)
        )

        count = int(
            high.sum()
        )

        if count == 0:
            continue

        high_true = int(
            (
                high
                & true_consensus
            ).sum()
        )

        high_false = int(
            (
                high
                & false_consensus
            ).sum()
        )

        high_correct_rate = safe_ratio(
            high_true,
            high_true + high_false,
        )

        high_false_rate = safe_ratio(
            high_false,
            high_true + high_false,
        )

        high_mass = float(
            U[high].sum()
        )

        high_false_mass = float(
            U[
                high
                & false_consensus
            ].sum()
        )

        high_weighted_false_ratio = (
            safe_ratio(
                high_false_mass,
                high_mass,
            )
        )

        quantiles[
            f"top_{int((1-q)*100):02d}pct"
        ] = {
            "threshold":
                threshold,

            "pair_count":
                count,

            "true_consensus_count":
                high_true,

            "false_consensus_count":
                high_false,

            "consensus_correct_rate":
                high_correct_rate,

            "false_consensus_rate":
                high_false_rate,

            "utility_mass":
                high_mass,

            "utility_weighted_false_ratio":
                high_weighted_false_ratio,
        }

    # --------------------------------------------------------
    # 9. Utility bins:
    # Does correctness monotonically improve with U?
    # --------------------------------------------------------

    bin_edges = np.quantile(
        positive_values,
        np.linspace(
            0.0,
            1.0,
            11,
        ),
    )

    bins = []

    for b in range(10):

        lo = float(
            bin_edges[b]
        )

        hi = float(
            bin_edges[b + 1]
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

        selected_count = int(
            selected.sum()
        )

        if selected_count == 0:
            bins.append(
                {
                    "bin":
                        b + 1,
                    "u_min":
                        lo,
                    "u_max":
                        hi,
                    "count":
                        0,
                    "correct_rate":
                        None,
                    "false_rate":
                        None,
                }
            )
            continue

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

        bins.append(
            {
                "bin":
                    b + 1,

                "u_min":
                    lo,

                "u_max":
                    hi,

                "count":
                    selected_count,

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
            }
        )

    formal_metrics = json.loads(
        metrics_path.read_text()
    )

    method_delta = {
        key: float(value)
        for key, value
        in formal_metrics[
            "delta_OURS_minus_BASE"
        ].items()
    }

    result = {
        "experiment":
            "P3-D0-B0_WHETHER_diagnostic",

        "dataset":
            "ALOI",

        "training_seed":
            seed,

        "DIAGNOSTIC_ONLY":
            True,

        "GT_loaded_after_formal_seal":
            True,

        "carrier_shapes": {
            "N_query":
                int(n_query),

            "actions":
                int(action_num),

            "classes":
                int(k),
        },

        "utility": {
            "total_pairs":
                total_pairs,

            "positive_u_pairs":
                positive_count,

            "positive_u_fraction":
                safe_ratio(
                    positive_count,
                    total_pairs,
                ),

            "total_utility_mass":
                utility_mass,

            "max_u_when_closure_false":
                max_u_when_open,
        },

        "consensus": {
            "closure_rate":
                float(
                    closure.mean()
                ),

            "generator_correct_rate":
                float(
                    generator_correct.mean()
                ),

            "verifier_correct_rate":
                float(
                    verifier_correct.mean()
                ),

            "true_consensus_rate_all_pairs":
                float(
                    true_consensus.mean()
                ),

            "false_consensus_rate_all_pairs":
                float(
                    false_consensus.mean()
                ),
        },

        "primary_WHETHER_metrics": {
            "utility_weighted_true_consensus_ratio":
                weighted_true_ratio,

            "HFCR_utility_weighted_false_consensus_ratio":
                hfcr,

            "weighted_partition_sum":
                weighted_partition_sum,
        },

        "high_utility_quantiles":
            quantiles,

        "utility_decile_diagnostics":
            bins,

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
        / "whether_diagnostic.json"
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
        f"ALOI SEED{seed} WHETHER DIAGNOSTIC"
    )
    print("=" * 96)

    print(
        "N_query =",
        n_query,
    )

    print(
        "actions =",
        action_num,
    )

    print(
        "positive U fraction =",
        result["utility"][
            "positive_u_fraction"
        ],
    )

    print(
        "closure rate =",
        result["consensus"][
            "closure_rate"
        ],
    )

    print(
        "utility-weighted TRUE consensus =",
        weighted_true_ratio,
    )

    print(
        "HFCR =",
        hfcr,
    )

    print(
        "TRUE + FALSE utility partition =",
        weighted_partition_sum,
    )

    print()
    print(
        "HIGH-U QUANTILES:"
    )

    for name, row in quantiles.items():
        print(
            name,
            "threshold=",
            row["threshold"],
            "correct_rate=",
            row["consensus_correct_rate"],
            "false_rate=",
            row["false_consensus_rate"],
            "weighted_false_ratio=",
            row[
                "utility_weighted_false_ratio"
            ],
        )

    print()
    print(
        "FORMAL METHOD DELTA:"
    )

    print(
        json.dumps(
            method_delta,
            indent=2,
        )
    )

    print()
    print(
        "P3_D0_B0_WHETHER_DIAGNOSTIC_PASS"
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=(20, 30),
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

    analyze_seed(
        args.training_seed
    )


if __name__ == "__main__":
    main()
