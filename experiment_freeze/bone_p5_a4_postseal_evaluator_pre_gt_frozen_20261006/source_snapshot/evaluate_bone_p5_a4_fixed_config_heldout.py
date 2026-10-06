"""
P5-A4 post-seal evaluator for the final fixed-configuration
held-out-seed validation.

Full GT may be read only after the P5-A3 global pre-GT seal
has been verified.

No training, parameter update, seed selection, model selection,
or hyperparameter selection occurs here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol
    as protocol,
)

from experiments.utility_relation import (
    evaluate_bone_p3_c0_cross_dataset_formal
    as metric_eval,
)


ROOT = Path(
    "/root/autodl-tmp/BONE"
)

PRE_ROOT = (
    ROOT
    / "outputs/bone_p5/"
      "p5_a2_fixed_config_heldout_pre_gt"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs/bone_p5/"
      "p5_a4_fixed_config_heldout_postseal"
)

MSRC_GT_ARTIFACT = (
    ROOT
    / "outputs/bone_p1/"
      "msrc_snr2p5_exact2of4_seed20/"
      "bone_p1_msrc_corruption.npz"
)

DATASETS = (
    "MSRC",
    "BDGP",
    "HandWrite",
    "Caltech-5V",
    "MNIST-USPS",
    "COIL20",
    "ALOI",
    "CCV",
)

SEEDS = (
    149,
    163,
    179,
)

ARMS = (
    "BASE",
    "OURS",
)


def slug(
    dataset: str,
) -> str:

    return (
        dataset.lower()
        .replace("-", "_")
    )


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def verify_manifest(
    path: Path,
) -> None:

    for line in (
        path.read_text()
        .splitlines()
    ):

        if not line.strip():
            continue

        wanted, raw = (
            line.split(
                None,
                1,
            )
        )

        target = Path(
            raw.strip()
        )

        if not target.is_file():
            raise RuntimeError(
                f"MANIFEST_MISSING:{target}"
            )

        got = sha256_file(
            target
        )

        if got != wanted:
            raise RuntimeError(
                f"MANIFEST_HASH_FAILURE:{target}"
            )


def verify_all_pre_gt() -> dict[str, Any]:

    verify_manifest(
        PRE_ROOT
        / "GLOBAL_PRE_GT_MANIFEST.sha256"
    )

    verify_manifest(
        PRE_ROOT
        / "GLOBAL_PRE_GT_SEAL.sha256"
    )

    global_seal = json.loads(
        (
            PRE_ROOT
            / "GLOBAL_PRE_GT_SEAL.json"
        ).read_text()
    )

    if global_seal[
        "training_seeds"
    ] != [
        149,
        163,
        179,
    ]:
        raise RuntimeError(
            "GLOBAL_SEED_FAILURE"
        )

    if tuple(
        global_seal[
            "datasets"
        ]
    ) != DATASETS:
        raise RuntimeError(
            "GLOBAL_DATASET_FAILURE"
        )

    true_fields = (
        "paper_clean_held_out_confirmatory",
        "all_24_cells_complete",
        "all_48_predictions_frozen",
        "all_carriers_fresh",
        "all_carriers_epoch100",
    )

    for key in true_fields:

        if global_seal.get(
            key
        ) is not True:

            raise RuntimeError(
                f"GLOBAL_TRUE_CONTRACT_FAILURE:{key}"
            )

    false_fields = (
        "hyperparameter_search",
        "parameter_change_after_preregistration",
        "seed_replacement",
        "dataset_dropping",
        "GT_read_before_global_seal",
        "metrics_computed_before_global_seal",
        "GT_used_for_training",
        "GT_used_for_selection",
    )

    for key in false_fields:

        if global_seal.get(
            key
        ) is not False:

            raise RuntimeError(
                f"GLOBAL_FALSE_CONTRACT_FAILURE:{key}"
            )

    if (
        global_seal[
            "dataset_seed_cell_count"
        ] != 24
        or global_seal[
            "arm_cell_count"
        ] != 48
    ):
        raise RuntimeError(
            "GLOBAL_COUNT_FAILURE"
        )

    for dataset in DATASETS:

        for seed in SEEDS:

            root = (
                PRE_ROOT
                / slug(dataset)
                / f"seed{seed}"
            )

            seal = json.loads(
                (
                    root
                    / "PRE_GT_SEAL.json"
                ).read_text()
            )

            audit = json.loads(
                (
                    root
                    / "PRE_GT_AUDIT.json"
                ).read_text()
            )

            complete = json.loads(
                (
                    root
                    / "CELL_COMPLETE.json"
                ).read_text()
            )

            if (
                complete[
                    "PRE_GT_COMPLETE"
                ]
                is not True
                or complete[
                    "GT_READ"
                ]
                is not False
            ):
                raise RuntimeError(
                    f"CELL_COMPLETE_FAILURE:{dataset}:{seed}"
                )

            if (
                seal[
                    "BASE_prediction_frozen"
                ]
                is not True
                or seal[
                    "OURS_prediction_frozen"
                ]
                is not True
                or seal[
                    "GT_loaded_before_seal"
                ]
                is not False
            ):
                raise RuntimeError(
                    f"CELL_SEAL_FAILURE:{dataset}:{seed}"
                )

            for rel, wanted in (
                seal[
                    "artifacts"
                ].items()
            ):

                path = (
                    root
                    / rel
                )

                if (
                    not path.is_file()
                    or sha256_file(path)
                    != wanted
                ):
                    raise RuntimeError(
                        f"CELL_ARTIFACT_FAILURE:{path}"
                    )

            required_false = (
                "GT_loaded_before_seal",
                "full_gt_used_for_training",
                "full_gt_used_for_selection",
                "full_gt_used_for_hyperparameter_selection",
                "ACC_NMI_ARI_computed",
                "hyperparameter_search",
            )

            for key in required_false:

                if audit.get(
                    key
                ) is not False:

                    raise RuntimeError(
                        f"CELL_FIREWALL_FAILURE:"
                        f"{dataset}:{seed}:{key}"
                    )

            if audit[
                "BASE_OURS_initial_state_parity"
            ] is not True:

                raise RuntimeError(
                    "INITIAL_STATE_PARITY_FAILURE"
                )

            if audit[
                "BASE_OURS_native_order_parity"
            ] is not True:

                raise RuntimeError(
                    "NATIVE_ORDER_PARITY_FAILURE"
                )

    return global_seal


def load_predictions() -> dict:

    result = {}

    for dataset in DATASETS:

        for seed in SEEDS:

            root = (
                PRE_ROOT
                / slug(dataset)
                / f"seed{seed}"
            )

            cell = {}

            reference_ids = None

            for arm in ARMS:

                path = (
                    root
                    / arm
                    / "pre_gt_predictions.npz"
                )

                with np.load(
                    path,
                    allow_pickle=False,
                ) as z:

                    ids = np.asarray(
                        z["sample_ids"],
                        dtype=np.int64,
                    ).reshape(-1)

                    pred = np.asarray(
                        z["predictions"],
                        dtype=np.int64,
                    ).reshape(-1)

                if (
                    ids.shape
                    != pred.shape
                ):
                    raise RuntimeError(
                        f"PREDICTION_SHAPE_FAILURE:"
                        f"{dataset}:{seed}:{arm}"
                    )

                if reference_ids is None:

                    reference_ids = (
                        ids.copy()
                    )

                elif not np.array_equal(
                    reference_ids,
                    ids,
                ):

                    raise RuntimeError(
                        f"BASE_OURS_ID_MISMATCH:"
                        f"{dataset}:{seed}"
                    )

                cell[arm] = pred

            cell[
                "sample_ids"
            ] = reference_ids

            result[
                (
                    dataset,
                    seed,
                )
            ] = cell

    return result


def load_gt_after_global_seal(
    dataset: str,
    sample_ids: np.ndarray,
) -> np.ndarray:
    """
    Sole full-GT access path.
    This function must not be called until every pre-GT seal passes.
    """

    if dataset == "MSRC":

        if not MSRC_GT_ARTIFACT.is_file():
            raise RuntimeError(
                "MSRC_GT_ARTIFACT_MISSING"
            )

        with np.load(
            MSRC_GT_ARTIFACT,
            allow_pickle=False,
        ) as z:

            labels = np.asarray(
                z["labels"],
                dtype=np.int64,
            ).reshape(-1)

            row_ids = np.asarray(
                z["row_ids"],
                dtype=np.int64,
            ).reshape(-1)

        if not np.array_equal(
            row_ids,
            np.arange(
                labels.size,
                dtype=np.int64,
            ),
        ):
            raise RuntimeError(
                "MSRC_GT_ROW_ALIGNMENT_FAILURE"
            )

    else:

        gt_path = (
            protocol.DATA_ROOT
            / f"{dataset}.mat"
        )

        loaded = scipy.io.loadmat(
            gt_path,
            variable_names=(
                "Y",
            ),
        )

        if "Y" not in loaded:
            raise RuntimeError(
                f"MISSING_GT_Y:{dataset}"
            )

        labels = np.asarray(
            loaded["Y"],
            dtype=np.int64,
        ).reshape(-1)

    if not np.array_equal(
        np.sort(
            sample_ids
        ),
        np.arange(
            labels.size,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"SAMPLE_ID_GT_ALIGNMENT_FAILURE:{dataset}"
        )

    return np.ascontiguousarray(
        labels[
            sample_ids
        ],
        dtype=np.int64,
    )


def metrics3(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    raw = (
        metric_eval.compute_metrics(
            y_true,
            y_pred,
        )
    )

    result = {
        "ACC":
            float(
                raw["ACC"]
            ),

        "NMI":
            float(
                raw["NMI"]
            ),

        "ARI":
            float(
                raw["ARI"]
            ),
    }

    result["Mean3"] = float(
        (
            result["ACC"]
            + result["NMI"]
            + result["ARI"]
        )
        / 3.0
    )

    return result


def evaluate(
    output_root: Path,
) -> dict[str, Any]:

    output_root = Path(
        output_root
    )

    if output_root.exists():
        raise RuntimeError(
            "OUTPUT_EXISTS_FAIL_CLOSED"
        )

    # ----------------------------------------------------------
    # NO GT MAY BE ACCESSED ABOVE THIS POINT.
    # ----------------------------------------------------------

    global_seal = (
        verify_all_pre_gt()
    )

    predictions = (
        load_predictions()
    )

    print(
        "ALL_PRE_GT_SEALS_VERIFIED_BEFORE_FIRST_GT_READ=PASS",
        flush=True,
    )

    print(
        "ALL_48_PREDICTIONS_FROZEN_BEFORE_GT=true",
        flush=True,
    )

    print(
        "GT_HAS_NOT_BEEN_READ_YET=true",
        flush=True,
    )

    # ----------------------------------------------------------
    # FIRST AUTHORIZED GT ACCESS BELOW.
    # ----------------------------------------------------------

    rows = []

    for dataset in DATASETS:

        for seed in SEEDS:

            cell = predictions[
                (
                    dataset,
                    seed,
                )
            ]

            labels = (
                load_gt_after_global_seal(
                    dataset,
                    cell[
                        "sample_ids"
                    ],
                )
            )

            print(
                f"GT_LOAD_AFTER_GLOBAL_SEAL=PASS "
                f"dataset={dataset} seed={seed}",
                flush=True,
            )

            arm_metrics = {}

            for arm in ARMS:

                arm_metrics[arm] = (
                    metrics3(
                        labels,
                        cell[arm],
                    )
                )

            row = {
                "dataset":
                    dataset,

                "training_seed":
                    seed,
            }

            for arm in ARMS:

                for metric in (
                    "ACC",
                    "NMI",
                    "ARI",
                    "Mean3",
                ):

                    row[
                        f"{arm}_{metric}"
                    ] = (
                        arm_metrics[
                            arm
                        ][
                            metric
                        ]
                    )

            for metric in (
                "ACC",
                "NMI",
                "ARI",
                "Mean3",
            ):

                row[
                    f"Delta{metric}"
                ] = float(
                    arm_metrics[
                        "OURS"
                    ][metric]
                    -
                    arm_metrics[
                        "BASE"
                    ][metric]
                )

            rows.append(row)

    if len(rows) != 24:
        raise RuntimeError(
            "EXPECTED_24_RESULT_ROWS"
        )

    summaries = []

    for dataset in DATASETS:

        group = [
            row
            for row in rows
            if row[
                "dataset"
            ] == dataset
        ]

        if len(group) != 3:
            raise RuntimeError(
                "THREE_SEED_GROUP_FAILURE"
            )

        summary = {
            "dataset":
                dataset,
        }

        fields = (
            "BASE_ACC",
            "BASE_NMI",
            "BASE_ARI",
            "BASE_Mean3",
            "OURS_ACC",
            "OURS_NMI",
            "OURS_ARI",
            "OURS_Mean3",
            "DeltaACC",
            "DeltaNMI",
            "DeltaARI",
            "DeltaMean3",
        )

        for field in fields:

            values = np.asarray(
                [
                    row[field]
                    for row in group
                ],
                dtype=np.float64,
            )

            summary[
                f"{field}_mean"
            ] = float(
                values.mean()
            )

            summary[
                f"{field}_std"
            ] = float(
                values.std(
                    ddof=0
                )
            )

        summaries.append(
            summary
        )

    if len(
        summaries
    ) != 8:
        raise RuntimeError(
            "EXPECTED_8_DATASET_SUMMARIES"
        )

    macro = {}

    for metric in (
        "ACC",
        "NMI",
        "ARI",
        "Mean3",
    ):

        values = np.asarray(
            [
                row[
                    f"Delta{metric}_mean"
                ]
                for row in summaries
            ],
            dtype=np.float64,
        )

        macro[
            f"Delta{metric}"
        ] = float(
            values.mean()
        )

    positive_dataset_count = int(
        sum(
            row[
                "DeltaMean3_mean"
            ] > 0.0
            for row in summaries
        )
    )

    nonnegative_dataset_count = int(
        sum(
            row[
                "DeltaMean3_mean"
            ] >= 0.0
            for row in summaries
        )
    )

    positive_cell_count = int(
        sum(
            row[
                "DeltaMean3"
            ] > 0.0
            for row in rows
        )
    )

    nonnegative_cell_count = int(
        sum(
            row[
                "DeltaMean3"
            ] >= 0.0
            for row in rows
        )
    )

    strong = bool(
        macro[
            "DeltaMean3"
        ] > 0.0
        and positive_dataset_count
        >= 6
        and macro[
            "DeltaACC"
        ] > 0.0
        and macro[
            "DeltaNMI"
        ] > 0.0
        and macro[
            "DeltaARI"
        ] > 0.0
    )

    if strong:

        decision = (
            "STRONG_REPLICATION"
        )

    elif (
        macro[
            "DeltaMean3"
        ] > 0.0
    ):

        decision = (
            "PARTIAL_REPLICATION"
        )

    else:

        decision = (
            "FAILED_REPLICATION"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    per_seed_csv = (
        output_root
        / "P5_A4_PER_SEED_RESULTS.csv"
    )

    with per_seed_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_csv = (
        output_root
        / "P5_A4_THREE_SEED_SUMMARY.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    result = {
        "stage":
            "P5-A4",

        "role":
            "FINAL_FIXED_CONFIG_HELD_OUT_SEED_VALIDATION",

        "paper_clean_held_out_validation":
            True,

        "training_seeds": [
            149,
            163,
            179,
        ],

        "dataset_count":
            8,

        "seed_count":
            3,

        "dataset_seed_cell_count":
            24,

        "GT_loaded_only_after_global_pre_gt_seal":
            True,

        "GT_used_for_training":
            False,

        "GT_used_for_hyperparameter_selection":
            False,

        "GT_used_for_model_selection":
            False,

        "parameter_changed_after_preregistration":
            False,

        "seed_replaced_after_preregistration":
            False,

        "per_seed":
            rows,

        "three_seed_summary":
            summaries,

        "macro_delta": {
            **macro,

            "positive_dataset_count":
                positive_dataset_count,

            "nonnegative_dataset_count":
                nonnegative_dataset_count,

            "positive_seed_cell_count":
                positive_cell_count,

            "nonnegative_seed_cell_count":
                nonnegative_cell_count,
        },

        "replication_decision":
            decision,

        "strong_replication_rule":
            (
                "macro DeltaMean3 > 0 AND "
                "positive dataset count >= 6/8 AND "
                "macro DeltaACC > 0 AND "
                "macro DeltaNMI > 0 AND "
                "macro DeltaARI > 0"
            ),

        "global_pre_gt_seal_sha256":
            sha256_file(
                PRE_ROOT
                / "GLOBAL_PRE_GT_SEAL.json"
            ),
    }

    (
        output_root
        / "P5_A4_FINAL_RESULTS.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.execute:

        raise SystemExit(
            "refusing P5-A4 GT read without --execute"
        )

    result = evaluate(
        args.output_root
    )

    print()
    print("=" * 145)
    print(
        "P5-A4 HELD-OUT PER-SEED BASE vs OURS"
    )
    print("=" * 145)

    print(
        f"{'Dataset':<12} "
        f"{'Seed':>4} "
        f"{'BASE_ACC':>9} "
        f"{'OURS_ACC':>9} "
        f"{'D_ACC':>9} "
        f"{'BASE_NMI':>9} "
        f"{'OURS_NMI':>9} "
        f"{'D_NMI':>9} "
        f"{'BASE_ARI':>9} "
        f"{'OURS_ARI':>9} "
        f"{'D_ARI':>9} "
        f"{'D_M3':>9}"
    )

    print("-" * 145)

    for row in result[
        "per_seed"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"{row['training_seed']:>4d} "
            f"{row['BASE_ACC']:>9.6f} "
            f"{row['OURS_ACC']:>9.6f} "
            f"{row['DeltaACC']:>+9.6f} "
            f"{row['BASE_NMI']:>9.6f} "
            f"{row['OURS_NMI']:>9.6f} "
            f"{row['DeltaNMI']:>+9.6f} "
            f"{row['BASE_ARI']:>9.6f} "
            f"{row['OURS_ARI']:>9.6f} "
            f"{row['DeltaARI']:>+9.6f} "
            f"{row['DeltaMean3']:>+9.6f}"
        )

    print()
    print("=" * 130)
    print(
        "P5-A4 THREE-SEED DATASET SUMMARY"
    )
    print("=" * 130)

    print(
        f"{'Dataset':<12} "
        f"{'BASE_ACC':>10} "
        f"{'OURS_ACC':>10} "
        f"{'D_ACC':>10} "
        f"{'D_NMI':>10} "
        f"{'D_ARI':>10} "
        f"{'D_Mean3':>11}"
    )

    print("-" * 130)

    for row in result[
        "three_seed_summary"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"{row['BASE_ACC_mean']:>10.6f} "
            f"{row['OURS_ACC_mean']:>10.6f} "
            f"{row['DeltaACC_mean']:>+10.6f} "
            f"{row['DeltaNMI_mean']:>+10.6f} "
            f"{row['DeltaARI_mean']:>+10.6f} "
            f"{row['DeltaMean3_mean']:>+11.6f}"
        )

    macro = result[
        "macro_delta"
    ]

    print()
    print("=" * 100)
    print(
        "P5-A4 FINAL HELD-OUT MACRO RESULT"
    )
    print("=" * 100)

    print(
        "macro_DeltaACC =",
        macro[
            "DeltaACC"
        ],
    )

    print(
        "macro_DeltaNMI =",
        macro[
            "DeltaNMI"
        ],
    )

    print(
        "macro_DeltaARI =",
        macro[
            "DeltaARI"
        ],
    )

    print(
        "macro_DeltaMean3 =",
        macro[
            "DeltaMean3"
        ],
    )

    print(
        "positive_dataset_count =",
        macro[
            "positive_dataset_count"
        ],
        "/ 8",
    )

    print(
        "nonnegative_dataset_count =",
        macro[
            "nonnegative_dataset_count"
        ],
        "/ 8",
    )

    print(
        "positive_seed_cell_count =",
        macro[
            "positive_seed_cell_count"
        ],
        "/ 24",
    )

    print(
        "nonnegative_seed_cell_count =",
        macro[
            "nonnegative_seed_cell_count"
        ],
        "/ 24",
    )

    print()
    print(
        "P5_A4_REPLICATION_DECISION =",
        result[
            "replication_decision"
        ],
    )

    print(
        "GT_USED_FOR_TRAINING=false"
    )

    print(
        "GT_USED_FOR_HYPERPARAMETER_SELECTION=false"
    )

    print(
        "GT_USED_FOR_MODEL_SELECTION=false"
    )


if __name__ == "__main__":
    main()
