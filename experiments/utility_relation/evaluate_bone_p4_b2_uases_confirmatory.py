"""
P4-B2 UASES unseen-seed confirmatory post-seal evaluator.

The evaluator:
1. verifies the global pre-GT seal;
2. verifies all nine cell-level seals;
3. only then loads GT;
4. evaluates UASES-selected vs frozen fixed-H predictions;
5. applies the preregistered four-gate confirmatory decision.

No training or parameter update is possible here.
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
    as c0_eval,
)

from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)


ROOT = Path(
    "/root/autodl-tmp/BONE"
)

PRE_ROOT = (
    ROOT
    / "outputs/bone_p4/"
      "p4_b2_uases_confirmatory"
)

DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "outputs/bone_p4/"
      "p4_b2_uases_confirmatory_postseal"
)

DATASETS = (
    "MSRC",
    "HandWrite",
    "MNIST-USPS",
)

SEEDS = (
    113,
    127,
    139,
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

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def verify_manifest(
    manifest_path: Path,
) -> None:

    for line in (
        manifest_path
        .read_text()
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

        path = Path(
            raw.strip()
        )

        if not path.is_file():
            raise RuntimeError(
                f"MANIFEST_FILE_MISSING:{path}"
            )

        got = sha256_file(
            path
        )

        if got != wanted:
            raise RuntimeError(
                f"MANIFEST_HASH_FAILURE:{path}"
            )


def verify_all_pre_gt_seals() -> dict[str, Any]:

    global_manifest = (
        PRE_ROOT
        / "GLOBAL_PRE_GT_MANIFEST.sha256"
    )

    global_seal_path = (
        PRE_ROOT
        / "GLOBAL_PRE_GT_SEAL.json"
    )

    global_seal_sha = (
        PRE_ROOT
        / "GLOBAL_PRE_GT_SEAL.sha256"
    )

    verify_manifest(
        global_manifest
    )

    verify_manifest(
        global_seal_sha
    )

    global_seal = json.loads(
        global_seal_path.read_text()
    )

    required_true = (
        "confirmatory",
        "paper_formal_result",
        "all_9_cells_complete",
        "all_18_selected_h_frozen",
        "all_carriers_fresh",
        "all_carriers_epoch100",
        "historical_schedule_parity_pass",
    )

    for key in required_true:

        if global_seal.get(
            key
        ) is not True:

            raise RuntimeError(
                f"GLOBAL_TRUE_CONTRACT_FAILURE:{key}"
            )

    required_false = (
        "GT_read_before_global_seal",
        "metrics_computed_before_global_seal",
        "GT_used_for_training",
        "GT_used_for_stopping",
        "GT_used_for_selected_h",
        "UASES_formula_changed",
        "semantic_loss_changed",
        "T_rel_changed",
        "U_cycle_changed",
        "semantic_lr_changed",
        "semantic_exposure_changed",
        "confirmatory_gate_changed",
        "registered_seed_replaced",
    )

    for key in required_false:

        if global_seal.get(
            key
        ) is not False:

            raise RuntimeError(
                f"GLOBAL_FALSE_CONTRACT_FAILURE:{key}"
            )

    if global_seal[
        "training_seeds"
    ] != [
        113,
        127,
        139,
    ]:
        raise RuntimeError(
            "GLOBAL_SEED_FAILURE"
        )

    if global_seal[
        "datasets"
    ] != [
        "MSRC",
        "HandWrite",
        "MNIST-USPS",
    ]:
        raise RuntimeError(
            "GLOBAL_DATASET_FAILURE"
        )

    if (
        global_seal[
            "dataset_seed_cell_count"
        ]
        != 9
        or global_seal[
            "arm_cell_count"
        ]
        != 18
    ):
        raise RuntimeError(
            "GLOBAL_CELL_COUNT_FAILURE"
        )

    for dataset in DATASETS:

        for seed in SEEDS:

            root = (
                PRE_ROOT
                / slug(dataset)
                / f"seed{seed}"
            )

            seal_path = (
                root
                / "PRE_GT_SEAL.json"
            )

            audit_path = (
                root
                / "PRE_GT_AUDIT.json"
            )

            complete_path = (
                root
                / "CELL_COMPLETE.json"
            )

            trajectory_path = (
                root
                / "UASES_CONFIRMATORY_TRAJECTORY.json"
            )

            for path in (
                seal_path,
                audit_path,
                complete_path,
                trajectory_path,
            ):

                if not path.is_file():
                    raise RuntimeError(
                        f"MISSING_CELL_FILE:{path}"
                    )

            seal = json.loads(
                seal_path.read_text()
            )

            audit = json.loads(
                audit_path.read_text()
            )

            complete = json.loads(
                complete_path.read_text()
            )

            trajectory = json.loads(
                trajectory_path.read_text()
            )

            if (
                complete.get(
                    "PRE_GT_COMPLETE"
                )
                is not True
                or complete.get(
                    "GT_READ"
                )
                is not False
            ):
                raise RuntimeError(
                    f"CELL_COMPLETE_FAILURE:"
                    f"{dataset}:{seed}"
                )

            if (
                seal.get(
                    "selected_h_frozen"
                )
                is not True
                or seal.get(
                    "GT_loaded_before_seal"
                )
                is not False
            ):
                raise RuntimeError(
                    f"CELL_SEAL_FAILURE:"
                    f"{dataset}:{seed}"
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

                if sha256_file(
                    path
                ) != wanted:

                    raise RuntimeError(
                        f"CELL_HASH_FAILURE:{path}"
                    )

            false_fields = (
                "GT_loaded_before_seal",
                "full_gt_used_for_training",
                "full_gt_used_for_stopping",
                "full_gt_used_for_selection",
                "ACC_NMI_ARI_computed",
            )

            for key in false_fields:

                if audit.get(
                    key
                ) is not False:

                    raise RuntimeError(
                        f"CELL_GT_FIREWALL_FAILURE:"
                        f"{dataset}:{seed}:{key}"
                    )

            if audit[
                "fresh_carrier_trained"
            ] is not True:

                raise RuntimeError(
                    "FRESH_CARRIER_FAILURE"
                )

            if audit[
                "carrier_reused"
            ] is not False:

                raise RuntimeError(
                    "CARRIER_REUSE_FAILURE"
                )

            if int(
                audit[
                    "carrier_epochs"
                ]
            ) != 100:

                raise RuntimeError(
                    "CARRIER_EPOCH_FAILURE"
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

            if audit[
                "same_U0_for_BASE_and_OURS"
            ] is not True:

                raise RuntimeError(
                    "U0_PARITY_FAILURE"
                )

            if audit[
                "U0_recomputed_during_refinement"
            ] is not False:

                raise RuntimeError(
                    "U0_RECOMPUTE_FAILURE"
                )

            if trajectory[
                "GT_loaded"
            ] is not False:

                raise RuntimeError(
                    "TRAJECTORY_GT_FAILURE"
                )

    return global_seal


def load_predictions_pre_gt() -> dict:

    payload = {}

    for dataset in DATASETS:

        for seed in SEEDS:

            root = (
                PRE_ROOT
                / slug(dataset)
                / f"seed{seed}"
            )

            trajectory = json.loads(
                (
                    root
                    / "UASES_CONFIRMATORY_TRAJECTORY.json"
                ).read_text()
            )

            cell = {
                "fixed_h":
                    int(
                        trajectory[
                            "fixed_h"
                        ]
                    ),
            }

            ids_ref = None

            for arm in ARMS:

                selected_path = (
                    root
                    / arm
                    / "selected_pre_gt_predictions.npz"
                )

                fixed_path = (
                    root
                    / arm
                    / "fixed_h_pre_gt_predictions.npz"
                )

                with np.load(
                    selected_path,
                    allow_pickle=False,
                ) as z:

                    selected_ids = np.asarray(
                        z[
                            "sample_ids"
                        ],
                        dtype=np.int64,
                    )

                    selected_pred = np.asarray(
                        z[
                            "predictions"
                        ],
                        dtype=np.int64,
                    )

                    selected_h = int(
                        np.asarray(
                            z[
                                "selected_h"
                            ]
                        ).reshape(-1)[0]
                    )

                with np.load(
                    fixed_path,
                    allow_pickle=False,
                ) as z:

                    fixed_ids = np.asarray(
                        z[
                            "sample_ids"
                        ],
                        dtype=np.int64,
                    )

                    fixed_pred = np.asarray(
                        z[
                            "predictions"
                        ],
                        dtype=np.int64,
                    )

                    fixed_h = int(
                        np.asarray(
                            z[
                                "fixed_h"
                            ]
                        ).reshape(-1)[0]
                    )

                if not np.array_equal(
                    selected_ids,
                    fixed_ids,
                ):
                    raise RuntimeError(
                        f"SELECTED_FIXED_ID_MISMATCH:"
                        f"{dataset}:{seed}:{arm}"
                    )

                if (
                    selected_pred.shape
                    != selected_ids.shape
                    or fixed_pred.shape
                    != selected_ids.shape
                ):
                    raise RuntimeError(
                        "PREDICTION_SHAPE_FAILURE"
                    )

                if selected_h != int(
                    trajectory[
                        arm
                    ][
                        "selected_h"
                    ]
                ):
                    raise RuntimeError(
                        "SELECTED_H_ARCHIVE_FAILURE"
                    )

                if fixed_h != int(
                    trajectory[
                        "fixed_h"
                    ]
                ):
                    raise RuntimeError(
                        "FIXED_H_ARCHIVE_FAILURE"
                    )

                if ids_ref is None:

                    ids_ref = (
                        selected_ids.copy()
                    )

                elif not np.array_equal(
                    ids_ref,
                    selected_ids,
                ):
                    raise RuntimeError(
                        "BASE_OURS_SAMPLE_ID_MISMATCH"
                    )

                cell[
                    arm
                ] = {
                    "selected_h":
                        selected_h,

                    "selected_predictions":
                        selected_pred,

                    "fixed_predictions":
                        fixed_pred,
                }

            cell[
                "sample_ids"
            ] = ids_ref

            payload[
                (
                    dataset,
                    seed,
                )
            ] = cell

    return payload


def load_gt_after_seal(
    dataset: str,
    sample_ids: np.ndarray,
) -> np.ndarray:
    """
    Sole full-GT access path.
    Must only be called after all seals have been verified.
    """

    if dataset == "MSRC":

        artifact = (
            msrc_adapter
            .p3_a4_materializer
            .P1_ARTIFACT
        )

        wanted = (
            msrc_adapter
            .p3_a4_materializer
            .P1_ARTIFACT_SHA256
        )

        got = (
            msrc_adapter
            .p3_a4_materializer
            .sha256_file(
                artifact
            )
        )

        if got != wanted:
            raise RuntimeError(
                "MSRC_PARENT_HASH_FAILURE"
            )

        with np.load(
            artifact,
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
                "MSRC_GT_ROW_ID_FAILURE"
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
            f"GT_SAMPLE_ID_ALIGNMENT_FAILURE:{dataset}"
        )

    return np.ascontiguousarray(
        labels[
            sample_ids
        ],
        dtype=np.int64,
    )


def metrics3(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:

    raw = c0_eval.compute_metrics(
        labels,
        predictions,
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
            "POSTSEAL_OUTPUT_EXISTS_FAIL_CLOSED"
        )

    # ---------------------------------------------------------
    # ALL PRE-GT VERIFICATION MUST FINISH HERE.
    # ---------------------------------------------------------

    global_seal = (
        verify_all_pre_gt_seals()
    )

    frozen = (
        load_predictions_pre_gt()
    )

    print(
        "ALL_PRE_GT_SEALS_VERIFIED_BEFORE_FIRST_GT_READ=PASS",
        flush=True,
    )

    print(
        "ALL_18_SELECTED_H_FROZEN_BEFORE_GT=true",
        flush=True,
    )

    print(
        "GT_HAS_NOT_BEEN_READ_YET=true",
        flush=True,
    )

    # ---------------------------------------------------------
    # FIRST GT ACCESS BELOW THIS LINE.
    # ---------------------------------------------------------

    rows = []

    ours_base_rows = []

    gt_cache = {}

    for dataset in DATASETS:

        for seed in SEEDS:

            cell = frozen[
                (
                    dataset,
                    seed,
                )
            ]

            labels = (
                load_gt_after_seal(
                    dataset,
                    cell[
                        "sample_ids"
                    ],
                )
            )

            gt_cache[
                (
                    dataset,
                    seed,
                )
            ] = labels

            print(
                f"GT_LOAD_AFTER_GLOBAL_SEAL=PASS "
                f"dataset={dataset} "
                f"seed={seed}",
                flush=True,
            )

            selected_by_arm = {}

            for arm in ARMS:

                selected = metrics3(
                    labels,
                    cell[
                        arm
                    ][
                        "selected_predictions"
                    ],
                )

                fixed = metrics3(
                    labels,
                    cell[
                        arm
                    ][
                        "fixed_predictions"
                    ],
                )

                delta = {
                    metric:
                        float(
                            selected[
                                metric
                            ]
                            - fixed[
                                metric
                            ]
                        )
                    for metric in (
                        "ACC",
                        "NMI",
                        "ARI",
                        "Mean3",
                    )
                }

                row = {
                    "dataset":
                        dataset,

                    "training_seed":
                        seed,

                    "arm":
                        arm,

                    "fixed_h":
                        int(
                            cell[
                                "fixed_h"
                            ]
                        ),

                    "selected_h":
                        int(
                            cell[
                                arm
                            ][
                                "selected_h"
                            ]
                        ),

                    "selected_ACC":
                        selected["ACC"],

                    "selected_NMI":
                        selected["NMI"],

                    "selected_ARI":
                        selected["ARI"],

                    "selected_Mean3":
                        selected[
                            "Mean3"
                        ],

                    "fixed_ACC":
                        fixed["ACC"],

                    "fixed_NMI":
                        fixed["NMI"],

                    "fixed_ARI":
                        fixed["ARI"],

                    "fixed_Mean3":
                        fixed[
                            "Mean3"
                        ],

                    "DeltaACC":
                        delta["ACC"],

                    "DeltaNMI":
                        delta["NMI"],

                    "DeltaARI":
                        delta["ARI"],

                    "DeltaMean3":
                        delta[
                            "Mean3"
                        ],
                }

                rows.append(
                    row
                )

                selected_by_arm[
                    arm
                ] = selected

            ours_base_rows.append(
                {
                    "dataset":
                        dataset,

                    "training_seed":
                        seed,

                    "DeltaACC_OURS_minus_BASE":
                        float(
                            selected_by_arm[
                                "OURS"
                            ][
                                "ACC"
                            ]
                            -
                            selected_by_arm[
                                "BASE"
                            ][
                                "ACC"
                            ]
                        ),

                    "DeltaNMI_OURS_minus_BASE":
                        float(
                            selected_by_arm[
                                "OURS"
                            ][
                                "NMI"
                            ]
                            -
                            selected_by_arm[
                                "BASE"
                            ][
                                "NMI"
                            ]
                        ),

                    "DeltaARI_OURS_minus_BASE":
                        float(
                            selected_by_arm[
                                "OURS"
                            ][
                                "ARI"
                            ]
                            -
                            selected_by_arm[
                                "BASE"
                            ][
                                "ARI"
                            ]
                        ),

                    "DeltaMean3_OURS_minus_BASE":
                        float(
                            selected_by_arm[
                                "OURS"
                            ][
                                "Mean3"
                            ]
                            -
                            selected_by_arm[
                                "BASE"
                            ][
                                "Mean3"
                            ]
                        ),
                }
            )

    if len(rows) != 18:
        raise RuntimeError(
            "EXPECTED_18_ARM_CELL_RESULTS"
        )

    if len(
        ours_base_rows
    ) != 9:
        raise RuntimeError(
            "EXPECTED_9_OURS_BASE_RESULTS"
        )

    # ---------------------------------------------------------
    # Dataset-arm three-seed aggregation.
    # std uses population convention ddof=0 for reporting only.
    # It does NOT enter any confirmatory selection decision.
    # ---------------------------------------------------------

    group_rows = []

    for dataset in DATASETS:

        for arm in ARMS:

            group = [
                row
                for row
                in rows
                if (
                    row[
                        "dataset"
                    ] == dataset
                    and row[
                        "arm"
                    ] == arm
                )
            ]

            if len(group) != 3:
                raise RuntimeError(
                    "THREE_SEED_GROUP_FAILURE"
                )

            record = {
                "dataset":
                    dataset,

                "arm":
                    arm,
            }

            for field in (
                "selected_ACC",
                "selected_NMI",
                "selected_ARI",
                "selected_Mean3",
                "fixed_ACC",
                "fixed_NMI",
                "fixed_ARI",
                "fixed_Mean3",
                "DeltaACC",
                "DeltaNMI",
                "DeltaARI",
                "DeltaMean3",
            ):

                values = np.asarray(
                    [
                        row[field]
                        for row
                        in group
                    ],
                    dtype=np.float64,
                )

                record[
                    f"{field}_mean"
                ] = float(
                    values.mean()
                )

                record[
                    f"{field}_std"
                ] = float(
                    values.std(
                        ddof=0
                    )
                )

            group_rows.append(
                record
            )

    if len(
        group_rows
    ) != 6:
        raise RuntimeError(
            "EXPECTED_6_DATASET_ARM_GROUPS"
        )

    deltas = np.asarray(
        [
            row[
                "DeltaMean3"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    dataset_arm_means = np.asarray(
        [
            row[
                "DeltaMean3_mean"
            ]
            for row
            in group_rows
        ],
        dtype=np.float64,
    )

    mean_delta = float(
        deltas.mean()
    )

    nonnegative_count = int(
        np.count_nonzero(
            deltas >= 0.0
        )
    )

    min_dataset_arm_mean = float(
        dataset_arm_means.min()
    )

    min_single = float(
        deltas.min()
    )

    gates = {
        "mean_DeltaMean3_gte_0":
            bool(
                mean_delta >= 0.0
            ),

        "nonnegative_count_gte_9_of_18":
            bool(
                nonnegative_count >= 9
            ),

        "minimum_dataset_arm_3seed_mean_DeltaMean3_gte_minus_0p01":
            bool(
                min_dataset_arm_mean
                >= -0.01
            ),

        "minimum_single_DeltaMean3_gte_minus_0p02":
            bool(
                min_single
                >= -0.02
            ),
    }

    decision = (
        "CONFIRMED"
        if all(
            gates.values()
        )
        else "NOT_CONFIRMED"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    arm_csv = (
        output_root
        / "P4_B2_18_ARM_CELL_SELECTED_VS_FIXED.csv"
    )

    with arm_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=tuple(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    group_csv = (
        output_root
        / "P4_B2_3SEED_DATASET_ARM_SUMMARY.csv"
    )

    with group_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=tuple(
                group_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            group_rows
        )

    ours_base_csv = (
        output_root
        / "P4_B2_OURS_SELECTED_MINUS_BASE_SELECTED.csv"
    )

    with ours_base_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=tuple(
                ours_base_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            ours_base_rows
        )

    result = {
        "stage":
            "P4-B2",

        "role":
            "UNSEEN_SEED_CONFIRMATORY_POSTSEAL",

        "paper_formal_result":
            True,

        "confirmatory":
            True,

        "GT_loaded_only_after_global_pre_gt_seal":
            True,

        "GT_used_for_training":
            False,

        "GT_used_for_stopping":
            False,

        "GT_used_for_selected_h":
            False,

        "selected_h_changed_after_GT":
            False,

        "training_performed_by_evaluator":
            False,

        "arm_cell_count":
            18,

        "dataset_arm_group_count":
            6,

        "per_arm_cell":
            rows,

        "dataset_arm_3seed_summary":
            group_rows,

        "ours_selected_minus_base_selected":
            ours_base_rows,

        "confirmatory_gate": {
            "mean_DeltaMean3":
                mean_delta,

            "nonnegative_DeltaMean3_count":
                nonnegative_count,

            "minimum_dataset_arm_3seed_mean_DeltaMean3":
                min_dataset_arm_mean,

            "minimum_single_DeltaMean3":
                min_single,

            "conditions":
                gates,

            "decision":
                decision,
        },

        "global_pre_gt_seal_sha256":
            sha256_file(
                PRE_ROOT
                / "GLOBAL_PRE_GT_SEAL.json"
            ),
    }

    result_path = (
        output_root
        / "P4_B2_CONFIRMATORY_RESULTS.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    gate_path = (
        output_root
        / "P4_B2_FINAL_DECISION.json"
    )

    gate_path.write_text(
        json.dumps(
            result[
                "confirmatory_gate"
            ],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def main() -> None:

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
            "refusing post-seal GT read without --execute"
        )

    result = evaluate(
        args.output_root
    )

    print()
    print("=" * 150)
    print(
        "P4-B2 18 ARM-CELL: UASES SELECTED vs FROZEN FIXED-H"
    )
    print("=" * 150)

    print(
        f"{'Dataset':<12} "
        f"{'Seed':>4} "
        f"{'Arm':<5} "
        f"{'FixH':>5} "
        f"{'SelH':>5} "
        f"{'SelACC':>9} "
        f"{'SelNMI':>9} "
        f"{'SelARI':>9} "
        f"{'SelM3':>9} "
        f"{'FixM3':>9} "
        f"{'DeltaM3':>10}"
    )

    print("-" * 150)

    for row in result[
        "per_arm_cell"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"{row['training_seed']:>4d} "
            f"{row['arm']:<5} "
            f"{row['fixed_h']:>5d} "
            f"{row['selected_h']:>5d} "
            f"{row['selected_ACC']:>9.6f} "
            f"{row['selected_NMI']:>9.6f} "
            f"{row['selected_ARI']:>9.6f} "
            f"{row['selected_Mean3']:>9.6f} "
            f"{row['fixed_Mean3']:>9.6f} "
            f"{row['DeltaMean3']:>+10.6f}"
        )

    print()
    print("=" * 120)
    print(
        "P4-B2 DATASET × ARM THREE-SEED SUMMARY"
    )
    print("=" * 120)

    print(
        f"{'Dataset':<12} "
        f"{'Arm':<5} "
        f"{'SelACC mean':>12} "
        f"{'SelNMI mean':>12} "
        f"{'SelARI mean':>12} "
        f"{'SelM3 mean':>12} "
        f"{'DeltaM3 mean':>13} "
        f"{'DeltaM3 std':>12}"
    )

    print("-" * 120)

    for row in result[
        "dataset_arm_3seed_summary"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"{row['arm']:<5} "
            f"{row['selected_ACC_mean']:>12.6f} "
            f"{row['selected_NMI_mean']:>12.6f} "
            f"{row['selected_ARI_mean']:>12.6f} "
            f"{row['selected_Mean3_mean']:>12.6f} "
            f"{row['DeltaMean3_mean']:>+13.6f} "
            f"{row['DeltaMean3_std']:>12.6f}"
        )

    print()
    print("=" * 100)
    print(
        "OURS SELECTED - BASE SELECTED"
    )
    print("=" * 100)

    for row in result[
        "ours_selected_minus_base_selected"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"seed={row['training_seed']:<3} "
            f"ACC={row['DeltaACC_OURS_minus_BASE']:+.6f} "
            f"NMI={row['DeltaNMI_OURS_minus_BASE']:+.6f} "
            f"ARI={row['DeltaARI_OURS_minus_BASE']:+.6f} "
            f"Mean3={row['DeltaMean3_OURS_minus_BASE']:+.6f}"
        )

    gate = result[
        "confirmatory_gate"
    ]

    print()
    print("=" * 100)
    print(
        "P4-B2 PREREGISTERED CONFIRMATORY GATES"
    )
    print("=" * 100)

    print(
        "mean_DeltaMean3 =",
        gate[
            "mean_DeltaMean3"
        ],
    )

    print(
        "nonnegative_DeltaMean3_count =",
        gate[
            "nonnegative_DeltaMean3_count"
        ],
        "/ 18",
    )

    print(
        "minimum_dataset_arm_3seed_mean_DeltaMean3 =",
        gate[
            "minimum_dataset_arm_3seed_mean_DeltaMean3"
        ],
    )

    print(
        "minimum_single_DeltaMean3 =",
        gate[
            "minimum_single_DeltaMean3"
        ],
    )

    for name, passed in (
        gate[
            "conditions"
        ].items()
    ):

        print(
            name,
            "=",
            "PASS"
            if passed
            else "FAIL",
        )

    print()
    print(
        "P4_B2_FINAL_DECISION =",
        gate[
            "decision"
        ],
    )

    print(
        "GT_USED_FOR_TRAINING=false"
    )

    print(
        "GT_USED_FOR_STOPPING=false"
    )

    print(
        "GT_USED_FOR_SELECTION=false"
    )

    print(
        "SELECTED_H_CHANGED_AFTER_GT=false"
    )


if __name__ == "__main__":
    main()
