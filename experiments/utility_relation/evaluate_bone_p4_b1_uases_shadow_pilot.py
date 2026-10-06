"""
P4-B1 UASES validity-pilot post-seal evaluator.

Important:
    - verifies ALL pre-GT seals before the first GT read;
    - never trains or updates a model;
    - evaluates frozen H* and frozen fixed-H;
    - full H=5..100 trajectory is used only for post-hoc oracle diagnosis;
    - oracle information cannot modify H*;
    - applies the preregistered GO / NO-GO gate unchanged.
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
    bone_p3_c0_cross_dataset_formal_protocol as protocol,
)
from experiments.utility_relation import (
    evaluate_bone_p3_c0_cross_dataset_formal as c0_eval,
)
from experiments.utility_relation.p3_c1_b1_optuna import (
    msrc_adapter,
)


ROOT = Path("/root/autodl-tmp/BONE")

PRE_GT_ROOT = (
    ROOT
    / "outputs/bone_p4/"
      "p4_b1_uases_shadow_pilot"
)

PREREG = (
    ROOT
    / "experiment_freeze/"
      "bone_p4_b1_uases_multidataset_validity_pilot_preregistered_20261006/"
      "PROTOCOL.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs/bone_p4/"
      "p4_b1_uases_postseal_evaluation"
)

H_GRID = tuple(
    range(
        5,
        101,
        5,
    )
)

ARMS = (
    "BASE",
    "OURS",
)

CELLS = (
    (
        "MSRC",
        20,
        PRE_GT_ROOT
        / "msrc/seed20",
    ),
    (
        "HandWrite",
        20,
        PRE_GT_ROOT
        / "handwrite/seed20",
    ),
    (
        "MNIST-USPS",
        50,
        PRE_GT_ROOT
        / "mnist_usps/seed50",
    ),
)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def _verify_manifest(
    manifest_path: Path,
) -> None:
    for line in manifest_path.read_text().splitlines():

        if not line.strip():
            continue

        wanted, raw_path = (
            line.split(
                None,
                1,
            )
        )

        path = Path(
            raw_path.strip()
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
    """
    This function MUST finish before any GT loader is called.
    """

    manifest = (
        PRE_GT_ROOT
        / "GLOBAL_PRE_GT_MANIFEST.sha256"
    )

    global_seal_path = (
        PRE_GT_ROOT
        / "GLOBAL_PRE_GT_SEAL.json"
    )

    global_seal_hash_path = (
        PRE_GT_ROOT
        / "GLOBAL_PRE_GT_SEAL.sha256"
    )

    _verify_manifest(
        manifest
    )

    _verify_manifest(
        global_seal_hash_path
    )

    global_seal = json.loads(
        global_seal_path.read_text()
    )

    required_true = (
        "all_three_cells_complete",
        "all_selected_h_frozen",
        "all_physical_trajectories_reached_H100",
    )

    if any(
        global_seal.get(key)
        is not True
        for key in required_true
    ):
        raise RuntimeError(
            "GLOBAL_PRE_GT_COMPLETION_FAILURE"
        )

    required_false = (
        "GT_read_before_global_seal",
        "ACC_NMI_ARI_computed_before_global_seal",
        "UASES_formula_changed",
        "patience_changed",
        "H_max_changed",
        "checkpoint_interval_changed",
    )

    if any(
        global_seal.get(key)
        is not False
        for key in required_false
    ):
        raise RuntimeError(
            "GLOBAL_PRE_GT_FIREWALL_FAILURE"
        )

    if (
        global_seal.get(
            "pilot_cell_count"
        )
        != 3
        or global_seal.get(
            "pilot_arm_cell_count"
        )
        != 6
    ):
        raise RuntimeError(
            "GLOBAL_PRE_GT_CELL_COUNT_FAILURE"
        )

    # Verify every cell-level seal again.
    for dataset, seed, root in CELLS:

        seal_path = (
            root
            / "PRE_GT_SEAL.json"
        )

        audit_path = (
            root
            / "PRE_GT_AUDIT.json"
        )

        trajectory_path = (
            root
            / "UASES_TRAJECTORY.json"
        )

        complete_path = (
            root
            / "CELL_COMPLETE.json"
        )

        for path in (
            seal_path,
            audit_path,
            trajectory_path,
            complete_path,
        ):
            if not path.is_file():
                raise RuntimeError(
                    f"MISSING_CELL_ARTIFACT:{path}"
                )

        seal = json.loads(
            seal_path.read_text()
        )

        audit = json.loads(
            audit_path.read_text()
        )

        trajectory = json.loads(
            trajectory_path.read_text()
        )

        complete = json.loads(
            complete_path.read_text()
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
                f"CELL_COMPLETE_FAILURE:{dataset}:{seed}"
            )

        if (
            seal.get(
                "GT_loaded_before_seal"
            )
            is not False
            or seal.get(
                "selected_h_frozen"
            )
            is not True
        ):
            raise RuntimeError(
                f"CELL_SEAL_FIREWALL_FAILURE:{dataset}:{seed}"
            )

        for rel, wanted in (
            seal["artifacts"]
        ).items():

            path = (
                root
                / rel
            )

            if sha256_file(
                path
            ) != wanted:
                raise RuntimeError(
                    f"CELL_SEAL_HASH_FAILURE:{path}"
                )

        audit_false = (
            "GT_loaded_before_seal",
            "full_gt_used_for_training",
            "full_gt_used_for_stopping",
            "full_gt_used_for_selection",
            "ACC_NMI_ARI_computed",
            "post_stop_shadow_can_change_selected_h",
        )

        if any(
            audit.get(key)
            is not False
            for key in audit_false
        ):
            raise RuntimeError(
                f"CELL_AUDIT_FIREWALL_FAILURE:{dataset}:{seed}"
            )

        if (
            audit.get(
                "BASE_OURS_initial_state_parity"
            )
            is not True
            or audit.get(
                "BASE_OURS_native_order_parity"
            )
            is not True
            or audit.get(
                "same_U0_for_BASE_and_OURS"
            )
            is not True
            or audit.get(
                "same_stopping_rule_for_BASE_and_OURS"
            )
            is not True
        ):
            raise RuntimeError(
                f"CELL_PARITY_FAILURE:{dataset}:{seed}"
            )

        if (
            audit.get(
                "native_epoch_count"
            )
            != 100
        ):
            raise RuntimeError(
                f"CELL_H100_FAILURE:{dataset}:{seed}"
            )

        if (
            trajectory.get(
                "GT_loaded"
            )
            is not False
            or trajectory.get(
                "shadow_mode"
            )
            is not True
        ):
            raise RuntimeError(
                f"TRAJECTORY_FIREWALL_FAILURE:{dataset}:{seed}"
            )

    return global_seal


def _load_prediction_trajectories_after_seal():
    """
    Prediction artifacts are not GT.
    Called only after full seal verification for simpler provenance.
    """

    payload = {}

    for dataset, seed, root in CELLS:

        trajectory = json.loads(
            (
                root
                / "UASES_TRAJECTORY.json"
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

        sample_ids_ref = None

        for arm in ARMS:

            with np.load(
                root
                / arm
                / "checkpoint_predictions.npz",
                allow_pickle=False,
            ) as z:

                sample_ids = np.asarray(
                    z[
                        "sample_ids"
                    ],
                    dtype=np.int64,
                )

                preds = {
                    h:
                        np.asarray(
                            z[
                                f"pred_H{h}"
                            ],
                            dtype=np.int64,
                        )
                    for h in H_GRID
                }

            if sample_ids_ref is None:
                sample_ids_ref = (
                    sample_ids.copy()
                )

            elif not np.array_equal(
                sample_ids_ref,
                sample_ids,
            ):
                raise RuntimeError(
                    f"BASE_OURS_SAMPLE_ID_FAILURE:{dataset}:{seed}"
                )

            for h in H_GRID:
                if (
                    preds[h].shape
                    != sample_ids.shape
                ):
                    raise RuntimeError(
                        f"PREDICTION_SHAPE_FAILURE:{dataset}:{seed}:{arm}:{h}"
                    )

            selected_h = int(
                trajectory[
                    arm
                ][
                    "selected_h"
                ]
            )

            stop_h = int(
                trajectory[
                    arm
                ][
                    "stop_h"
                ]
            )

            if selected_h not in H_GRID:
                raise RuntimeError(
                    "INVALID_FROZEN_SELECTED_H"
                )

            cell[arm] = {
                "selected_h":
                    selected_h,

                "stop_h":
                    stop_h,

                "predictions":
                    preds,
            }

        cell[
            "sample_ids"
        ] = sample_ids_ref

        payload[
            (
                dataset,
                seed,
            )
        ] = cell

    return payload


def _load_gt_after_all_seals(
    dataset: str,
    sample_ids: np.ndarray,
) -> np.ndarray:
    """
    FIRST and ONLY full-GT loading path in this evaluator.
    """

    if dataset == "MSRC":

        artifact = (
            msrc_adapter
            .p3_a4_materializer
            .P1_ARTIFACT
        )

        expected_hash = (
            msrc_adapter
            .p3_a4_materializer
            .P1_ARTIFACT_SHA256
        )

        got_hash = (
            msrc_adapter
            .p3_a4_materializer
            .sha256_file(
                artifact
            )
        )

        if got_hash != expected_hash:
            raise RuntimeError(
                "MSRC_P1_ARTIFACT_HASH_FAILURE"
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


def _metrics(
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
            "POSTSEAL_OUTPUT_ALREADY_EXISTS_FAIL_CLOSED"
        )

    # ========================================================
    # CRITICAL FIREWALL:
    # ALL seal checks finish before FIRST GT read.
    # ========================================================

    global_seal = (
        verify_all_pre_gt_seals()
    )

    frozen = (
        _load_prediction_trajectories_after_seal()
    )

    print(
        "ALL_PRE_GT_SEALS_VERIFIED_BEFORE_FIRST_GT_READ=PASS",
        flush=True,
    )

    print(
        "ALL_6_SELECTED_H_FROZEN_BEFORE_GT=true",
        flush=True,
    )

    print(
        "GT_HAS_NOT_BEEN_READ_YET=true",
        flush=True,
    )

    # ========================================================
    # FIRST GT READ occurs below this point.
    # ========================================================

    all_checkpoint_rows = []

    summary_rows = []

    cell_summary = []

    for dataset, seed, _root in CELLS:

        cell = frozen[
            (
                dataset,
                seed,
            )
        ]

        labels = (
            _load_gt_after_all_seals(
                dataset,
                cell[
                    "sample_ids"
                ],
            )
        )

        print(
            f"GT_LOAD_AFTER_SEAL=PASS "
            f"dataset={dataset} "
            f"seed={seed}",
            flush=True,
        )

        arm_selected = {}

        for arm in ARMS:

            trajectory_metrics = []

            for h in H_GRID:

                result = _metrics(
                    labels,
                    cell[
                        arm
                    ][
                        "predictions"
                    ][h],
                )

                row = {
                    "dataset":
                        dataset,

                    "training_seed":
                        seed,

                    "arm":
                        arm,

                    "H":
                        h,

                    **result,
                }

                trajectory_metrics.append(
                    row
                )

                all_checkpoint_rows.append(
                    row
                )

            selected_h = (
                cell[
                    arm
                ][
                    "selected_h"
                ]
            )

            fixed_h = int(
                cell[
                    "fixed_h"
                ]
            )

            selected = next(
                row
                for row
                in trajectory_metrics
                if row["H"]
                == selected_h
            )

            fixed = next(
                row
                for row
                in trajectory_metrics
                if row["H"]
                == fixed_h
            )

            # Oracle is POST-HOC diagnostic only.
            # Deterministic tie break: earliest H.
            oracle = sorted(
                trajectory_metrics,
                key=lambda row: (
                    -row[
                        "Mean3"
                    ],
                    row["H"],
                ),
            )[0]

            selected_rank = (
                1
                + sum(
                    row[
                        "Mean3"
                    ]
                    > selected[
                        "Mean3"
                    ]
                    for row
                    in trajectory_metrics
                )
            )

            delta_mean3 = float(
                selected[
                    "Mean3"
                ]
                - fixed[
                    "Mean3"
                ]
            )

            oracle_regret = float(
                oracle[
                    "Mean3"
                ]
                - selected[
                    "Mean3"
                ]
            )

            summary = {
                "dataset":
                    dataset,

                "training_seed":
                    seed,

                "arm":
                    arm,

                "fixed_h":
                    fixed_h,

                "selected_h":
                    selected_h,

                "stop_h":
                    int(
                        cell[
                            arm
                        ][
                            "stop_h"
                        ]
                    ),

                "selected_ACC":
                    selected[
                        "ACC"
                    ],

                "selected_NMI":
                    selected[
                        "NMI"
                    ],

                "selected_ARI":
                    selected[
                        "ARI"
                    ],

                "selected_Mean3":
                    selected[
                        "Mean3"
                    ],

                "fixed_ACC":
                    fixed[
                        "ACC"
                    ],

                "fixed_NMI":
                    fixed[
                        "NMI"
                    ],

                "fixed_ARI":
                    fixed[
                        "ARI"
                    ],

                "fixed_Mean3":
                    fixed[
                        "Mean3"
                    ],

                "DeltaMean3":
                    delta_mean3,

                "oracle_h":
                    int(
                        oracle["H"]
                    ),

                "oracle_ACC":
                    oracle[
                        "ACC"
                    ],

                "oracle_NMI":
                    oracle[
                        "NMI"
                    ],

                "oracle_ARI":
                    oracle[
                        "ARI"
                    ],

                "oracle_Mean3":
                    oracle[
                        "Mean3"
                    ],

                "oracle_regret":
                    oracle_regret,

                "selected_rank":
                    int(
                        selected_rank
                    ),
            }

            summary_rows.append(
                summary
            )

            arm_selected[
                arm
            ] = summary

        cell_summary.append(
            {
                "dataset":
                    dataset,

                "training_seed":
                    seed,

                "OURS_selected_minus_BASE_selected": {
                    metric:
                        float(
                            arm_selected[
                                "OURS"
                            ][
                                f"selected_{metric}"
                            ]
                            -
                            arm_selected[
                                "BASE"
                            ][
                                f"selected_{metric}"
                            ]
                        )
                    for metric
                    in (
                        "ACC",
                        "NMI",
                        "ARI",
                        "Mean3",
                    )
                },
            }
        )

    if len(
        all_checkpoint_rows
    ) != 120:
        raise RuntimeError(
            "EXPECTED_120_CHECKPOINT_METRICS"
        )

    if len(
        summary_rows
    ) != 6:
        raise RuntimeError(
            "EXPECTED_6_ARM_CELL_SUMMARIES"
        )

    deltas = np.asarray(
        [
            row[
                "DeltaMean3"
            ]
            for row
            in summary_rows
        ],
        dtype=np.float64,
    )

    ranks = np.asarray(
        [
            row[
                "selected_rank"
            ]
            for row
            in summary_rows
        ],
        dtype=np.int64,
    )

    mean_delta = float(
        deltas.mean()
    )

    nonnegative_count = int(
        np.count_nonzero(
            deltas >= 0.0
        )
    )

    minimum_delta = float(
        deltas.min()
    )

    top_half_count = int(
        np.count_nonzero(
            ranks <= 10
        )
    )

    gates = {
        "mean_DeltaMean3_gte_0":
            bool(
                mean_delta >= 0.0
            ),

        "nonnegative_count_gte_3":
            bool(
                nonnegative_count
                >= 3
            ),

        "minimum_single_DeltaMean3_gte_minus_0p02":
            bool(
                minimum_delta
                >= -0.02
            ),

        "top_half_selected_count_gte_4":
            bool(
                top_half_count
                >= 4
            ),
    }

    decision = (
        "GO"
        if all(
            gates.values()
        )
        else "NO_GO"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_csv = (
        output_root
        / "P4_B1_ALL_CHECKPOINT_METRICS.csv"
    )

    with checkpoint_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                "dataset",
                "training_seed",
                "arm",
                "H",
                "ACC",
                "NMI",
                "ARI",
                "Mean3",
            ),
        )

        writer.writeheader()

        writer.writerows(
            all_checkpoint_rows
        )

    summary_csv = (
        output_root
        / "P4_B1_SELECTED_FIXED_ORACLE_SUMMARY.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=tuple(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    result = {
        "stage":
            "P4-B1",

        "role":
            "POSTSEAL_VALIDITY_PILOT",

        "paper_formal_result":
            False,

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

        "oracle_used_for_selection":
            False,

        "checkpoint_metric_count":
            len(
                all_checkpoint_rows
            ),

        "arm_cell_count":
            len(
                summary_rows
            ),

        "summary_rows":
            summary_rows,

        "cell_selected_comparison":
            cell_summary,

        "gate": {
            "mean_DeltaMean3":
                mean_delta,

            "nonnegative_DeltaMean3_count":
                nonnegative_count,

            "minimum_single_DeltaMean3":
                minimum_delta,

            "selected_top_half_count":
                top_half_count,

            "conditions":
                gates,

            "decision":
                decision,
        },

        "global_pre_gt_seal_sha256":
            sha256_file(
                PRE_GT_ROOT
                / "GLOBAL_PRE_GT_SEAL.json"
            ),
    }

    result_json = (
        output_root
        / "P4_B1_POSTSEAL_RESULTS.json"
    )

    result_json.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    gate_json = (
        output_root
        / "P4_B1_GO_NO_GO.json"
    )

    gate_json.write_text(
        json.dumps(
            result[
                "gate"
            ],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if not args.execute:
        raise SystemExit(
            "refusing to load GT without --execute"
        )

    result = evaluate(
        args.output_root
    )

    print()
    print("=" * 132)
    print("P4-B1 SELECTED vs FIXED-H vs POST-HOC ORACLE")
    print("=" * 132)

    print(
        f"{'Dataset':<12} "
        f"{'Arm':<5} "
        f"{'FixH':>5} "
        f"{'SelH':>5} "
        f"{'ACC':>9} "
        f"{'NMI':>9} "
        f"{'ARI':>9} "
        f"{'Mean3':>9} "
        f"{'FixM3':>9} "
        f"{'DeltaM3':>10} "
        f"{'OracleH':>8} "
        f"{'OracleM3':>9} "
        f"{'Regret':>9} "
        f"{'Rank':>5}"
    )

    print("-" * 132)

    for row in result[
        "summary_rows"
    ]:

        print(
            f"{row['dataset']:<12} "
            f"{row['arm']:<5} "
            f"{row['fixed_h']:>5d} "
            f"{row['selected_h']:>5d} "
            f"{row['selected_ACC']:>9.6f} "
            f"{row['selected_NMI']:>9.6f} "
            f"{row['selected_ARI']:>9.6f} "
            f"{row['selected_Mean3']:>9.6f} "
            f"{row['fixed_Mean3']:>9.6f} "
            f"{row['DeltaMean3']:>+10.6f} "
            f"{row['oracle_h']:>8d} "
            f"{row['oracle_Mean3']:>9.6f} "
            f"{row['oracle_regret']:>9.6f} "
            f"{row['selected_rank']:>5d}"
        )

    print()
    print("=" * 90)
    print("OURS SELECTED - BASE SELECTED")
    print("=" * 90)

    for row in result[
        "cell_selected_comparison"
    ]:

        d = row[
            "OURS_selected_minus_BASE_selected"
        ]

        print(
            f"{row['dataset']:<12} "
            f"ACC={d['ACC']:+.6f} "
            f"NMI={d['NMI']:+.6f} "
            f"ARI={d['ARI']:+.6f} "
            f"Mean3={d['Mean3']:+.6f}"
        )

    gate = result[
        "gate"
    ]

    print()
    print("=" * 90)
    print("P4-B1 PREREGISTERED GO / NO-GO")
    print("=" * 90)

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
        "/ 6",
    )

    print(
        "minimum_single_DeltaMean3 =",
        gate[
            "minimum_single_DeltaMean3"
        ],
    )

    print(
        "selected_top_half_count =",
        gate[
            "selected_top_half_count"
        ],
        "/ 6",
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
        "P4_B1_DECISION =",
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
        "ORACLE_DIAGNOSTIC_ONLY=true"
    )


if __name__ == "__main__":
    main()
