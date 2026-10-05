"""
Post-seal evaluator for P3-C1-B0 candidate validation.

The evaluator verifies the seal BEFORE loading full GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol as c0_protocol,
)
from experiments.utility_relation import (
    evaluate_bone_p3_c0_cross_dataset_formal as c0_eval,
)
from experiments.utility_relation import (
    bone_p3_c1_b0_candidate_protocol as candidate,
)
from experiments.utility_relation import (
    run_bone_p3_c1_b0_candidate_validation as runner,
)


def evaluate(
    dataset: str,
    seed: int,
) -> dict:

    root = runner.root_for(
        dataset,
        seed,
    )

    seal_path = (
        root
        / "seal"
        / "pre_gt_seal.json"
    )

    audit_path = (
        root
        / "seal"
        / "pre_gt_audit.json"
    )

    if not seal_path.is_file():
        raise FileNotFoundError(
            seal_path
        )

    seal = json.loads(
        seal_path.read_text()
    )

    audit = json.loads(
        audit_path.read_text()
    )

    formal_root = runner.formal_root(
        dataset,
        seed,
    )

    targets = {
        "carrier_state":
            formal_root
            / "carrier"
            / "carrier_model_state.pt",

        "carrier_npz":
            formal_root
            / "carrier"
            / "carrier_pre_gt.npz",

        "base_state":
            root
            / "base"
            / "final_model_state.pt",

        "ours_state":
            root
            / "ours"
            / "final_model_state.pt",

        "base_predictions":
            root
            / "base"
            / "pre_gt_predictions.npz",

        "ours_predictions":
            root
            / "ours"
            / "pre_gt_predictions.npz",

        "audit":
            audit_path,

        "runner_source":
            Path(runner.__file__),

        "protocol_source":
            Path(candidate.__file__),
    }

    expected = seal.get(
        "sha256"
    )

    if not isinstance(
        expected,
        dict,
    ):
        raise RuntimeError(
            "invalid seal"
        )

    if set(expected) != set(targets):
        raise RuntimeError(
            "seal schema mismatch"
        )

    for name, path in targets.items():

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        actual = (
            c0_protocol.sha256_file(
                path
            )
        )

        if actual != expected[name]:
            raise RuntimeError(
                f"seal mismatch: {name}"
            )

    # ---------------------------------------------------------
    # GT firewall checks BEFORE GT read.
    # ---------------------------------------------------------

    required_false = (
        "GT_loaded_before_seal",
        "full_gt_used_for_training",
        "full_gt_used_for_model_selection",
    )

    for key in required_false:
        if audit.get(key) is not False:
            raise RuntimeError(
                f"GT firewall failed: {key}"
            )

    if (
        dataset
        in candidate.VALIDATION_DATASETS
    ):

        if audit.get(
            "full_gt_used_for_hyperparameter_selection"
        ) is not False:
            raise RuntimeError(
                "validation GT selection leakage"
            )

        if audit.get(
            "hyperparameters_selected_on"
        ) != "ALOI_only":
            raise RuntimeError(
                "candidate provenance mismatch"
            )

    if not (
        audit[
            "initial_BASE_state_sha256"
        ]
        ==
        audit[
            "initial_OURS_state_sha256"
        ]
        ==
        audit[
            "carrier_state_sha256"
        ]
    ):
        raise RuntimeError(
            "shared carrier parity failed"
        )

    # =========================================================
    # FIRST full-GT access occurs here.
    # =========================================================

    labels = np.asarray(
        scipy.io.loadmat(
            c0_protocol.DATA_ROOT
            / f"{dataset}.mat",
            variable_names=("Y",),
        )["Y"]
    ).reshape(-1)

    arms = {}

    for arm in (
        "base",
        "ours",
    ):

        pred_path = (
            root
            / arm
            / "pre_gt_predictions.npz"
        )

        with np.load(
            pred_path,
            allow_pickle=False,
        ) as d:

            sample_ids = np.asarray(
                d["sample_ids"],
                dtype=np.int64,
            )

            predictions = np.asarray(
                d["predictions"],
                dtype=np.int64,
            )

        if not np.array_equal(
            np.sort(sample_ids),
            np.arange(labels.size),
        ):
            raise RuntimeError(
                "sample-ID parity failure"
            )

        if predictions.shape != sample_ids.shape:
            raise RuntimeError(
                "prediction shape mismatch"
            )

        arms[
            arm.upper()
        ] = c0_eval.compute_metrics(
            labels[
                sample_ids
            ],
            predictions,
        )

    delta = {
        metric:
            float(
                arms["OURS"][metric]
                - arms["BASE"][metric]
            )
        for metric
        in arms["BASE"]
    }

    result = {
        "experiment":
            "P3-C1-B0_candidate_validation",

        "dataset":
            dataset,

        "training_seed":
            seed,

        "semantic_lr":
            candidate.SEMANTIC_LR,

        "refinement_budget":
            candidate.REFINEMENT_BUDGET,

        "BASE":
            arms["BASE"],

        "OURS":
            arms["OURS"],

        "delta_OURS_minus_BASE":
            delta,

        "hyperparameters_selected_on":
            "ALOI_only",

        "held_out_validation":
            (
                dataset
                in candidate.VALIDATION_DATASETS
            ),
    }

    output = (
        root
        / "postseal_metrics.json"
    )

    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "P3_C1_B0_POSTSEAL_EVALUATION_PASS"
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=(
            candidate.VALIDATION_DATASETS
            + (
                candidate.TUNING_DATASET,
            )
        ),
        required=True,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        choices=candidate.SEEDS,
        required=True,
    )

    args = parser.parse_args()

    evaluate(
        args.dataset,
        args.training_seed,
    )


if __name__ == "__main__":
    main()
