"""Post-PRE_GT_SEAL-only ACC/NMI/ARI evaluation for all three B3 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation.evaluate_bone_p3_c0_cross_dataset_formal import (
    compute_metrics,
)

from . import protocol
from .run_confirmatory import verify_complete_run


def evaluate_confirmatory(
    training_seed: int,
    output_root: str | Path = protocol.OUTPUT_ROOT,
) -> dict:
    root = protocol.run_root(output_root, training_seed)
    verified = verify_complete_run(root)  # This must complete before the sole Y read.
    config, audit = verified["config"], verified["audit"]
    if (
        config.get("dataset") != protocol.DATASET
        or config.get("training_seed") != training_seed
        or audit.get("ACC_NMI_ARI_computed") is not False
    ):
        raise RuntimeError("B3_EVALUATION_RUN_IDENTITY_INVALID")
    metrics_path = root / "CONFIRMATORY_METRICS.json"
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") != protocol.DATASET
            or existing.get("training_seed") != training_seed
            or existing.get("pre_gt_seal_sha256")
            != protocol.sha256_file(root / "PRE_GT_SEAL.json")
        ):
            raise RuntimeError("B3_EXISTING_METRICS_IDENTITY_MISMATCH")
        return existing

    # The only full-GT access in B3: reporting-only, after complete seal verification.
    labels = np.asarray(
        scipy.io.loadmat(
            protocol.REPO_ROOT / "data/MSRC.mat", variable_names=("Y",)
        )["Y"]
    ).reshape(-1)
    arms = {}
    for arm in protocol.ARMS:
        with np.load(
            root / arm / "pre_gt_predictions.npz", allow_pickle=False
        ) as archive:
            sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
            predictions = np.asarray(archive["predictions"], dtype=np.int64)
        if (
            sample_ids.shape != predictions.shape
            or not np.array_equal(np.sort(sample_ids), np.arange(labels.size))
        ):
            raise RuntimeError("B3_PREDICTION_SAMPLE_ID_PARITY_FAILURE")
        values = compute_metrics(labels[sample_ids], predictions)
        arms[arm] = {name: float(values[name]) for name in ("ACC", "NMI", "ARI")}

    def delta(left: str, right: str) -> dict[str, float]:
        return {
            metric: arms[left][metric] - arms[right][metric]
            for metric in ("ACC", "NMI", "ARI")
        }

    result = {
        "status": "REPORTING_ONLY",
        "dataset": protocol.DATASET,
        "training_seed": training_seed,
        "arm_parameters": protocol.ARM_PARAMETERS,
        "pre_gt_seal_sha256": protocol.sha256_file(root / "PRE_GT_SEAL.json"),
        **arms,
        "B1_minus_BASE": delta("OURS_B1", "BASE"),
        "B2_minus_BASE": delta("OURS_B2", "BASE"),
        "B2_minus_B1": delta("OURS_B2", "OURS_B1"),
        "metrics_used_for_parameter_selection": False,
        "parameter_retuning_allowed": False,
        "full_gt_access_phase": "POST_TRAINING_AFTER_PRE_GT_SEAL",
    }
    protocol.write_json(metrics_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed",
        type=int,
        choices=protocol.CONFIRMATORY_SEEDS,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to read full GT without --execute")
    print(
        json.dumps(
            evaluate_confirmatory(args.training_seed, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
