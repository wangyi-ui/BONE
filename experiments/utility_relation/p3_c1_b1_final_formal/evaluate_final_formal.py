"""Post-seal-only final evaluator; this is the only final module that reads Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation.evaluate_bone_p3_c0_cross_dataset_formal import (
    compute_metrics,
)

from . import final_formal_protocol as protocol
from .run_final_formal import verify_complete_run


def evaluate_final_formal(
    dataset: str,
    training_seed: int,
    output_root: str | Path = protocol.DEFAULT_OUTPUT_ROOT,
) -> dict:
    root = protocol.run_root(output_root, dataset, training_seed)
    verified = verify_complete_run(root)  # Must finish before the sole Y read below.
    config, audit = verified["config"], verified["audit"]
    if config["dataset"] != dataset or config["training_seed"] != training_seed:
        raise RuntimeError("EVALUATION_RUN_IDENTITY_MISMATCH")
    if audit.get("ACC_NMI_ARI_computed") is not False:
        raise RuntimeError("PRE_GT_AUDIT_ALREADY_CONTAINS_METRICS")

    metrics_path = root / "FORMAL_METRICS.json"
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") != dataset
            or existing.get("training_seed") != training_seed
            or existing.get("pre_gt_seal_sha256")
            != protocol.sha256_file(root / "PRE_GT_SEAL.json")
        ):
            raise RuntimeError("EXISTING_FORMAL_METRICS_IDENTITY_MISMATCH")
        return existing

    # Explicitly reporting-only, post-seal full-GT access.
    label_path = protocol.REPO_ROOT / "data" / f"{dataset}.mat"
    labels = np.asarray(
        scipy.io.loadmat(label_path, variable_names=("Y",))["Y"]
    ).reshape(-1)
    arms = {}
    for arm in ("BASE", "OURS"):
        with np.load(root / arm / "pre_gt_predictions.npz", allow_pickle=False) as archive:
            sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
            predictions = np.asarray(archive["predictions"], dtype=np.int64)
        if (
            sample_ids.shape != predictions.shape
            or not np.array_equal(np.sort(sample_ids), np.arange(labels.size))
        ):
            raise RuntimeError("PREDICTION_SAMPLE_ID_PARITY_FAILURE")
        arm_metrics = compute_metrics(labels[sample_ids], predictions)
        arms[arm] = {name: arm_metrics[name] for name in ("ACC", "NMI", "ARI")}
    result = {
        "status": "REPORTING_ONLY",
        "dataset": dataset,
        "training_seed": training_seed,
        "semantic_lr": config["semantic_lr"],
        "semantic_exposure": config["semantic_exposure"],
        "final_parameter_freeze_sha256": config["final_parameter_freeze_sha256"],
        "pre_gt_seal_sha256": protocol.sha256_file(root / "PRE_GT_SEAL.json"),
        "BASE": arms["BASE"],
        "OURS": arms["OURS"],
        "delta_OURS_minus_BASE": {
            name: arms["OURS"][name] - arms["BASE"][name]
            for name in ("ACC", "NMI", "ARI")
        },
        "metrics_used_for_parameter_selection": False,
        "parameter_retuning_allowed": False,
        "full_gt_access_phase": "POST_TRAINING_AFTER_PRE_GT_SEAL",
    }
    protocol.write_json(metrics_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=protocol.DATASETS, required=True)
    parser.add_argument(
        "--training-seed", type=int, choices=protocol.FORMAL_SEEDS, required=True
    )
    parser.add_argument("--output-root", type=Path, default=protocol.DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--execute", action="store_true", help="verify seal, then compute reporting metrics"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to read full GT without --execute")
    print(
        json.dumps(
            evaluate_final_formal(args.dataset, args.training_seed, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

