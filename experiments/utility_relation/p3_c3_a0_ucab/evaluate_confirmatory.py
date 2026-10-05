"""Post-seal-only UCAB evaluator. Seal validation precedes the sole full-GT read."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation.evaluate_bone_p3_c0_cross_dataset_formal import compute_metrics
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol

from . import protocol
from . import validation_io as io


def _legal_gt_path(dataset: str, full_gt_path: str | Path | None) -> Path:
    expected = io.ROOT / "data" / f"{dataset}.mat"
    candidate = expected if full_gt_path is None else Path(full_gt_path)
    if candidate.resolve() != expected.resolve():
        raise ValueError("ILLEGAL_FULL_GT_SOURCE")
    if not candidate.is_file():
        raise FileNotFoundError("FULL_GT_SOURCE_MISSING")
    return candidate


def _validated_predictions(root: Path, dataset: str, training_seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = int(c0_protocol.DATASET_SPECS[dataset].n)
    result = {}
    for arm in io.arms_for_seed(training_seed):
        path = root / arm / "pre_gt_predictions.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
                predictions = np.asarray(archive["predictions"], dtype=np.int64)
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError(f"PREDICTION_SCHEMA_INVALID:{arm}") from error
        if sample_ids.ndim != 1 or predictions.ndim != 1 or sample_ids.shape != predictions.shape or sample_ids.size != n:
            raise RuntimeError(f"PREDICTION_SAMPLE_ID_SHAPE_MISMATCH:{arm}")
        if np.unique(sample_ids).size != n or not np.array_equal(np.sort(sample_ids), np.arange(n)):
            raise RuntimeError(f"PREDICTION_SAMPLE_ID_ALIGNMENT_MISMATCH:{arm}")
        result[arm] = (sample_ids, predictions)
    return result


def evaluate_confirmatory(seed_root: str | Path, full_gt_path: str | Path | None = None) -> dict:
    """Evaluate one sealed seed root; its seal supplies the dataset and seed identity."""
    root = Path(seed_root)
    verified = io.verify_pre_gt_seal(root)  # Required ordering: no GT access before this line.
    dataset, training_seed = verified["seal"]["dataset"], verified["seal"]["training_seed"]
    io.verify_all_pre_gt_runs_sealed(root.parents[1])
    predictions = _validated_predictions(root, dataset, training_seed)
    metrics_path = root / "formal_metrics.json"
    if metrics_path.is_file():
        existing = io.read_json(metrics_path, "FORMAL_METRICS")
        if existing.get("dataset") != dataset or existing.get("training_seed") != training_seed or existing.get("pre_gt_seal_sha256") != io.sha256_file(root / "seal/pre_gt_seal.json"):
            raise RuntimeError("EXISTING_FORMAL_METRICS_IDENTITY_MISMATCH")
        return existing
    label_path = _legal_gt_path(dataset, full_gt_path)
    labels = np.asarray(scipy.io.loadmat(label_path, variable_names=("Y",))["Y"]).reshape(-1)
    if labels.size != next(iter(predictions.values()))[0].size:
        raise RuntimeError("FULL_GT_PREDICTION_CARDINALITY_MISMATCH")
    arms = {arm: {name: float(value) for name, value in compute_metrics(labels[sample_ids], prediction).items() if name in io.METRICS} for arm, (sample_ids, prediction) in predictions.items()}
    result = {"status": "REPORTING_ONLY", "dataset": dataset, "training_seed": training_seed, "seed_role": io.seed_role(training_seed), "arms": list(io.arms_for_seed(training_seed)), "pre_gt_seal_sha256": io.sha256_file(root / "seal/pre_gt_seal.json"), "metrics": arms, "metrics_used_for_selection": False, "full_gt_access_phase": "POST_SEAL_AFTER_ALL_12_PRE_GT_RUNS"}
    for arm, values in arms.items():
        result[arm] = values
    comparisons = (("TRUE_BUDGET", "CURRENT"),) if io.seed_role(training_seed) == "historical" else (("TRUE_BUDGET", "CURRENT"), ("TRUE_BUDGET", "MEAN_BUDGET"), ("TRUE_BUDGET", "SHUFFLE_BUDGET"))
    result["paired_deltas"] = {f"{left}_minus_{right}": {metric: arms[left][metric] - arms[right][metric] for metric in io.METRICS} for left, right in comparisons}
    io.write_json(metrics_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", required=True, type=Path)
    parser.add_argument("--full-gt-path", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to load full GT without --execute")
    print(evaluate_confirmatory(args.seed_root, args.full_gt_path))


if __name__ == "__main__":
    main()
