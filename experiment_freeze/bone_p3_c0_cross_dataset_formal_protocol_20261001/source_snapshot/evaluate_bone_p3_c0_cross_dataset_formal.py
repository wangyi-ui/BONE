"""Post-seal-only P3-C0 evaluator. This is the only module allowed to load Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as protocol


def cluster_acc_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Deterministic maximum-cardinality Hungarian clustering accuracy."""
    truth, predicted = np.asarray(y_true, dtype=np.int64).reshape(-1), np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0: raise ValueError("invalid prediction/label shapes")
    rows, cols = np.unique(predicted), np.unique(truth)
    counts = np.zeros((rows.size, cols.size), dtype=np.int64)
    for row, col in zip(predicted, truth): counts[np.searchsorted(rows, row), np.searchsorted(cols, col)] += 1
    matched = linear_sum_assignment(counts.max() - counts)
    return float(counts[matched].sum() / truth.size)


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, predicted = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    return float(sum(np.max(np.unique(truth[predicted == cluster], return_counts=True)[1]) for cluster in np.unique(predicted)) / truth.size)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"ACC": cluster_acc_hungarian(y_true, y_pred), "NMI": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
            "Purity": purity(y_true, y_pred), "ARI": float(metrics.adjusted_rand_score(y_true, y_pred))}


def _seed_root(dataset_name: str, seed: int, output_root: Path) -> Path:
    return Path(output_root) / protocol.dataset_slug(dataset_name) / f"seed{seed}"


def verify_pre_gt_seal(dataset_name: str, seed: int, output_root: Path = protocol.OUTPUT_ROOT) -> dict[str, Any]:
    """Verify every pre-GT artifact and paired-state invariant before loading Y."""
    root = _seed_root(dataset_name, seed, output_root); seal_path, audit_path = root / "seal" / "pre_gt_seal.json", root / "seal" / "pre_gt_audit.json"
    if not seal_path.is_file() or not audit_path.is_file(): raise ValueError("missing P3-C0 pre-GT seal or audit")
    seal = json.loads(seal_path.read_text()); expected = seal.get("sha256")
    targets = {"carrier_state": root / "carrier" / "carrier_model_state.pt", "carrier_npz": root / "carrier" / "carrier_pre_gt.npz",
               "base_state": root / "base" / "final_model_state.pt", "ours_state": root / "ours" / "final_model_state.pt",
               "base_predictions": root / "base" / "pre_gt_predictions.npz", "ours_predictions": root / "ours" / "pre_gt_predictions.npz",
               "audit": audit_path, "runner_source": protocol.REPO_ROOT / "experiments/utility_relation/run_bone_p3_c0_cross_dataset_formal.py",
               "protocol_source": Path(protocol.__file__), "p3_a1_source": protocol.PARENT_FREEZE_PATHS["p3_a1_source"],
               "p3_a3_source": protocol.PARENT_FREEZE_PATHS["p3_a3_source"], "p3_a4_source": protocol.PARENT_FREEZE_PATHS["p3_a4_source"],
               "p3_b0_source": protocol.PARENT_FREEZE_PATHS["p3_b0_source"], "p3_b1_runner": protocol.PARENT_FREEZE_PATHS["p3_b1_runner"],
               "p3_a3_split": protocol.artifact_paths(dataset_name)[0], "p3_a4_mask": protocol.artifact_paths(dataset_name)[1]}
    if not isinstance(expected, dict) or set(expected) != set(targets): raise ValueError("invalid P3-C0 pre-GT seal schema")
    for name, path in targets.items():
        if not path.is_file() or expected[name] != protocol.sha256_file(path): raise ValueError(f"invalid P3-C0 seal entry: {name}")
    audit = json.loads(audit_path.read_text())
    required_false = ("GT_loaded_before_seal", "full_gt_used_for_training", "full_gt_used_for_model_selection", "full_gt_used_for_hyperparameter_selection")
    if audit.get("dataset") != dataset_name or audit.get("training_seed") != seed or any(audit.get(name) is not False for name in required_false):
        raise ValueError("invalid P3-C0 pre-GT isolation audit")
    if audit.get("carrier_state_sha256") != audit.get("initial_BASE_state_sha256") or audit.get("carrier_state_sha256") != audit.get("initial_OURS_state_sha256"):
        raise ValueError("BASE/OURS shared-carrier parity failed")
    protocol.verify_native_hashes()
    protocol.verify_parent_freeze_hashes()
    return audit


def evaluate_seed(dataset_name: str, seed: int, output_root: Path = protocol.OUTPUT_ROOT) -> dict[str, Any]:
    """Verify seal first, then align raw Y with audited sample IDs and report metrics."""
    audit = verify_pre_gt_seal(dataset_name, seed, output_root)
    root = _seed_root(dataset_name, seed, output_root)
    # This explicit post-seal read is the sole full-GT access in P3-C0.
    labels = np.asarray(scipy.io.loadmat(protocol.DATA_ROOT / f"{dataset_name}.mat", variable_names=("Y",))["Y"]).reshape(-1)
    arms: dict[str, dict[str, float]] = {}
    for arm in ("base", "ours"):
        with np.load(root / arm / "pre_gt_predictions.npz", allow_pickle=False) as values:
            sample_ids, predictions = np.asarray(values["sample_ids"], dtype=np.int64), np.asarray(values["predictions"], dtype=np.int64)
        if not np.array_equal(np.sort(sample_ids), np.arange(labels.size)) or predictions.shape != sample_ids.shape: raise ValueError("prediction sample-ID parity failed")
        arms[arm.upper()] = compute_metrics(labels[sample_ids], predictions)
    result = {"dataset": dataset_name, "training_seed": seed, "pre_gt_audit_sha256": protocol.sha256_file(root / "seal" / "pre_gt_audit.json"),
              "BASE": arms["BASE"], "OURS": arms["OURS"], "delta_OURS_minus_BASE": {name: arms["OURS"][name] - arms["BASE"][name] for name in arms["BASE"]}}
    (root / "formal_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset", choices=protocol.FORMAL_DATASETS, required=True)
    parser.add_argument("--training-seed", type=int, choices=protocol.FORMAL_SEEDS, required=True); parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="verify one sealed dataset/seed and then read GT")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute: raise SystemExit("refusing to load GT without --execute")
    print(json.dumps(evaluate_seed(args.dataset, args.training_seed, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__": main()
