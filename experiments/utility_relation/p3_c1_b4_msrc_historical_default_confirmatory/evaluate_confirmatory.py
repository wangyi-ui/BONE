"""Post-seal-only B4 reporting evaluator; never used for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io

from experiments.utility_relation.evaluate_bone_p3_c0_cross_dataset_formal import compute_metrics
from . import protocol
from .run_confirmatory import verify_complete_run


def evaluate_confirmatory(seed: int, output_root: str | Path = protocol.OUTPUT_ROOT) -> dict:
    root = protocol.run_root(output_root, seed)
    verify_complete_run(root)
    result_path = root / "CONFIRMATORY_METRICS.json"
    if result_path.is_file(): return json.loads(result_path.read_text(encoding="utf-8"))
    labels = np.asarray(scipy.io.loadmat(protocol.REPO_ROOT / "data/MSRC.mat", variable_names=("Y",))["Y"]).reshape(-1)
    arms = {}
    for arm in protocol.ARMS:
        with np.load(root / arm / "pre_gt_predictions.npz", allow_pickle=False) as archive:
            ids, predictions = np.asarray(archive["sample_ids"], dtype=np.int64), np.asarray(archive["predictions"], dtype=np.int64)
        if ids.shape != predictions.shape or not np.array_equal(np.sort(ids), np.arange(labels.size)): raise RuntimeError("B4_PREDICTION_ID_PARITY_FAILURE")
        values = compute_metrics(labels[ids], predictions)
        arms[arm] = {metric: float(values[metric]) for metric in ("ACC", "NMI", "ARI")}
    output = {"status": "REPORTING_ONLY", "dataset": protocol.DATASET, "training_seed": seed, **arms, "delta_historical_default_minus_base": {metric: arms["OURS_HISTORICAL_DEFAULT"][metric] - arms["BASE"][metric] for metric in ("ACC", "NMI", "ARI")}, "metrics_used_for_parameter_selection": False, "parameter_retuning_allowed": False, "full_gt_access_phase": "POST_TRAINING_AFTER_PRE_GT_SEAL"}
    protocol.write_json(result_path, output); return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--training-seed", choices=protocol.CONFIRMATORY_SEEDS, type=int, required=True); parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute: raise SystemExit("refusing GT read without --execute")
    print(json.dumps(evaluate_confirmatory(args.training_seed, args.output_root), indent=2))


if __name__ == "__main__": main()
