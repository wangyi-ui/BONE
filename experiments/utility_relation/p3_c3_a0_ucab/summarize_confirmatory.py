"""Reporting-only final gate summary for the preregistered UCAB validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import protocol
from . import validation_io as io


def _gate(deltas: np.ndarray) -> dict:
    return {"macro_mean": float(deltas.mean()), "positive_count": int(np.count_nonzero(deltas > 0)), "pass": bool(deltas.mean() > 0 and np.count_nonzero(deltas > 0) >= 4)}


def _load_metrics(output_root: str | Path) -> list[dict]:
    rows = []
    for dataset in protocol.DATASETS:
        for seed in (*protocol.HISTORICAL_PAIRED_SEEDS, *protocol.CONFIRMATORY_SEEDS):
            path = io.run_root(output_root, dataset, seed) / "formal_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"FORMAL_METRICS_MISSING:{dataset}:seed{seed}")
            row = io.read_json(path, "FORMAL_METRICS")
            if row.get("dataset") != dataset or row.get("training_seed") != seed or row.get("seed_role") != io.seed_role(seed):
                raise RuntimeError(f"FORMAL_METRICS_IDENTITY_INVALID:{dataset}:seed{seed}")
            if tuple(row.get("arms", ())) != io.arms_for_seed(seed):
                raise RuntimeError(f"FORMAL_METRICS_ARM_SET_INVALID:{dataset}:seed{seed}")
            rows.append(row)
    return rows


def _delta(row: dict, left: str, right: str, metric: str) -> float:
    return float(row[left][metric]) - float(row[right][metric])


def summarize_confirmatory(output_root: str | Path = io.OUTPUT_ROOT) -> dict:
    """Require all 12 results; historical rows are deliberately excluded from gates."""
    rows = _load_metrics(output_root)
    unseen = [row for row in rows if row["seed_role"] == "confirmatory"]
    historical = [row for row in rows if row["seed_role"] == "historical"]
    current_acc = np.asarray([_delta(row, "TRUE_BUDGET", "CURRENT", "ACC") for row in unseen])
    by_dataset = {dataset: np.asarray([_delta(row, "TRUE_BUDGET", "CURRENT", "ACC") for row in unseen if row["dataset"] == dataset]) for dataset in protocol.DATASETS}
    primary_core = _gate(current_acc)
    primary_pass = bool(primary_core["pass"] and all(values.size == 3 and values.mean() > 0 for values in by_dataset.values()))
    mean_gate = _gate(np.asarray([_delta(row, "TRUE_BUDGET", "MEAN_BUDGET", "ACC") for row in unseen]))
    shuffle_gate = _gate(np.asarray([_delta(row, "TRUE_BUDGET", "SHUFFLE_BUDGET", "ACC") for row in unseen]))
    mechanism_pass = bool(mean_gate["pass"] and shuffle_gate["pass"])
    secondary_means = {metric: float(np.mean([_delta(row, "TRUE_BUDGET", "CURRENT", metric) for row in unseen])) for metric in io.METRICS}
    secondary_pass = sum(value > 0 for value in secondary_means.values()) >= 2
    historical_summary = {metric: {"macro_mean": float(np.mean([_delta(row, "TRUE_BUDGET", "CURRENT", metric) for row in historical])), "positive_count": int(np.count_nonzero([_delta(row, "TRUE_BUDGET", "CURRENT", metric) > 0 for row in historical]))} for metric in io.METRICS}
    final_pass = primary_pass and mechanism_pass and secondary_pass
    return {
        "status": "REPORTING_ONLY",
        "completed_run_count": len(rows),
        "all_12_complete": True,
        "PRIMARY_GATE": "PASS" if primary_pass else "FAIL",
        "MECHANISM_GATE": "PASS" if mechanism_pass else "FAIL",
        "SECONDARY_GATE": "PASS" if secondary_pass else "FAIL",
        "P3_C3_A0_UCAB_CONFIRMATORY": "PASS" if final_pass else "FAIL_CLOSED",
        "primary": {**primary_core, "dataset_mean_delta_acc": {dataset: float(values.mean()) for dataset, values in by_dataset.items()}},
        "mechanism": {"true_minus_mean": mean_gate, "true_minus_shuffle": shuffle_gate},
        "secondary": {"macro_mean_deltas": secondary_means, "positive_metric_count": sum(value > 0 for value in secondary_means.values())},
        "historical_matched_evidence": historical_summary,
        "historical_affects_confirmatory_decision": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=io.OUTPUT_ROOT)
    parser.add_argument("--write", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize_confirmatory(args.output_root)
    if args.write:
        io.write_json(args.write, result)
    import json
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
