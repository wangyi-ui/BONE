"""Reporting-only summary of completed final formal metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import final_formal_protocol as protocol


def summarize_final_formal(
    output_root: str | Path = protocol.DEFAULT_OUTPUT_ROOT,
    require_all_24: bool = True,
) -> dict:
    rows = []
    for dataset in protocol.DATASETS:
        for seed in protocol.FORMAL_SEEDS:
            path = protocol.run_root(output_root, dataset, seed) / "FORMAL_METRICS.json"
            if not path.is_file():
                if require_all_24:
                    raise FileNotFoundError(f"FORMAL_METRICS_MISSING:{dataset}:seed{seed}")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("dataset") != dataset
                or row.get("training_seed") != seed
                or row.get("status") != "REPORTING_ONLY"
                or row.get("parameter_retuning_allowed") is not False
            ):
                raise RuntimeError(f"FORMAL_METRICS_INVALID:{dataset}:seed{seed}")
            rows.append(row)
    datasets = {}
    for dataset in protocol.DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        if not selected:
            continue
        datasets[dataset] = {}
        for metric in ("ACC", "NMI", "ARI"):
            base = np.asarray([row["BASE"][metric] for row in selected], dtype=float)
            ours = np.asarray([row["OURS"][metric] for row in selected], dtype=float)
            delta = ours - base
            datasets[dataset][metric] = {
                "BASE_mean": float(base.mean()),
                "BASE_std_population": float(base.std()),
                "OURS_mean": float(ours.mean()),
                "OURS_std_population": float(ours.std()),
                "delta_mean": float(delta.mean()),
                "delta_std_population": float(delta.std()),
                "positive_delta_seed_count": int(np.count_nonzero(delta > 0)),
            }
    return {
        "status": "REPORTING_ONLY",
        "completed_run_count": len(rows),
        "intended_run_count": 24,
        "all_24_complete": len(rows) == 24,
        "formal_seeds": list(protocol.FORMAL_SEEDS),
        "parameter_retuning_allowed": False,
        "datasets": datasets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=protocol.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--write", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = summarize_final_formal(args.output_root, require_all_24=not args.allow_partial)
    if args.write:
        protocol.write_json(args.write, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

