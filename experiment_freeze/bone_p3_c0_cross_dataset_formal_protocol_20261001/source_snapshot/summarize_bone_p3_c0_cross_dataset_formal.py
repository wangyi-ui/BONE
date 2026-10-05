"""One-dataset P3-C0 formal-result summarizer; it never selects a winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as protocol


METRICS = ("ACC", "NMI", "Purity", "ARI")


def summarize_dataset(dataset_name: str, output_root: Path = protocol.OUTPUT_ROOT) -> dict[str, Any]:
    """Aggregate exactly seeds 20/30/50 for one dataset, without any performance gate."""
    if dataset_name not in protocol.FORMAL_DATASETS: raise ValueError("unsupported dataset")
    records = []
    for seed in protocol.FORMAL_SEEDS:
        path = Path(output_root) / protocol.dataset_slug(dataset_name) / f"seed{seed}" / "formal_metrics.json"
        if not path.is_file(): raise FileNotFoundError(f"missing formal metrics for {dataset_name} seed{seed}")
        record = json.loads(path.read_text())
        if record.get("dataset") != dataset_name or record.get("training_seed") != seed: raise ValueError("metric dataset/seed mismatch")
        records.append(record)
    summary: dict[str, Any] = {"dataset": dataset_name, "seeds": list(protocol.FORMAL_SEEDS), "aggregation": "mean/std population over fixed seeds; no selection gate", "metrics": {}}
    for metric in METRICS:
        base = np.asarray([record["BASE"][metric] for record in records], dtype=float)
        ours = np.asarray([record["OURS"][metric] for record in records], dtype=float)
        delta = ours - base
        summary["metrics"][metric] = {"BASE": {"mean": float(base.mean()), "std": float(base.std())},
                                      "OURS": {"mean": float(ours.mean()), "std": float(ours.std())},
                                      "delta_OURS_minus_BASE": {"mean": float(delta.mean()), "std": float(delta.std()), "positive_delta_seed_count": int((delta > 0).sum()), "out_of": 3}}
    root = Path(output_root) / protocol.dataset_slug(dataset_name); (root / "formal_dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset", choices=protocol.FORMAL_DATASETS, required=True)
    parser.add_argument("--output-root", type=Path, default=protocol.OUTPUT_ROOT); parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute: raise SystemExit("refusing to summarize without --execute")
    print(json.dumps(summarize_dataset(args.dataset, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__": main()
