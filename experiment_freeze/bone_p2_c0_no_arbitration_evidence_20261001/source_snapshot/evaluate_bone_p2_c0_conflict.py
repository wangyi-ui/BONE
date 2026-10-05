"""Read-only preregistered gate for a sealed P2-C0 diagnostic output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np

from experiments.utility_relation.bone_p2_c0_conflict_protocol import DEFAULT_OUTPUT_DIR, EPOCHS


def diagnostic_verdict(rows: list[Dict[str, float]]) -> str:
    """Apply the fixed P2-C0 arbitration gate to exactly twenty epoch rows."""
    if len(rows) != EPOCHS:
        raise ValueError("P2-C0 gate requires exactly 20 epoch rows")
    global_conflicts = sum(row["global_inter_conflict"] for row in rows)
    high_better = sum(row["rho_HIGH"] > row["rho_LOW"] for row in rows)
    mean_gap = float(np.mean([row["high_minus_low"] for row in rows]))
    conflict_gap = float(np.mean([row["low_conflict"] for row in rows]) - np.mean([row["high_conflict"] for row in rows]))
    if global_conflicts >= 5 and high_better >= 15 and mean_gap >= .05 and conflict_gap >= .15:
        return "P2_C0_SUPPORTS_ARBITRATION"
    if global_conflicts < 2 or mean_gap <= 0:
        return "P2_C0_NO_ARBITRATION_EVIDENCE"
    return "P2_C0_INCONCLUSIVE"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_diagnostic(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    """Verify the pre-GT diagnostic seal then apply its GT-free gate; no labels are loaded."""
    output_dir = Path(output_dir)
    metrics_path, seal_path = output_dir / "p2_c0_conflict_metrics.json", output_dir / "p2_c0_conflict_seal.json"
    if not metrics_path.is_file() or not seal_path.is_file():
        raise ValueError("missing P2-C0 diagnostic output or seal")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("sha256", {}).get("metrics") != sha256_file(metrics_path):
        raise ValueError("invalid P2-C0 diagnostic seal")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("epochs")
    if not isinstance(rows, list):
        raise ValueError("invalid P2-C0 epoch metrics")
    payload["verdict"] = diagnostic_verdict(rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="verify a sealed P2-C0 diagnostic and apply its GT-free gate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.execute:
        raise SystemExit("refusing to evaluate P2-C0 diagnostic without --execute")
    print(json.dumps(evaluate_diagnostic(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
