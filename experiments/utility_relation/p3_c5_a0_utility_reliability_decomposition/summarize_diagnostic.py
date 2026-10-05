"""Fail-closed aggregation and deterministic route selection for C5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decomposition import aggregate_component_gates, route_decision, validate_cell
from .protocol import DATASETS, GAMMA_PROHIBITED, manifest_cells


def summarize(cells: list[dict]) -> dict:
    expected = set(manifest_cells())
    found = [(item.get("dataset"), item.get("training_seed")) for item in cells]
    if len(cells) != 24 or set(found) != expected or len(set(found)) != len(found):
        raise ValueError("EXACT_8_X_3_CELL_MANIFEST_REQUIRED")
    for item in cells:
        if item.get("stage") != "P3-C5-A0" or item.get("dataset") not in DATASETS:
            raise ValueError("WRONG_C5_CELL_IDENTITY")
        validate_cell(item)
    gates = aggregate_component_gates(cells)
    route = route_decision(gates["CLOSURE_STRUCTURAL_SUPPORT"], gates["CONF_GEN_STRUCTURAL_SUPPORT"], gates["SUPPORT_VER_STRUCTURAL_SUPPORT"])
    return {"cell_count": len(cells), "component_gates": gates, **route,
            "gamma_prohibited": GAMMA_PROHIBITED, "SHARPENING_ROUTE_AUTHORIZED": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-json", type=Path, required=True); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); result = summarize(json.loads(args.cells_json.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
