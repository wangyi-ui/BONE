"""Exact preregistered P3-C4-A0 aggregation gates (no clustering metrics)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from .protocol import DATASETS, SEEDS, manifest_cells


def _positive(values: list[float | None]) -> int:
    return sum(value is not None and value > 0 for value in values)


def summarize(cells: list[dict]) -> dict:
    expected = set(manifest_cells())
    found = {(item.get("dataset"), item.get("training_seed")) for item in cells}
    if found != expected or len(cells) != 24: raise ValueError("EXACT_8_X_3_CELL_MANIFEST_REQUIRED")
    names = ("Delta_TRUE_UNIFORM", "Delta_TRUE_SHUFFLE", "DeltaAlignment_Q5_Q1", "Spearman_quintile_alignment", "Spearman_U_semantic_margin")
    values = {name: [item.get(name) for item in cells] for name in names}
    if any(any(value is None for value in group) for group in values.values()): raise ValueError("INVALID_ALIGNMENT_CELL_CANNOT_BE_SUMMARIZED")
    macro = {name: fmean(group) for name, group in values.items()}
    by_dataset = {dataset: [item for item in cells if item["dataset"] == dataset] for dataset in DATASETS}
    ds_mean = {name: {dataset: fmean([item[name] for item in group]) for dataset, group in by_dataset.items()} for name in names}
    relative = (macro["Delta_TRUE_UNIFORM"] > 0 and macro["Delta_TRUE_SHUFFLE"] > 0 and
                _positive(values["Delta_TRUE_UNIFORM"]) >= 16 and _positive(values["Delta_TRUE_SHUFFLE"]) >= 16 and
                _positive(list(ds_mean["Delta_TRUE_UNIFORM"].values())) >= 6 and
                _positive(list(ds_mean["Delta_TRUE_SHUFFLE"].values())) >= 6)
    sharpening = relative and macro["DeltaAlignment_Q5_Q1"] > 0 and _positive(values["DeltaAlignment_Q5_Q1"]) >= 16 and macro["Spearman_quintile_alignment"] > 0
    coupling = macro["Spearman_U_semantic_margin"] > 0 and _positive(values["Spearman_U_semantic_margin"]) >= 16
    route = ("RELATIVE_UTILITY_MAPPING_REFINEMENT_ALLOWED" if relative and sharpening else
             "KEEP_CURRENT_RELATIVE_U_NO_SHARPENING" if relative else "REVISIT_U_CYCLE_RELIABILITY")
    return {"cell_count": len(cells), "macro_mean": macro, "dataset_mean": ds_mean,
            "positive_cell_count": {name: _positive(group) for name, group in values.items()},
            "RELATIVE_U_ALIGNMENT_SUPPORTED": relative, "SHARPENING_ROUTE_AUTHORIZED": sharpening,
            "SEMANTIC_DIRECTION_COUPLING_SUPPORTED": coupling, "T_REL_U_COUPLING_SUSPECT": not coupling,
            "ROUTE": route, "gamma_prohibited": not relative or not sharpening}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.cells_json.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
