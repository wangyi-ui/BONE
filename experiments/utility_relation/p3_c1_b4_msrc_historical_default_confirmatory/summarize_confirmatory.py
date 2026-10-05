"""Preregistered B4 reporting summary and fixed primary gate."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from . import protocol


def summarize_confirmatory(output_root: str | Path = protocol.OUTPUT_ROOT) -> dict:
    rows = []
    for seed in protocol.CONFIRMATORY_SEEDS:
        path = protocol.run_root(output_root, seed) / "CONFIRMATORY_METRICS.json"
        if not path.is_file(): raise FileNotFoundError(f"B4_METRICS_MISSING:seed{seed}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("metrics_used_for_parameter_selection") is not False: raise RuntimeError("B4_REPORTING_SELECTION_VIOLATION")
        rows.append(row)
    means, positive = {}, {}
    for metric in ("ACC", "NMI", "ARI"):
        values = np.asarray([row["delta_historical_default_minus_base"][metric] for row in rows], dtype=float)
        means[metric] = {"mean": float(values.mean()), "std_population": float(values.std(ddof=0))}
        positive[metric] = int(np.count_nonzero(values > 0))
    return {"status": "REPORTING_ONLY", "dataset": protocol.DATASET, "seeds": list(protocol.CONFIRMATORY_SEEDS), "per_seed": rows, "mean_deltas": means, "positive_seed_counts": positive, "HISTORICAL_DEFAULT_CONFIRMATION": protocol.primary_confirmation_gate({key: value["mean"] for key, value in means.items()}), "primary_gate": "mean_ACC>0 AND mean_NMI>0 AND mean_ARI>0", "parameter_retuning_allowed": False}
