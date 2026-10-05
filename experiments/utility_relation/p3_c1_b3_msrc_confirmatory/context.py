"""Pre-GT-only input resolution and readiness checks for the MSRC confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.utility_relation.p3_c1_b1_final_formal import (
    final_formal_context as final_context,
)

from . import protocol


ResolvedInputs = final_context.ResolvedInputs


def resolve_training_inputs(dataset: str = protocol.DATASET) -> ResolvedInputs:
    if dataset != protocol.DATASET:
        raise ValueError("B3_EXACT_DATASET_MSRC_ONLY")
    resolved = final_context.resolve_training_inputs(dataset)
    if resolved.dataset != protocol.DATASET:
        raise RuntimeError("B3_MSRC_ROUTE_IDENTITY_MISMATCH")
    if resolved.route != protocol.final_protocol.MSRC_ROUTE:
        raise RuntimeError("B3_MSRC_ROUTE_DRIFT")
    return resolved


def validate_carrier(resolved: ResolvedInputs, arrays: dict[str, np.ndarray]) -> dict:
    return final_context.validate_a2_for_carrier(resolved, arrays)


def preflight(require_preregistration: bool = False) -> dict:
    parent = protocol.verify_parent_integrity()
    resolved = resolve_training_inputs()
    paths = final_context.frozen_input_paths(protocol.DATASET)
    input_hashes = {name: protocol.sha256_file(path) for name, path in paths.items()}
    preregistration = (
        protocol.verify_preregistration() if require_preregistration else None
    )
    return {
        "STATUS": "PASS",
        "DATASET": protocol.DATASET,
        "SEEDS": list(protocol.CONFIRMATORY_SEEDS),
        "ARMS": list(protocol.ARMS),
        "RUN_COUNT": len(protocol.CONFIRMATORY_SEEDS) * len(protocol.ARMS),
        "CARRIER_COUNT": len(protocol.CONFIRMATORY_SEEDS),
        "CARRIER_EPOCHS": protocol.CARRIER_EPOCHS,
        "REFINEMENT_EPOCHS": protocol.REFINEMENT_EPOCHS,
        "FULL_GT_READ": False,
        "FORMAL_20_30_50_RESULTS_READ": False,
        "REAL_CONFIRMATORY_RUN": False,
        "MSRC_ROUTE": resolved.route,
        "INPUT_HASHES": input_hashes,
        "PARENT_INTEGRITY": parent,
        "PREREGISTRATION": preregistration,
    }
