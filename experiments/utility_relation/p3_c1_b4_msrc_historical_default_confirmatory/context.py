"""MSRC-only pre-GT input binding for B4."""

from __future__ import annotations

from experiments.utility_relation.p3_c1_b1_final_formal import final_formal_context

from . import audit, protocol


ResolvedInputs = final_formal_context.ResolvedInputs


def resolve_training_inputs(dataset: str = protocol.DATASET) -> ResolvedInputs:
    if dataset != protocol.DATASET:
        raise ValueError("B4_EXACT_DATASET_MSRC_ONLY")
    resolved = final_formal_context.resolve_training_inputs(dataset)
    if resolved.spec.batch_size != protocol.NATIVE_BATCH_SIZE:
        raise RuntimeError("B4_HISTORICAL_NATIVE_BATCH_SIZE_MISMATCH")
    return resolved


def preflight() -> dict:
    parent = audit.verify_parent_integrity()
    resolved = resolve_training_inputs()
    return {
        "STATUS": "PASS", "DATASET": protocol.DATASET,
        "SEEDS": list(protocol.CONFIRMATORY_SEEDS), "ARMS": list(protocol.ARMS),
        "CARRIER_EPOCHS": protocol.CARRIER_EPOCHS,
        "REFINEMENT_EPOCHS": protocol.REFINEMENT_EPOCHS,
        "HISTORICAL_DEFAULT_SEMANTIC_LR": protocol.HISTORICAL_DEFAULT_SEMANTIC_LR,
        "HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE": protocol.HISTORICAL_DEFAULT_SEMANTIC_EXPOSURE,
        "FULL_GT_READ": False, "REAL_CONFIRMATORY_RUN": False,
        "MSRC_ROUTE": resolved.route, "PARENT_INTEGRITY": parent,
    }
