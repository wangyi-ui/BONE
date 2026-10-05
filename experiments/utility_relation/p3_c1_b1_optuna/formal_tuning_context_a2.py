"""Formal context using A2 feasibility and R3 numeric fold ordering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import numpy as np

from . import amendment_a1, amendment_a2, protocol
from . import formal_tuning_context as frozen_context
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .formal_tuning_context_r3 import canonicalize_formal_tuning_context_r3


@dataclass(frozen=True)
class FormalTuningContextA2(FormalTuningContext):
    """Frozen-context-compatible carrier with explicit A2 provenance."""

    @property
    def active_protocol(self) -> str:
        return amendment_a2.AMENDMENT_ID

    @property
    def candidate_protocol(self) -> str:
        return amendment_a1.AMENDMENT_ID


def _a2_feasibility(carrier: CarrierContext) -> dict:
    arrays = carrier.arrays
    sample_ids = np.asarray(arrays["sample_ids"], dtype=np.int64)
    labeled_ids = np.asarray(arrays["labeled_ids"], dtype=np.int64)
    labeled_targets = np.asarray(arrays["labeled_targets"])
    positions = {int(raw_id): index for index, raw_id in enumerate(sample_ids)}
    try:
        masses = np.asarray([
            np.asarray(arrays["U_cycle"])[positions[int(raw_id)]].sum()
            for raw_id in labeled_ids
        ], dtype=float)
    except KeyError as error:
        raise ValueError("SPARSE_LABEL_ID_NOT_IN_CARRIER") from error
    return amendment_a2.validate_coverage_aware_feasibility(
        labeled_targets,
        labeled_ids,
        masses,
        expected_class_ids=range(carrier.k),
    )


def _candidate_protocol_fingerprint(
    dataset: str,
    carriers: dict[int, CarrierContext],
) -> str:
    """Retain the frozen candidate/cache identity; A2 changes no candidate."""
    payload = {
        "active_protocol": amendment_a1.AMENDMENT_ID,
        "dataset": dataset,
        "dev_seeds": list(protocol.DEV_SEEDS),
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "carrier_state_sha256": {
            str(seed): carriers[seed].state_file_sha256
            for seed in protocol.DEV_SEEDS
        },
        "carrier_pre_gt_sha256": {
            str(seed): carriers[seed].pre_gt_sha256
            for seed in protocol.DEV_SEEDS
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _build_uncanonicalized_context_a2(
    dataset: str,
    formal_tuning_root: str | Path,
    carrier_loader: Callable[[str, int, Path], CarrierContext] | None = None,
) -> FormalTuningContextA2:
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    root = Path(formal_tuning_root)
    loader = carrier_loader or frozen_context._carrier_context
    carriers = {
        seed: loader(dataset, seed, root) for seed in protocol.DEV_SEEDS
    }
    feasibility = {
        seed: _a2_feasibility(carriers[seed]) for seed in protocol.DEV_SEEDS
    }
    return FormalTuningContextA2(
        dataset=dataset,
        formal_tuning_root=root,
        carriers=MappingProxyType(carriers),
        feasibility=MappingProxyType(feasibility),
        protocol_fingerprint=_candidate_protocol_fingerprint(dataset, carriers),
    )


def build_formal_tuning_context_a2(
    dataset: str,
    formal_tuning_root: str | Path,
    carrier_loader: Callable[[str, int, Path], CarrierContext] | None = None,
) -> FormalTuningContextA2:
    """Load frozen carriers without invoking the superseded A1 feasibility."""
    base = _build_uncanonicalized_context_a2(
        dataset, formal_tuning_root, carrier_loader=carrier_loader
    )
    canonical, _ = canonicalize_formal_tuning_context_r3(base)
    return canonical


def build_formal_tuning_context_a2_with_audit(
    dataset: str,
    formal_tuning_root: str | Path,
    carrier_loader: Callable[[str, int, Path], CarrierContext] | None = None,
) -> tuple[FormalTuningContextA2, dict]:
    base = _build_uncanonicalized_context_a2(
        dataset, formal_tuning_root, carrier_loader=carrier_loader
    )
    context, canonicalization = canonicalize_formal_tuning_context_r3(base)
    coverage = amendment_a2.aggregate_dev_seed_coverage_audits(
        context.feasibility
    )
    return context, {
        "dataset": dataset,
        "amendment_id": amendment_a2.AMENDMENT_ID,
        "dev_seeds": dict(context.feasibility),
        "cross_seed_coverage": coverage,
        "numeric_canonicalization": canonicalization,
        "candidate_protocol_fingerprint": context.protocol_fingerprint,
        "full_gt_used": False,
    }


build_context = build_formal_tuning_context_a2
load_context = build_formal_tuning_context_a2
