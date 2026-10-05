"""Read-only R3 context copy with numerically canonical sparse folds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from . import formal_tuning_context as frozen_context
from .canonical_fold_order_r3 import canonicalize_sparse_folds
from .formal_tuning_context import FormalTuningContext


def canonicalize_formal_tuning_context_r3(
    context: FormalTuningContext,
) -> tuple[FormalTuningContext, dict]:
    """Copy a verified frozen context and replace only carrier fold ordering."""
    carriers = {}
    carrier_audits = {}
    for dev_seed, carrier in context.carriers.items():
        folds, fold_audits = canonicalize_sparse_folds(carrier.folds, carrier.k)
        carriers[dev_seed] = replace(carrier, folds=folds)
        carrier_audits[str(dev_seed)] = {
            "dataset": context.dataset,
            "dev_seed": dev_seed,
            "K": carrier.k,
            "folds": fold_audits,
            "arrays_identity_preserved": carriers[dev_seed].arrays is carrier.arrays,
            "views_identity_preserved": carriers[dev_seed].views is carrier.views,
            "state_path_unchanged": carriers[dev_seed].state_path == carrier.state_path,
            "pre_gt_path_unchanged": carriers[dev_seed].pre_gt_path == carrier.pre_gt_path,
        }
    copied = replace(
        context,
        carriers=MappingProxyType(carriers),
        feasibility=MappingProxyType(dict(context.feasibility)),
    )
    audit = {
        "dataset": context.dataset,
        "protocol_fingerprint_unchanged": (
            copied.protocol_fingerprint == context.protocol_fingerprint
        ),
        "full_gt_used": False,
        "carriers": carrier_audits,
    }
    return copied, audit


def build_formal_tuning_context_r3(
    dataset: str,
    formal_tuning_root: str | Path,
    builder: Callable[..., FormalTuningContext] | None = None,
) -> FormalTuningContext:
    base = (builder or frozen_context.build_formal_tuning_context)(
        dataset, formal_tuning_root
    )
    canonical, _ = canonicalize_formal_tuning_context_r3(base)
    return canonical


def build_formal_tuning_context_r3_with_audit(
    dataset: str,
    formal_tuning_root: str | Path,
    builder: Callable[..., FormalTuningContext] | None = None,
) -> tuple[FormalTuningContext, dict]:
    base = (builder or frozen_context.build_formal_tuning_context)(
        dataset, formal_tuning_root
    )
    return canonicalize_formal_tuning_context_r3(base)


build_context = build_formal_tuning_context_r3
load_context = build_formal_tuning_context_r3

