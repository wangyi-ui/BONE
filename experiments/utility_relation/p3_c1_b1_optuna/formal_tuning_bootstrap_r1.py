"""BASE-reference bootstrap gate for the versioned P3-C1-B1-A1 runner."""

from __future__ import annotations

from typing import Callable, Mapping

from . import protocol
from .formal_base_reference import (
    BaseReference, build_base_reference, load_verified_base_reference,
)
from .formal_tuning_context import FormalTuningContext


def bootstrap_base_references(
    context: FormalTuningContext,
    device: str,
    replace_incomplete: bool = False,
    builder: Callable[..., BaseReference] = build_base_reference,
    verifier: Callable[..., BaseReference] = load_verified_base_reference,
) -> Mapping[int, BaseReference]:
    """Materialize and re-verify both BASE caches before study creation.

    The builder is called once per registered dev seed.  A verified cache is
    returned by the frozen implementation without retraining.  Any missing
    cache or builder/verifier exception propagates fail-closed; callers must
    create no Optuna study until this function returns successfully.
    """
    references: dict[int, BaseReference] = {}
    for dev_seed in protocol.DEV_SEEDS:
        references[dev_seed] = builder(
            context, dev_seed, device=device,
            replace_incomplete=replace_incomplete,
        )
    verified: dict[int, BaseReference] = {}
    for dev_seed in protocol.DEV_SEEDS:
        verified[dev_seed] = verifier(context, dev_seed)
    if set(verified) != set(protocol.DEV_SEEDS):
        raise RuntimeError("BASE_REFERENCE_SEED_COVERAGE_FAILURE")
    for dev_seed, reference in verified.items():
        if reference.dataset != context.dataset or reference.dev_seed != dev_seed:
            raise RuntimeError("BASE_REFERENCE_IDENTITY_FAILURE")
        if reference.native_epochs != protocol.REFINEMENT_HORIZON:
            raise RuntimeError("BASE_REFERENCE_NATIVE_HORIZON_FAILURE")
    return verified


def verify_bootstrap_complete(
    context: FormalTuningContext,
    references: Mapping[int, BaseReference],
) -> bool:
    """Pure gate predicate used by tests/audits; it never creates a study."""
    if set(references) != set(protocol.DEV_SEEDS):
        return False
    return all(
        reference.dataset == context.dataset
        and reference.dev_seed == seed
        and reference.native_epochs == protocol.REFINEMENT_HORIZON
        for seed, reference in references.items()
    )


bootstrap_formal_bases = bootstrap_base_references

