"""B5-owned candidate schema, separate from the historical B1 LR registry.

The frozen evaluator consumes only ``semantic_lr`` and ``semantic_exposure``
while executing a branch.  This module owns the B5 protocol decision of
whether those runtime values (and optimizer family) are registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from . import protocol


def _canonical_lr(value: float | str | Decimal) -> str:
    """Return a non-tolerant decimal identity for a registered LR."""
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("B5_UNREGISTERED_SEMANTIC_LR") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError("B5_UNREGISTERED_SEMANTIC_LR")
    return str(decimal.normalize())


_REGISTERED_LRS = {
    (dataset, optimizer): frozenset(_canonical_lr(lr) for lr in lrs)
    for dataset, optimizer_grid in protocol.LR_GRID.items()
    for optimizer, lrs in optimizer_grid.items()
}


@dataclass(frozen=True)
class B5CandidateConfig:
    """Immutable B5 runtime config accepted by the frozen branch executor."""

    dataset: str
    optimizer_name: str
    semantic_lr: float
    semantic_exposure: int
    execution_mode: str

    @property
    def runtime_fields(self) -> tuple[float, int]:
        return (self.semantic_lr, self.semantic_exposure)


def validate_b5_candidate(candidate: protocol.Candidate) -> B5CandidateConfig:
    """Fail closed against exactly the preregistered B5 candidate grid.

    Decimal identities are compared exactly; no floating-point tolerance can
    admit a nearby unregistered learning rate.
    """
    if candidate.dataset not in protocol.DATASETS:
        raise ValueError("B5_UNREGISTERED_DATASET")
    if candidate.optimizer_name not in protocol.OPTIMIZERS:
        raise ValueError("B5_UNREGISTERED_OPTIMIZER")
    if candidate.semantic_exposure != protocol.EXPOSURES[candidate.dataset]:
        raise ValueError("B5_UNREGISTERED_SEMANTIC_EXPOSURE")
    registered = _REGISTERED_LRS[(candidate.dataset, candidate.optimizer_name)]
    if _canonical_lr(candidate.semantic_lr) not in registered:
        raise ValueError("B5_UNREGISTERED_SEMANTIC_LR")
    return B5CandidateConfig(
        dataset=candidate.dataset,
        optimizer_name=candidate.optimizer_name,
        semantic_lr=float(candidate.semantic_lr),
        semantic_exposure=int(candidate.semantic_exposure),
        execution_mode="REUSE_ONLY" if candidate.optimizer_name == "Adam" else "REAL_TRAINING",
    )


def validate_real_training_candidate(candidate: protocol.Candidate) -> B5CandidateConfig:
    config = validate_b5_candidate(candidate)
    if config.execution_mode != "REAL_TRAINING":
        raise RuntimeError("B5_ADAM_CANDIDATE_REUSE_ONLY")
    return config
