"""R1 routing amendment composed over the frozen formal materializer."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

from . import formal_carrier_materializer as frozen_materializer
from . import formal_msrc_carrier_adapter, protocol
from .parent_integrity import sha256_file


CHANGE_TYPE = "IMPLEMENTATION_ROUTING_ONLY"
SCIENTIFIC_PROTOCOL_CHANGED = False
U_CYCLE_CHANGED = False
T_REL_CHANGED = False
A1_CHANGED = False
SPARSE_LABELS_CHANGED = False
SEARCH_SPACE_CHANGED = False
DEV_SEEDS_CHANGED = False
CARRIER_EPOCHS_CHANGED = False
_PATCH_LOCK = threading.Lock()


def _invoke_adapter_r1(dataset: str, dev_seed: int, device: str, staging: Path):
    if dataset == "MSRC":
        return formal_msrc_carrier_adapter.build_formal_msrc_development_carrier(
            dev_seed, device, staging
        )
    return _FROZEN_INVOKE(dataset, dev_seed, device, staging)


_FROZEN_INVOKE = frozen_materializer._invoke_frozen_adapter
_FROZEN_PROVENANCE = frozen_materializer._source_provenance


def _source_provenance_r1(dataset: str) -> dict[str, str]:
    provenance = _FROZEN_PROVENANCE(dataset)
    if dataset == "MSRC":
        provenance = dict(provenance)
        provenance["formal_msrc_adapter_source"] = sha256_file(
            Path(formal_msrc_carrier_adapter.__file__)
        )
    return provenance


@contextmanager
def _r1_route_context():
    """Patch only frozen extension points during one serialized invocation."""
    with _PATCH_LOCK:
        before_invoke = frozen_materializer._invoke_frozen_adapter
        before_provenance = frozen_materializer._source_provenance
        frozen_materializer._invoke_frozen_adapter = _invoke_adapter_r1
        frozen_materializer._source_provenance = _source_provenance_r1
        try:
            yield
        finally:
            frozen_materializer._invoke_frozen_adapter = before_invoke
            frozen_materializer._source_provenance = before_provenance


def materialize_one_r1(
    dataset: str,
    dev_seed: int,
    device: str,
    output_root: str | Path,
    replace_incomplete: bool = False,
) -> dict:
    """Delegate all frozen lifecycle semantics, replacing only the MSRC route."""
    if dataset not in protocol.DATASETS or dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("INVALID_R1_FORMAL_CARRIER_IDENTITY")
    with _r1_route_context():
        return frozen_materializer.materialize_one(
            dataset, dev_seed, device, output_root, replace_incomplete
        )


def verify_complete_carrier(path: str | Path, dataset: str, dev_seed: int) -> dict:
    return frozen_materializer.verify_complete_carrier(path, dataset, dev_seed)


def expected_identities() -> tuple[tuple[str, int], ...]:
    return frozen_materializer.expected_identities()
