"""Read-only formal-carrier context for the P3-C1-B1-A1 main search.

The context loader validates already-materialized carriers and frozen sparse
folds.  It never trains a model, reads ground-truth labels, or creates an
Optuna study.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0

from . import amendment_a1, formal_carrier_materializer_r1, formal_msrc_carrier_adapter
from . import msrc_adapter, protocol, sparse_cv
from .parent_integrity import sha256_file


@dataclass(frozen=True)
class CarrierContext:
    dataset: str
    dev_seed: int
    directory: Path
    state_path: Path
    pre_gt_path: Path
    audit_path: Path
    arrays: Mapping[str, np.ndarray]
    views: tuple[np.ndarray, ...]
    adapter: object
    spec: object
    k: int
    folds: tuple[sparse_cv.SparseFold, sparse_cv.SparseFold]
    state_file_sha256: str
    pre_gt_sha256: str

    @property
    def full_gt_loaded(self) -> bool:
        return False


@dataclass(frozen=True)
class FormalTuningContext:
    dataset: str
    formal_tuning_root: Path
    carriers: Mapping[int, CarrierContext]
    feasibility: Mapping[int, dict]
    protocol_fingerprint: str

    @property
    def active_protocol(self) -> str:
        return amendment_a1.AMENDMENT_ID

    @property
    def full_gt_loaded(self) -> bool:
        return False

    def carrier(self, dev_seed: int) -> CarrierContext:
        if dev_seed not in self.carriers:
            raise ValueError("UNREGISTERED_DEV_SEED")
        return self.carriers[dev_seed]


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_views_and_adapter(dataset: str):
    """Load X views and raw-ID adapter without touching Y/full ground truth."""
    if dataset == "MSRC":
        views, labeled_ids, labeled_targets, _, _ = msrc_adapter._load_frozen_msrc_inputs()
        adapter = b0.SampleIdAdapter(np.arange(formal_msrc_carrier_adapter.MSRC_SPEC.n, dtype=np.int64))
        spec = formal_msrc_carrier_adapter.MSRC_SPEC
        expected_classes = range(msrc_adapter.p1_protocol.K)
    else:
        labeled_ids, labeled_targets, mask, _ = c0._load_frozen_inputs(dataset)
        raw_views = c0.load_raw_views(dataset)
        views, adapter = b0.preprocess_with_raw_ids(raw_views, mask)
        spec = c0_protocol.DATASET_SPECS[dataset]
        expected_classes = range(int(labeled_targets.max()) + 1)
    folds = sparse_cv.build_sparse_label_folds(
        labeled_ids, labeled_targets, expected_class_ids=expected_classes
    )
    return tuple(np.ascontiguousarray(view) for view in views), adapter, spec, folds


def _validate_carrier_payload(dataset: str, arrays: dict[str, np.ndarray]) -> None:
    required = set(formal_carrier_materializer_r1.frozen_materializer.REQUIRED_CARRIER_KEYS)
    if not required.issubset(arrays):
        raise ValueError(f"MISSING_CARRIER_KEYS:{sorted(required - set(arrays))}")
    if dataset == "MSRC":
        formal_msrc_carrier_adapter.validate_generated_arrays(arrays)
    else:
        # Reuse the frozen validator by writing nothing: call its pure checks
        # through a small temporary in-memory equivalent is not possible, so
        # duplicate only shape/provenance assertions already used by the
        # materializer and never any scientific computation.
        q = arrays["q_views"]
        spec = c0_protocol.DATASET_SPECS[dataset]
        if q.shape[:2] != (spec.n, spec.v) or q.ndim != 3:
            raise ValueError("INVALID_CARRIER_Q_SHAPE")
        if arrays["p_gen"].shape != arrays["p_ver"].shape or arrays["p_gen"].shape[0] != q.shape[0] or arrays["p_gen"].shape[2] != q.shape[2]:
            raise ValueError("INVALID_CARRIER_POSTERIOR_SHAPE")
        if arrays["U_cycle"].shape != arrays["p_gen"].shape[:2]:
            raise ValueError("INVALID_CARRIER_U_SHAPE")
        if arrays["p_gen"].shape[1] != action_protocol.expected_action_count(arrays["q_views"].shape[1]):
            raise ValueError("INVALID_CARRIER_ACTION_COUNT")
        if arrays["generator_actions"].shape[0] != arrays["p_gen"].shape[1] or arrays["verifier_actions"].shape[0] != arrays["p_gen"].shape[1]:
            raise ValueError("INVALID_CARRIER_ACTION_ARRAY")
        if not all(np.isfinite(arrays[name]).all() for name in ("q_views", "p_gen", "p_ver", "U_cycle")):
            raise ValueError("NONFINITE_CARRIER_ARRAY")
        y_gen, y_ver = np.argmax(arrays["p_gen"], axis=2), np.argmax(arrays["p_ver"], axis=2)
        conf_gen = np.max(arrays["p_gen"], axis=2)
        support_ver = np.take_along_axis(arrays["p_ver"], y_gen[..., None], axis=2)[..., 0]
        expected_u = (y_gen == y_ver) * np.sqrt(conf_gen * support_ver)
        if not np.allclose(arrays["U_cycle"], expected_u, rtol=1e-6, atol=1e-7):
            raise ValueError("U_FORMULA_MISMATCH")
    if arrays["sample_ids"].ndim != 1 or len(np.unique(arrays["sample_ids"])) != len(arrays["sample_ids"]):
        raise ValueError("INVALID_SAMPLE_IDS")
    targets = np.asarray(arrays["labeled_targets"])
    if arrays["labeled_ids"].ndim != 1 or targets.shape != arrays["labeled_ids"].shape:
        raise ValueError("INVALID_SPARSE_LABEL_SHAPE")
    unique_targets, counts = np.unique(targets, return_counts=True)
    if len(unique_targets) != int(np.asarray(arrays["q_views"]).shape[2]) or not np.array_equal(unique_targets, np.arange(len(unique_targets))) or np.any(counts != 2):
        raise ValueError("SPARSE_LABELS_MUST_BE_EXACTLY_TWO_PER_CLASS")


def _feasibility(arrays: Mapping[str, np.ndarray]) -> dict:
    sample_ids = np.asarray(arrays["sample_ids"], dtype=np.int64)
    labeled_ids = np.asarray(arrays["labeled_ids"], dtype=np.int64)
    labeled_targets = np.asarray(arrays["labeled_targets"], dtype=np.int64)
    positions = {int(raw): index for index, raw in enumerate(sample_ids.tolist())}
    try:
        masses = np.asarray([
            np.asarray(arrays["U_cycle"])[positions[int(raw)]].sum()
            for raw in labeled_ids
        ], dtype=float)
    except KeyError as error:
        raise ValueError("SPARSE_LABEL_ID_NOT_IN_CARRIER") from error
    # This is the preregistered A1 gate; zero/zero raises fail-closed.
    return amendment_a1.validate_class_feasibility(
        labeled_targets, masses, expected_class_ids=range(int(labeled_targets.max()) + 1)
    )


def _carrier_context(dataset: str, seed: int, root: Path) -> CarrierContext:
    directory = formal_carrier_materializer_r1.frozen_materializer.carrier_directory(root, dataset, seed)
    verified = formal_carrier_materializer_r1.verify_complete_carrier(directory, dataset, seed)
    marker = verified["marker"]
    audit = verified["audit"]
    if audit.get("full_gt_loaded") is not False or audit.get("full_gt_used") is not False:
        raise ValueError("FULL_GT_CARRIER_CONTEXT_FORBIDDEN")
    state_path = directory / "carrier_model_state.pt"
    pre_gt_path = directory / "carrier_pre_gt.npz"
    audit_path = directory / "carrier_audit.json"
    arrays = _load_arrays(pre_gt_path)
    _validate_carrier_payload(dataset, arrays)
    views, adapter, spec, folds = _load_views_and_adapter(dataset)
    if len(views) != int(spec.v) or views[0].shape[0] != int(spec.n):
        raise ValueError("CARRIER_VIEW_CONTEXT_MISMATCH")
    if not np.array_equal(np.sort(np.asarray(arrays["sample_ids"])), np.arange(int(spec.n))):
        raise ValueError("CARRIER_SAMPLE_ID_SET_MISMATCH")
    k = int(np.asarray(arrays["q_views"]).shape[2])
    if k != int(np.asarray(arrays["labeled_targets"]).max()) + 1:
        raise ValueError("CARRIER_CLASS_COUNT_MISMATCH")
    return CarrierContext(
        dataset=dataset, dev_seed=seed, directory=directory,
        state_path=state_path, pre_gt_path=pre_gt_path, audit_path=audit_path,
        arrays=arrays, views=views, adapter=adapter, spec=spec, k=k, folds=folds,
        state_file_sha256=sha256_file(state_path), pre_gt_sha256=sha256_file(pre_gt_path),
    )


def build_formal_tuning_context(
    dataset: str,
    formal_tuning_root: str | Path,
) -> FormalTuningContext:
    """Validate both immutable dev carriers and construct a read-only context."""
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    root = Path(formal_tuning_root)
    carriers = {seed: _carrier_context(dataset, seed, root) for seed in protocol.DEV_SEEDS}
    feasibility = {seed: _feasibility(carriers[seed].arrays) for seed in protocol.DEV_SEEDS}
    fingerprint_payload = {
        "active_protocol": amendment_a1.AMENDMENT_ID,
        "dataset": dataset,
        "dev_seeds": list(protocol.DEV_SEEDS),
        "carrier_epochs": protocol.CARRIER_EPOCHS,
        "refinement_horizon": protocol.REFINEMENT_HORIZON,
        "carrier_state_sha256": {str(seed): carriers[seed].state_file_sha256 for seed in protocol.DEV_SEEDS},
        "carrier_pre_gt_sha256": {str(seed): carriers[seed].pre_gt_sha256 for seed in protocol.DEV_SEEDS},
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
    return FormalTuningContext(
        dataset=dataset, formal_tuning_root=root, carriers=carriers,
        feasibility=feasibility, protocol_fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
    )


# Short aliases make the API explicit for CLI/tests without introducing a
# second implementation.
load_formal_tuning_context = build_formal_tuning_context
build_context = build_formal_tuning_context
load_context = build_formal_tuning_context
validate_a1_feasibility = _feasibility
