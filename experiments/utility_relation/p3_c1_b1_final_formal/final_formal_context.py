"""Pre-GT input resolution and read-only 24-run readiness audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as c0
from experiments.utility_relation.p3_c1_b1_optuna import amendment_a2, msrc_adapter

from . import final_formal_protocol as protocol


@dataclass(frozen=True)
class ResolvedInputs:
    dataset: str
    route: str
    views: tuple[np.ndarray, ...]
    adapter: b0.SampleIdAdapter
    spec: object
    k: int
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    input_hashes: dict[str, str]
    r3_audit: dict


def frozen_input_paths(dataset: str) -> dict[str, Path]:
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_FINAL_FORMAL_DATASET")
    if dataset == "MSRC":
        return {
            "data": protocol.REPO_ROOT / "data" / "MSRC.mat",
            "sparse_split": protocol.REPO_ROOT / "outputs/bone_p3/sparse_splits/MSRC_2perclass.npz",
            "corruption_mask": protocol.REPO_ROOT / "outputs/bone_p3/corruptions/MSRC_mask.npz",
            "legacy_corruption": msrc_adapter.p3_a4_materializer.P1_ARTIFACT,
        }
    split, mask = c0_protocol.artifact_paths(dataset)
    return {
        "data": c0_protocol.DATA_ROOT / f"{dataset}.mat",
        "sparse_split": split,
        "corruption_mask": mask,
    }


def _canonicalize_full_sparse_labels(
    labeled_ids: np.ndarray, labeled_targets: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    ids = np.asarray(labeled_ids, dtype=np.int64)
    targets = np.asarray(labeled_targets, dtype=np.int64)
    if ids.shape != targets.shape or ids.ndim != 1 or ids.size != 2 * k:
        raise ValueError("INVALID_FINAL_SPARSE_LABEL_SHAPE")
    if not np.array_equal(np.bincount(targets, minlength=k), np.full(k, 2)):
        raise ValueError("INVALID_FINAL_SPARSE_CLASS_COVERAGE")
    before_pairs = sorted((int(t), int(i)) for i, t in zip(ids, targets))
    order = np.lexsort((ids, targets))
    canonical_ids, canonical_targets = ids[order], targets[order]
    after_pairs = [(int(t), int(i)) for i, t in zip(canonical_ids, canonical_targets)]
    encoded = repr(before_pairs).encode("utf-8")
    pair_hash = hashlib.sha256(encoded).hexdigest()
    if before_pairs != after_pairs:
        raise RuntimeError("R3_FULL_LABEL_CANONICALIZATION_FAILURE")
    return canonical_ids, canonical_targets, {
        "R3_NUMERIC_CANONICALIZATION_PRESERVED": True,
        "ordering_changed": not np.array_equal(order, np.arange(ids.size)),
        "pair_set_sha256": pair_hash,
        "anchor_membership_changed": False,
        "label_values_changed": False,
    }


def resolve_training_inputs(dataset: str) -> ResolvedInputs:
    """Load features, sparse labels and corruption only; never request full Y."""
    paths = frozen_input_paths(dataset)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"FROZEN_INPUT_MISSING:{dataset}:{name}")
    if dataset == "MSRC":
        views, ids, targets, _split, _mask = msrc_adapter._load_frozen_msrc_inputs()
        spec = msrc_adapter.MSRC_SPEC
        adapter = b0.SampleIdAdapter(np.arange(spec.n, dtype=np.int64))
        route = protocol.MSRC_ROUTE
        k = int(msrc_adapter.p1_protocol.K)
    else:
        ids, targets, mask, _ = c0._load_frozen_inputs(dataset)
        raw_views = c0.load_raw_views(dataset)
        views, adapter = b0.preprocess_with_raw_ids(raw_views, mask)
        spec = c0_protocol.DATASET_SPECS[dataset]
        route = protocol.GENERIC_ROUTE
        k = int(np.asarray(targets).max()) + 1
    canonical_ids, canonical_targets, r3_audit = _canonicalize_full_sparse_labels(
        ids, targets, k
    )
    if not np.array_equal(np.sort(adapter.sample_ids), np.arange(spec.n)):
        raise RuntimeError("RAW_SAMPLE_ID_SET_MISMATCH")
    return ResolvedInputs(
        dataset=dataset,
        route=route,
        views=tuple(views),
        adapter=adapter,
        spec=spec,
        k=k,
        labeled_ids=canonical_ids,
        labeled_targets=canonical_targets,
        input_hashes={name: protocol.sha256_file(path) for name, path in paths.items()},
        r3_audit=r3_audit,
    )


def validate_a2_for_carrier(resolved: ResolvedInputs, arrays: dict[str, np.ndarray]) -> dict:
    positions = resolved.adapter.row_positions(resolved.labeled_ids)
    masses = np.asarray(arrays["U_cycle"])[positions].sum(axis=1)
    return amendment_a2.validate_coverage_aware_feasibility(
        resolved.labeled_targets,
        resolved.labeled_ids,
        masses,
        expected_class_ids=range(resolved.k),
    )


def preflight_all_24(freeze_dir: str | Path = protocol.DEFAULT_FREEZE) -> dict:
    bindings = protocol.load_final_parameter_bindings(freeze_dir)
    parent_hashes = protocol.verify_parent_sources()
    runs, unresolved = [], []
    for dataset in protocol.DATASETS:
        paths = frozen_input_paths(dataset)
        missing = [name for name, path in paths.items() if not path.is_file()]
        validation_error = None
        if not missing:
            try:
                if dataset == "MSRC":
                    views, ids, targets, _split, _mask = msrc_adapter._load_frozen_msrc_inputs()
                    if (
                        len(views) != msrc_adapter.MSRC_SPEC.v
                        or any(view.shape[0] != msrc_adapter.MSRC_SPEC.n for view in views)
                    ):
                        raise ValueError("MSRC_VIEW_SHAPE_MISMATCH")
                    k = int(msrc_adapter.p1_protocol.K)
                else:
                    ids, targets, mask, _ = c0._load_frozen_inputs(dataset)
                    spec = c0_protocol.DATASET_SPECS[dataset]
                    if mask.shape != (spec.n, spec.v):
                        raise ValueError("GENERIC_MASK_SHAPE_MISMATCH")
                    k = int(np.asarray(targets).max()) + 1
                _canonicalize_full_sparse_labels(ids, targets, k)
            except Exception as error:  # Serialized as an audit failure, never ignored.
                validation_error = f"{type(error).__name__}:{error}"
        unresolved_reasons = [*missing]
        if validation_error:
            unresolved_reasons.append(validation_error)
        for seed in protocol.FORMAL_SEEDS:
            binding = bindings[dataset]
            item = {
                "dataset": dataset,
                "training_seed": seed,
                "semantic_lr": binding.semantic_lr,
                "semantic_exposure": binding.semantic_exposure,
                "final_parameter_freeze_sha256": binding.final_parameter_freeze_sha256,
                "dataset_adapter": protocol.MSRC_ROUTE if dataset == "MSRC" else protocol.GENERIC_ROUTE,
                "corruption_protocol": "frozen seed20 balanced-half-view SNR=2.5dB",
                "required_frozen_dependencies": {
                    name: str(path.relative_to(protocol.REPO_ROOT)) for name, path in paths.items()
                },
                "parent_source_count": len(parent_hashes),
                "adapter_validation": "PASS" if not validation_error else "FAIL",
                "resolvable": not unresolved_reasons,
                "unresolved": unresolved_reasons,
            }
            runs.append(item)
            if unresolved_reasons:
                unresolved.append(item)
    return {
        "STATUS": "PASS" if not unresolved else "FAIL",
        "INTENDED_RUN_COUNT": len(protocol.DATASETS) * len(protocol.FORMAL_SEEDS),
        "RESOLVABLE_RUN_COUNT": len(runs) - len(unresolved),
        "UNRESOLVED_RUN_COUNT": len(unresolved),
        "ALL_24_FORMAL_RUNS_READY": len(runs) == 24 and not unresolved,
        "FORMAL_SEEDS": list(protocol.FORMAL_SEEDS),
        "DATASETS": list(protocol.DATASETS),
        "MSRC_ROUTE": protocol.MSRC_ROUTE,
        "REAL_FORMAL_RUN": False,
        "FULL_GT_READ": False,
        "runs": runs,
    }
