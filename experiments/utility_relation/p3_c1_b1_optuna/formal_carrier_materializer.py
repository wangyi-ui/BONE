"""Atomic one-at-a-time formal development carrier orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import bone_p3_c0_cross_dataset_formal_protocol as c0_protocol
from experiments.utility_relation import materialize_bone_p2_a0_carrier as p2_carrier
from experiments.utility_relation import materialize_bone_p3_b0_carrier as b0_materializer

from . import amendment_a1, carrier_adapter, msrc_adapter, protocol
from .parent_integrity import sha256_file


ACTIVE_PROTOCOL = amendment_a1.AMENDMENT_ID
FORMAL_OUTPUT_SUBDIR = Path("outputs/bone_p3/p3_c1_b1_optuna/formal_tuning")
DATASET_SLUGS = {
    "HandWrite": "handwrite",
    "BDGP": "bdgp",
    "MSRC": "msrc",
    "COIL20": "coil20",
    "Caltech-5V": "caltech_5v",
    "MNIST-USPS": "mnist_usps",
    "ALOI": "aloi",
    "CCV": "ccv",
}
REQUIRED_CARRIER_KEYS = (
    "q_views", "p_gen", "p_ver", "U_cycle", "generator_actions",
    "verifier_actions", "sample_ids", "labeled_ids", "labeled_targets",
)


def validate_protocol_identity() -> None:
    if amendment_a1.APPLIES_TO_ALL_DATASETS is not True:
        raise RuntimeError("A1_NOT_DATASET_INVARIANT")
    if amendment_a1.AMENDED_DATASETS is not protocol.DATASETS:
        raise RuntimeError("A1_DATASET_IDENTITY_MISMATCH")
    if set(DATASET_SLUGS) != set(protocol.DATASETS):
        raise RuntimeError("SLUG_REGISTRY_MISMATCH")
    if ACTIVE_PROTOCOL != "P3-C1-B1-A1":
        raise RuntimeError("WRONG_ACTIVE_PROTOCOL")


def validate_request(dataset: str, dev_seed: int, device: str) -> None:
    validate_protocol_identity()
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    if isinstance(dev_seed, bool) or dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("UNREGISTERED_DEV_SEED")
    if device != "cuda:0":
        raise ValueError("FORMAL_CARRIER_REQUIRES_CUDA_0")


def expected_identities() -> tuple[tuple[str, int], ...]:
    identities = tuple((dataset, seed) for dataset in protocol.DATASETS for seed in protocol.DEV_SEEDS)
    if len(identities) != len(protocol.DATASETS) * len(protocol.DEV_SEEDS):
        raise RuntimeError("FORMAL_CARRIER_IDENTITY_COUNT_FAILURE")
    return identities


def adapter_route(dataset: str) -> str:
    if dataset not in protocol.DATASETS:
        raise ValueError("UNREGISTERED_DATASET")
    return "msrc" if dataset == "MSRC" else "generic"


def carrier_directory(output_root: str | Path, dataset: str, dev_seed: int) -> Path:
    if dataset not in DATASET_SLUGS or dev_seed not in protocol.DEV_SEEDS:
        raise ValueError("INVALID_FORMAL_CARRIER_IDENTITY")
    return Path(output_root) / DATASET_SLUGS[dataset] / f"dev_seed{dev_seed}" / "carrier"


def _sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _expected_dimensions(dataset: str) -> tuple[int, int]:
    spec = msrc_adapter.MSRC_SPEC if dataset == "MSRC" else c0_protocol.DATASET_SPECS[dataset]
    return int(spec.n), int(spec.v)


def _artifact_paths(dataset: str) -> tuple[Path, Path]:
    if dataset == "MSRC":
        _, _, split_path = b0_materializer.load_frozen_sparse_split(dataset)
        _, _, corruption_path = b0_materializer.load_frozen_corruption_mask(dataset)
        return split_path, corruption_path
    return c0_protocol.artifact_paths(dataset)


def _source_provenance(dataset: str) -> dict[str, str]:
    adapter_source = Path(msrc_adapter.__file__) if dataset == "MSRC" else Path(carrier_adapter.__file__)
    return {
        "original_b1_protocol": sha256_file(Path(protocol.__file__)),
        "a1_amendment": sha256_file(Path(amendment_a1.__file__)),
        "adapter_source": sha256_file(adapter_source),
        "corruption_protocol_source": sha256_file(c0_protocol.PARENT_FREEZE_PATHS["p3_a4_source"]),
        "sparse_label_protocol_source": sha256_file(c0_protocol.PARENT_FREEZE_PATHS["p3_a3_source"]),
    }


def _load_and_validate_pre_gt(path: Path, dataset: str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_CARRIER_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"MISSING_PRE_GT_KEYS:{sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_CARRIER_KEYS}
    q, p_gen, p_ver, utility = (
        arrays["q_views"], arrays["p_gen"], arrays["p_ver"], arrays["U_cycle"]
    )
    n_expected, v_expected = _expected_dimensions(dataset)
    if q.ndim != 3 or q.shape[:2] != (n_expected, v_expected):
        raise ValueError("INVALID_Q_SHAPE")
    n, v, k = q.shape
    if p_gen.ndim != 3 or p_ver.shape != p_gen.shape or p_gen.shape[0] != n or p_gen.shape[2] != k:
        raise ValueError("INVALID_POSTERIOR_SHAPE")
    actions = p_gen.shape[1]
    if utility.shape != (n, actions):
        raise ValueError("INVALID_U_SHAPE")
    if arrays["generator_actions"].shape[0] != actions or arrays["verifier_actions"].shape[0] != actions:
        raise ValueError("INVALID_ACTION_SHAPE")
    if actions != action_protocol.expected_action_count(v):
        raise ValueError("INVALID_ACTION_COUNT")
    if arrays["sample_ids"].shape != (n,) or len(set(arrays["sample_ids"].tolist())) != n:
        raise ValueError("INVALID_SAMPLE_IDS")
    labeled_ids, labeled_targets = arrays["labeled_ids"], arrays["labeled_targets"]
    if labeled_ids.ndim != 1 or labeled_targets.shape != labeled_ids.shape:
        raise ValueError("INVALID_SPARSE_LABEL_SHAPE")
    unique_targets, counts = np.unique(labeled_targets, return_counts=True)
    if len(unique_targets) != k or not np.array_equal(unique_targets, np.arange(k)) or np.any(counts != 2):
        raise ValueError("SPARSE_LABELS_MUST_BE_EXACTLY_TWO_PER_CLASS")
    if not all(np.isfinite(value).all() for value in (q, p_gen, p_ver, utility)):
        raise ValueError("NONFINITE_CARRIER_ARTIFACT")
    if np.any(utility < 0):
        raise ValueError("NEGATIVE_UTILITY")

    y_gen = np.argmax(p_gen, axis=2)
    y_ver = np.argmax(p_ver, axis=2)
    conf_gen = np.max(p_gen, axis=2)
    support_ver = np.take_along_axis(p_ver, y_gen[..., None], axis=2)[..., 0]
    recomputed = (y_gen == y_ver) * np.sqrt(conf_gen * support_ver)
    if not np.allclose(utility, recomputed, rtol=1e-6, atol=1e-7):
        raise ValueError("U_FORMULA_PARITY_FAILURE")
    masses = utility.sum(axis=1)
    labeled_positions = {int(raw_id): index for index, raw_id in enumerate(arrays["sample_ids"])}
    labeled_mass = np.asarray([masses[labeled_positions[int(raw_id)]] for raw_id in labeled_ids])
    details = {
        "view_count": int(v), "cluster_count": int(k), "action_count": int(actions),
        "sample_id_hash": _sha256_array(arrays["sample_ids"]),
        "labeled_id_hash": _sha256_array(labeled_ids),
        "u_shape": list(utility.shape), "u_finite": True,
        "u_positive_count": int(np.count_nonzero(utility > 0)),
        "u_zero_count": int(np.count_nonzero(utility == 0)),
        "zero_mass_sample_count": int(np.count_nonzero(masses == 0)),
        "zero_mass_labeled_count": int(np.count_nonzero(labeled_mass == 0)),
        "u_formula_parity": True,
    }
    return arrays, details


def _invoke_frozen_adapter(dataset: str, dev_seed: int, device: str, staging: Path):
    if adapter_route(dataset) == "msrc":
        return msrc_adapter.build_msrc_development_carrier(dev_seed, device, staging)
    return carrier_adapter.build_development_carrier(dataset, dev_seed, device, staging)


def verify_complete_carrier(path: str | Path, dataset: str, dev_seed: int) -> dict:
    path = Path(path)
    marker_path = path / "COMPLETE.json"
    if not marker_path.is_file():
        raise ValueError("MISSING_COMPLETE_MARKER")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    required = {
        "carrier_model_state.pt": marker.get("carrier_state_sha256"),
        "carrier_pre_gt.npz": marker.get("carrier_pre_gt_sha256"),
        "carrier_audit.json": marker.get("audit_sha256"),
    }
    if marker.get("status") != "COMPLETE" or marker.get("dataset") != dataset:
        raise ValueError("COMPLETE_IDENTITY_MISMATCH")
    if marker.get("dev_seed") != dev_seed or marker.get("carrier_epochs") != protocol.CARRIER_EPOCHS:
        raise ValueError("COMPLETE_PROTOCOL_MISMATCH")
    if marker.get("active_protocol") != ACTIVE_PROTOCOL or marker.get("full_gt_loaded") is not False:
        raise ValueError("COMPLETE_PROTOCOL_MISMATCH")
    for filename, expected in required.items():
        artifact = path / filename
        if not artifact.is_file() or not isinstance(expected, str) or sha256_file(artifact) != expected:
            raise ValueError("COMPLETE_HASH_MISMATCH")
    audit = json.loads((path / "carrier_audit.json").read_text(encoding="utf-8"))
    if (audit.get("dataset") != dataset or audit.get("dev_seed") != dev_seed
            or audit.get("carrier_epochs") != protocol.CARRIER_EPOCHS
            or audit.get("completed_epochs") != protocol.CARRIER_EPOCHS
            or audit.get("active_protocol") != ACTIVE_PROTOCOL
            or audit.get("full_gt_loaded") is not False
            or audit.get("full_gt_used") is not False
            or audit.get("formal_gt_metrics_computed") is not False
            or audit.get("optuna_study_created") is not False
            or audit.get("refinement_run") is not False):
        raise ValueError("CARRIER_AUDIT_MISMATCH")
    return {"status": "ALREADY_COMPLETE_VERIFIED", "path": str(path), "marker": marker, "audit": audit}


def _remove_own_incomplete(target: Path, staging_paths: list[Path]) -> None:
    if target.exists():
        if target.name != "carrier" or target.parent.name.startswith("dev_seed") is False:
            raise RuntimeError("REFUSE_UNSAFE_INCOMPLETE_REPLACEMENT")
        shutil.rmtree(target)
    for staging in staging_paths:
        if staging.parent != target.parent or not staging.name.startswith(".carrier_staging_"):
            raise RuntimeError("REFUSE_UNSAFE_STAGING_REPLACEMENT")
        shutil.rmtree(staging)


def materialize_one(
    dataset: str,
    dev_seed: int,
    device: str,
    output_root: str | Path,
    replace_incomplete: bool = False,
) -> dict:
    """Materialize exactly one carrier and atomically publish it after validation."""
    validate_request(dataset, dev_seed, device)
    target = carrier_directory(output_root, dataset, dev_seed)
    parent = target.parent
    stale = sorted(parent.glob(".carrier_staging_*")) if parent.exists() else []
    if target.exists() and (target / "COMPLETE.json").is_file():
        return verify_complete_carrier(target, dataset, dev_seed)
    if target.exists() or stale:
        if not replace_incomplete:
            raise RuntimeError("INCOMPLETE_EXISTING_OUTPUT")
        _remove_own_incomplete(target, stale)

    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".carrier_staging_{uuid.uuid4().hex}"
    zero_mass_adapter_stop = False
    try:
        try:
            _invoke_frozen_adapter(dataset, dev_seed, device, staging)
        except RuntimeError as error:
            if str(error) != "ZERO_UTILITY_MASS":
                raise
            if not (staging / "carrier_model_state.pt").is_file() or not (staging / "carrier_pre_gt.npz").is_file():
                raise RuntimeError("ZERO_UTILITY_STOP_BEFORE_ARTIFACT_COMPLETION") from error
            zero_mass_adapter_stop = True

        state_path = staging / "carrier_model_state.pt"
        pre_gt_path = staging / "carrier_pre_gt.npz"
        if not state_path.is_file() or not pre_gt_path.is_file():
            raise RuntimeError("MISSING_CARRIER_ARTIFACT")
        _, details = _load_and_validate_pre_gt(pre_gt_path, dataset)
        split_path, corruption_path = _artifact_paths(dataset)
        spec = msrc_adapter.MSRC_SPEC if dataset == "MSRC" else c0_protocol.DATASET_SPECS[dataset]
        native_lr = p2_carrier.LR if dataset == "MSRC" else c0_protocol.NATIVE_LR
        state_hash, pre_gt_hash = sha256_file(state_path), sha256_file(pre_gt_path)
        audit = {
            "dataset": dataset, "dataset_slug": DATASET_SLUGS[dataset], "dev_seed": dev_seed,
            "carrier_epochs": protocol.CARRIER_EPOCHS, "completed_epochs": protocol.CARRIER_EPOCHS,
            "adapter_route": adapter_route(dataset), "active_protocol": ACTIVE_PROTOCOL,
            "native_lr": native_lr, "batch_size": int(spec.batch_size),
            **details,
            "carrier_state_sha256": state_hash, "carrier_pre_gt_sha256": pre_gt_hash,
            "corruption_provenance": str(corruption_path),
            "corruption_sha256": sha256_file(corruption_path),
            "sparse_label_provenance": str(split_path),
            "sparse_label_sha256": sha256_file(split_path),
            "frozen_source_sha256": _source_provenance(dataset),
            "legacy_zero_utility_gate_observed": zero_mass_adapter_stop,
            "full_gt_loaded": False, "full_gt_used": False,
            "formal_gt_metrics_computed": False,
            "optuna_study_created": False, "refinement_run": False,
        }
        audit_path = staging / "carrier_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audit_hash = sha256_file(audit_path)
        marker = {
            "status": "COMPLETE", "dataset": dataset, "dev_seed": dev_seed,
            "carrier_epochs": protocol.CARRIER_EPOCHS, "active_protocol": ACTIVE_PROTOCOL,
            "carrier_state_sha256": state_hash, "carrier_pre_gt_sha256": pre_gt_hash,
            "audit_sha256": audit_hash, "full_gt_loaded": False,
        }
        (staging / "COMPLETE.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, target)
        verified = verify_complete_carrier(target, dataset, dev_seed)
        return {"status": "COMPLETE", "path": str(target), "marker": verified["marker"], "audit": audit}
    except Exception:
        # A failed staging directory is intentionally left without a published
        # target; explicit --replace-incomplete owns later recovery.
        raise
