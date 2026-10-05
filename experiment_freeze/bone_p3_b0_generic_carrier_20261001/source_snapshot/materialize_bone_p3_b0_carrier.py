"""P3-B0 generic carrier materialization helpers and MSRC backward-parity audit.

No command in this module trains a model.  Future training code supplies q
posteriors extracted from BONE and uses ``materialize_from_q_views``; this file
only joins those posteriors with frozen P3-A3/P3-A4 inputs.  Its default CLI
performs a read-only MSRC replay audit against the historical P2 carrier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.utility_relation import bone_p3_a1_action_space_protocol as action_protocol
from experiments.utility_relation import bone_p3_b0_generic_carrier as carrier


REPO_ROOT = Path(__file__).resolve().parents[2]
P3_SPLIT_DIR = REPO_ROOT / "outputs" / "bone_p3" / "sparse_splits"
P3_CORRUPTION_DIR = REPO_ROOT / "outputs" / "bone_p3" / "corruptions"
P2_MSRC_CARRIER = REPO_ROOT / "outputs" / "bone_p2" / "p2_a0_msrc_seed20" / "p2_a0_carrier_pre_gt.npz"
P2_MSRC_MODEL_STATE = REPO_ROOT / "outputs" / "bone_p2" / "p2_a0_msrc_seed20" / "carrier_model_state.pt"
DEFAULT_AUDIT_PATH = REPO_ROOT / "outputs" / "bone_p3" / "p3_b0_msrc_backward_parity_audit.json"
VALID_TRAINING_SEEDS = (20, 30, 50)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_sparse_split(dataset_name: str, split_dir: Path = P3_SPLIT_DIR) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load only frozen sparse IDs/targets; full dataset labels are never read."""
    path = Path(split_dir) / f"{dataset_name}_2perclass.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"labeled_ids", "labeled_targets"}:
            raise ValueError("frozen sparse split has an invalid minimal schema")
        labeled_ids = np.asarray(archive["labeled_ids"], dtype=np.int64)
        labeled_targets = np.asarray(archive["labeled_targets"], dtype=np.int64)
    return labeled_ids, labeled_targets, path


def load_frozen_corruption_mask(dataset_name: str, corruption_dir: Path = P3_CORRUPTION_DIR) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load P3-A4's raw-row mask without regenerating or altering it."""
    path = Path(corruption_dir) / f"{dataset_name}_mask.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"corruption_mask", "raw_row_ids"}:
            raise ValueError("frozen corruption artifact has an invalid minimal schema")
        mask = np.asarray(archive["corruption_mask"])
        raw_row_ids = np.asarray(archive["raw_row_ids"], dtype=np.int64)
    if mask.dtype != np.bool_ or mask.shape[0] != raw_row_ids.size or not np.array_equal(raw_row_ids, np.arange(raw_row_ids.size, dtype=np.int64)):
        raise ValueError("frozen corruption mask must be indexed by complete raw row IDs")
    return mask, raw_row_ids, path


def _float_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    if actual.shape != expected.shape:
        return {"shape_equal": False, "exact": False, "allclose_1e-7": False, "max_abs_diff": None}
    max_abs = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    return {
        "shape_equal": True,
        "exact": bool(np.array_equal(actual, expected)),
        "allclose_1e-7": bool(np.allclose(actual, expected, rtol=0.0, atol=1e-7)),
        "max_abs_diff": max_abs,
    }


def _int_parity(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    return {"shape_equal": actual.shape == expected.shape, "exact": bool(np.array_equal(actual, expected))}


def msrc_backward_parity(query_chunk_size: int | None = None) -> dict[str, object]:
    """Reconstruct historical MSRC U/T from its q tensor; fail closed on semantics."""
    labeled_ids, labeled_targets, split_path = load_frozen_sparse_split("MSRC")
    mask, raw_ids, corruption_path = load_frozen_corruption_mask("MSRC")
    action_protocol.verify_msrc_action_order(P2_MSRC_CARRIER)
    with np.load(P2_MSRC_CARRIER, allow_pickle=False) as archive:
        historical = {name: np.asarray(archive[name]) for name in archive.files}
    rebuilt = carrier.build_generic_carrier(
        historical["q_views"], historical["sample_ids"], labeled_ids, labeled_targets, query_chunk_size=query_chunk_size
    )
    integer_pairs = {
        "sample_ids": (rebuilt["sample_ids"], historical["sample_ids"]),
        "generator_actions": (rebuilt["generator_actions"], historical["generator_subsets"]),
        "verifier_actions": (rebuilt["verifier_actions"], historical["verifier_subsets"]),
        "labeled_ids": (rebuilt["labeled_ids"], historical["labeled_ids"]),
        "labeled_targets": (rebuilt["labeled_targets"], historical["labeled_targets"]),
        "query_sample_ids": (rebuilt["query_sample_ids"], historical["unlabeled_ids"]),
        "anchor_sample_ids": (rebuilt["anchor_sample_ids"], historical["labeled_ids"]),
    }
    float_pairs = {
        "p_gen": (rebuilt["p_gen"], historical["p_gen"]),
        "p_ver": (rebuilt["p_ver"], historical["p_ver"]),
        "U_cycle": (rebuilt["U_cycle"], historical["u_cycle_all"]),
        "query_U_cycle": (rebuilt["query_U_cycle"], historical["u_cycle"]),
        "Pi": (rebuilt["Pi"], historical["pi"]),
        "T_rel": (rebuilt["T_rel"], historical["t_rel"]),
    }
    comparisons: dict[str, dict[str, object]] = {}
    comparisons.update({name: _int_parity(actual, expected) for name, (actual, expected) in integer_pairs.items()})
    comparisons.update({name: _float_parity(actual, expected) for name, (actual, expected) in float_pairs.items()})
    required_integer = tuple(integer_pairs)
    required_float = tuple(float_pairs)
    if not all(bool(comparisons[name]["exact"]) for name in required_integer):
        raise ValueError("MSRC integer/index backward parity failed")
    if not all(bool(comparisons[name]["allclose_1e-7"]) for name in required_float):
        raise ValueError("MSRC U/T semantic backward parity failed")
    return {
        "dataset": "MSRC", "status": "PASS", "query_chunk_size": query_chunk_size,
        "comparison": comparisons,
        "sample_id_semantics": carrier.RAW_ID_SEMANTICS,
        "sparse_split_sha256": sha256_file(split_path),
        "corruption_artifact_sha256": sha256_file(corruption_path),
        "historical_p2_carrier_sha256": sha256_file(P2_MSRC_CARRIER),
        "historical_mask_shape": list(mask.shape), "historical_raw_ids_exact": bool(np.array_equal(raw_ids, np.arange(210))),
        "GT_loaded_before_seal": False,
    }


def carrier_audit(
    dataset_name: str, payload: dict[str, np.ndarray], training_seed: int,
    corruption_path: Path, split_path: Path, carrier_model_state: Path,
    query_chunk_size: int | None,
) -> dict[str, Any]:
    """Build the pre-GT audit required for a future sealable carrier artifact."""
    if training_seed not in VALID_TRAINING_SEEDS:
        raise ValueError(f"training_seed must be one of {VALID_TRAINING_SEEDS}")
    q = payload["q_views"]
    n, v, k = q.shape
    tensors = {name: values for name, values in payload.items() if isinstance(values, np.ndarray)}
    return {
        "dataset": dataset_name, "N": int(n), "V": int(v), "K": int(k),
        "S": int(payload["generator_actions"].shape[0]), "L": int(payload["labeled_ids"].size),
        "training_seed": training_seed,
        "corruption_artifact_sha256": sha256_file(corruption_path),
        "sparse_split_sha256": sha256_file(split_path),
        "action_protocol_sha256": action_protocol.canonical_action_sha256(v),
        "carrier_model_state_sha256": sha256_file(carrier_model_state),
        "sample_id_semantics": carrier.RAW_ID_SEMANTICS,
        "query_chunk_size": query_chunk_size,
        "all_finite": bool(all(np.isfinite(values).all() for values in tensors.values() if np.issubdtype(values.dtype, np.floating))),
        "shape_audit": {name: list(values.shape) for name, values in tensors.items()},
        "GT_loaded_before_seal": False,
    }


def materialize_from_q_views(
    dataset_name: str, q_views: np.ndarray, sample_ids: np.ndarray, training_seed: int,
    carrier_model_state: Path, output_dir: Path, query_chunk_size: int | None = None,
) -> dict[str, Path]:
    """Write a generic pre-GT carrier from externally produced BONE q views.

    This function performs no training, raw-label loading, clustering metrics,
    or model inference.  It requires existing frozen sparse/corruption files.
    """
    labeled_ids, labeled_targets, split_path = load_frozen_sparse_split(dataset_name)
    _mask, _raw_ids, corruption_path = load_frozen_corruption_mask(dataset_name)
    payload = carrier.build_generic_carrier(q_views, sample_ids, labeled_ids, labeled_targets, query_chunk_size)
    audit = carrier_audit(dataset_name, payload, training_seed, corruption_path, split_path, Path(carrier_model_state), query_chunk_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    carrier_path = output_dir / "p3_b0_carrier_pre_gt.npz"
    audit_path = output_dir / "p3_b0_carrier_audit.json"
    np.savez_compressed(carrier_path, **payload)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"carrier": carrier_path, "audit": audit_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--query-chunk-size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = msrc_backward_parity(args.query_chunk_size)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(args.audit_path)}, indent=2))


if __name__ == "__main__":
    main()
