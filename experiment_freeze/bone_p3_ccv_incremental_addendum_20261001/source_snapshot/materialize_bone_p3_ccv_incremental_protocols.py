"""CCV-only addendum using frozen P3-A1/A3/A4 protocol implementations.

This is intentionally a thin integration layer: P3-A3 selects sparse labels
and P3-A4 constructs the mask through their existing implementations.  It does
not alter those protocols, materialize another dataset, train BONE, create
features/noise tensors, or construct a carrier.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from experiments.utility_relation import bone_p3_a1_action_space_protocol as a1
from experiments.utility_relation import bone_p3_a3_sparse_label_protocol as a3
from experiments.utility_relation import bone_p3_a4_corruption_protocol as a4
from experiments.utility_relation import materialize_bone_p3_a3_sparse_splits as a3_writer
from experiments.utility_relation import materialize_bone_p3_a4_corruptions as a4_writer


REPO_ROOT = Path(__file__).resolve().parents[2]
CCV_DATA = REPO_ROOT / "data" / "CCV.mat"
P3_A2_AUDIT = REPO_ROOT / "outputs" / "bone_p3" / "p3_a2_ccv_incremental_portability_audit.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "bone_p3" / "ccv_incremental"
CCV_SHA256 = "4fc88d5c925f9a6c00bbcafc30349df15a32dac44464eb40ef767ca3569f244a"
N, V, K, L = 6773, 3, 20, 40


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parent_hashes() -> dict[str, str]:
    paths = {
        "bone_p3_a1_action_space_protocol.py": REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_a1_action_space_protocol.py",
        "bone_p3_a3_sparse_label_protocol.py": REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_a3_sparse_label_protocol.py",
        "bone_p3_a4_corruption_protocol.py": REPO_ROOT / "experiments" / "utility_relation" / "bone_p3_a4_corruption_protocol.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_ccv_labels(data_path: Path = CCV_DATA) -> np.ndarray:
    """The sole full-GT read, permitted only for offline A3 split freezing."""
    return _load_ccv_labels_cached(Path(data_path))


@lru_cache(maxsize=1)
def _load_ccv_labels_cached(data_path: Path) -> np.ndarray:
    if sha256_file(data_path) != CCV_SHA256:
        raise ValueError("CCV.mat SHA256 does not match the verified source")
    mat = scipy.io.loadmat(data_path, variable_names=["Y"])
    if set(mat) & {"X1", "X2", "X3"}:
        raise RuntimeError("CCV incremental A3 materialization must not load features")
    labels = np.asarray(mat.get("Y")).reshape(-1)
    if labels.shape != (N,) or not np.array_equal(np.unique(labels), np.arange(K)):
        raise ValueError("CCV label schema is not the verified 0..19 raw labeling")
    return labels


def _row_histogram(mask: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(mask.sum(axis=1), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _action_audit() -> dict[str, Any]:
    action_list = a1.build_balanced_directed_actions(V)
    a1.validate_balanced_directed_actions(V, action_list)
    if len(action_list) != 6 or a1.generator_sizes(V) != (1, 2):
        raise ValueError("CCV V=3 action protocol does not meet frozen expectations")
    return {"status": "PASS", "V": V, "S": len(action_list), "generator_sizes": list(a1.generator_sizes(V)),
            "action_protocol_sha256": a1.canonical_action_sha256(V)}


def _build_sparse_record(labels: np.ndarray, output_dir: Path, source_sha: str, parent_hashes: dict[str, str]) -> dict[str, Any]:
    split = a3.build_sparse_label_split("CCV", labels)
    if split.split_mode != "P3_A3_HASH_STRATIFIED_V1" or split.labeled_count != L or split.k != K:
        raise ValueError("CCV sparse split does not satisfy P3-A3 frozen protocol")
    if not np.array_equal(split.raw_classes, np.arange(K)) or not np.array_equal(split.labeled_targets, labels[split.labeled_ids]):
        raise ValueError("CCV sparse split violates canonical raw label parity")
    npz_path = output_dir / "CCV_2perclass.npz"
    a3_writer.write_deterministic_npz(npz_path, split.labeled_ids, split.labeled_targets)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"labeled_ids", "labeled_targets"}:
            raise ValueError("CCV sparse artifact is not minimal")
        if not np.array_equal(archive["labeled_ids"], split.labeled_ids) or not np.array_equal(archive["labeled_targets"], split.labeled_targets):
            raise ValueError("CCV sparse artifact content mismatch")
    return {
        "dataset": "CCV", "N": N, "V": V, "K": K, "L": L,
        "labels_per_class": a3.LABELS_PER_CLASS, "label_fraction": L / N,
        "row_id_semantics": "RAW_FILE_ROW_ID", "split_mode": split.split_mode,
        "namespace": a3.NAMESPACE, "raw_to_canonical_label_mapping": a3.raw_to_canonical_label_mapping(split.raw_classes),
        "labeled_ids": split.labeled_ids.tolist(), "labeled_targets": split.labeled_targets.tolist(),
        "ids_targets_content_sha256": a3.split_content_sha256(split), "npz_sha256": sha256_file(npz_path),
        "source_ccv_mat_sha256": source_sha, "parent_protocol_sha256": parent_hashes["bone_p3_a3_sparse_label_protocol.py"],
        "GT_loaded_only_for_offline_A3": True,
    }


def _build_mask_record(output_dir: Path, source_sha: str, parent_hashes: dict[str, str]) -> dict[str, Any]:
    """Call frozen A4 with its allowed inputs only: name, N, V, seed=20."""
    mask = a4.build_balanced_half_view_mask("CCV", N, V)
    raw_row_ids = np.arange(N, dtype=np.int64)
    a4.validate_balanced_half_view_mask(mask, "CCV")
    histogram = _row_histogram(mask)
    patterns = a4.pattern_counts(mask)
    per_view = [int(value) for value in mask.sum(axis=0)]
    if histogram != {"1": 3387, "2": 3386} or int(mask.sum()) != 10159:
        raise ValueError("P3_A4_CCV_PORTABILITY_FAIL_CLOSED: row cardinality/global count")
    singleton_counts = [patterns.get(str(view), 0) for view in range(V)]
    pair_counts = [patterns.get("%d,%d" % pair, 0) for pair in ((0, 1), (0, 2), (1, 2))]
    if singleton_counts != [1129, 1129, 1129] or sum(pair_counts) != 3386 or max(pair_counts) - min(pair_counts) > 1:
        raise ValueError("P3_A4_CCV_PORTABILITY_FAIL_CLOSED: pattern balance")
    if sum(per_view) != 10159 or max(per_view) - min(per_view) > 1:
        raise ValueError("P3_A4_CCV_PORTABILITY_FAIL_CLOSED: per-view balance")
    npz_path = output_dir / "CCV_corruption_mask.npz"
    a4_writer.write_deterministic_mask_npz(npz_path, mask, raw_row_ids)
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"corruption_mask", "raw_row_ids"}:
            raise ValueError("CCV mask artifact is not minimal")
        if not np.array_equal(archive["corruption_mask"], mask) or not np.array_equal(archive["raw_row_ids"], raw_row_ids):
            raise ValueError("CCV mask artifact content mismatch")
    return {
        "dataset": "CCV", "N": N, "V": V, "snr_db": a4.TARGET_SNR_DB,
        "corruption_seed": a4.CORRUPTION_SEED, "row_id_semantics": "RAW_FILE_ROW_ID",
        "row_sum_histogram": histogram, "total_corrupted_cells": int(mask.sum()),
        "corruption_fraction": float(mask.mean()), "pattern_counts": patterns, "per_view_counts": per_view,
        "mask_content_sha256": a4.mask_sha256(mask), "npz_sha256": sha256_file(npz_path),
        "source_ccv_mat_sha256": source_sha, "parent_protocol_sha256": parent_hashes["bone_p3_a4_corruption_protocol.py"],
        "mask_builder_inputs": {"dataset_name": "CCV", "N": N, "V": V, "corruption_seed": a4.CORRUPTION_SEED},
        "no_GT_or_features_used_for_A4": True,
    }


def materialize(output_dir: Path = DEFAULT_OUTPUT_DIR, data_path: Path = CCV_DATA) -> dict[str, Any]:
    """Produce only CCV A3/A4 addendum files after all frozen gates validate."""
    output_dir = Path(output_dir)
    source_sha = sha256_file(data_path)
    if source_sha != CCV_SHA256:
        raise ValueError("CCV.mat SHA256 does not match the verified source")
    parent_hashes = _parent_hashes()
    labels = load_ccv_labels(data_path)  # Allowed exclusively for frozen A3 supervision.
    action = _action_audit()
    sparse = _build_sparse_record(labels, output_dir, source_sha, parent_hashes)
    mask = _build_mask_record(output_dir, source_sha, parent_hashes)
    p3_a2 = json.loads(P3_A2_AUDIT.read_text(encoding="utf-8"))
    if p3_a2.get("status") != "READY_FOR_P3_B0":
        raise ValueError("CCV P3-A2 incremental readiness is not READY_FOR_P3_B0")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p3_a3_ccv_sparse_addendum.json").write_text(json.dumps(sparse, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "p3_a4_ccv_corruption_addendum.json").write_text(json.dumps(mask, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "P3_A2_STATUS": "READY_FOR_P3_B0", "P3_A3_CCV_STATUS": "PASS", "P3_A4_CCV_STATUS": "PASS",
        "P3_A1_ACTION_STATUS": action["status"], "FINAL_CCV_PORTABILITY_STATUS": "READY_FOR_P3_B0",
        "p3_a2_incremental_audit_path": str(P3_A2_AUDIT.relative_to(REPO_ROOT)),
        "source_ccv_mat_sha256": source_sha, "parent_source_sha256": parent_hashes,
        "action": action, "sparse_ids_targets_content_sha256": sparse["ids_targets_content_sha256"],
        "corruption_mask_content_sha256": mask["mask_content_sha256"],
        "no_training": True, "no_gpu": True, "no_noisy_features": True, "no_carrier": True,
    }
    (output_dir / "p3_ccv_incremental_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"sparse": sparse, "mask": mask, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    result = materialize(build_parser().parse_args().output_dir)
    print(json.dumps({"status": result["summary"]["FINAL_CCV_PORTABILITY_STATUS"], "output": "CCV-only addendum"}, indent=2))


if __name__ == "__main__":
    main()
