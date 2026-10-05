"""Tests for P3-A3 deterministic, raw-row sparse-label protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import bone_p3_a3_sparse_label_protocol as protocol
from experiments.utility_relation import materialize_bone_p3_a3_sparse_splits as materializer


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BONE_SHA256 = "d13ba6ba91362210f9a614f7353e9de688ff1815f075916b457df6c767a14092"
ORIGINAL_AMV_SHA256 = "68f3edcd55e32456a5bcdbcbbc48329f19c5652dee1fc27420e2865f7068e5dc"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selector_sha_is_exact_and_candidate_order_independent() -> None:
    assert protocol.selector_payload("Toy", 7) == b"BONE_P3_A3_V1|Toy|7"
    assert protocol.selector_digest("Toy", 7) == hashlib.sha256(b"BONE_P3_A3_V1|Toy|7").hexdigest()
    rows = (3, 11, 2, 9, 7)
    assert protocol.rank_candidate_raw_rows("Toy", rows) == protocol.rank_candidate_raw_rows("Toy", reversed(rows))


def test_hash_stratified_split_is_row_order_independent_and_exactly_two_per_class() -> None:
    raw_labels = np.asarray([20, 10, 30, 20, 10, 30, 20, 10, 30, 20, 10, 30])
    first = protocol.build_sparse_label_split("Toy", raw_labels)
    second = protocol.build_sparse_label_split("Toy", raw_labels.copy())
    raw_classes, canonical = protocol.canonicalize_raw_labels(raw_labels)
    assert np.array_equal(first.labeled_ids, second.labeled_ids)
    assert np.array_equal(first.labeled_targets, second.labeled_targets)
    assert first.split_mode == "P3_A3_HASH_STRATIFIED_V1"
    assert np.array_equal(raw_classes, [10, 20, 30])
    assert np.array_equal(canonical[first.labeled_ids], first.labeled_targets)
    assert np.array_equal(np.bincount(first.labeled_targets), [2, 2, 2])
    assert np.unique(first.labeled_ids).size == first.labeled_ids.size


def test_canonical_mapping_and_insufficient_class_fail_closed() -> None:
    raw_classes, canonical = protocol.canonicalize_raw_labels(np.asarray([9, 4, 9, 4]))
    assert np.array_equal(raw_classes, [4, 9])
    assert np.array_equal(canonical, [1, 0, 1, 0])
    assert protocol.raw_to_canonical_label_mapping(raw_classes) == {"4": 0, "9": 1}
    with pytest.raises(ValueError, match="fewer than 2"):
        protocol.build_sparse_label_split("Toy", np.asarray([0, 0, 1]))


def test_msrc_is_exact_legacy_exception_not_hash_reconstruction() -> None:
    labels = materializer.load_raw_labels("MSRC")
    split = protocol.build_sparse_label_split("MSRC", labels)
    assert split.split_mode == "LEGACY_FROZEN"
    assert split.provenance_status == "PARTIAL_PROVENANCE"
    assert np.array_equal(split.labeled_ids, protocol.LEGACY_MSRC_IDS)
    assert np.array_equal(split.labeled_targets, protocol.LEGACY_MSRC_TARGETS)


def test_aloi_actual_k_is_100_and_fraction_is_exact() -> None:
    labels = materializer.load_raw_labels("ALOI")
    split = protocol.build_sparse_label_split("ALOI", labels)
    assert split.k == 100
    assert split.labeled_count == 200
    assert split.labeled_count / labels.size == pytest.approx(200 / 10800)


@pytest.mark.parametrize("dataset_name", materializer.READY_DATASETS)
def test_ready_dataset_fraction_and_repeatable_content(dataset_name: str) -> None:
    labels = materializer.load_raw_labels(dataset_name)
    first = protocol.build_sparse_label_split(dataset_name, labels)
    second = protocol.build_sparse_label_split(dataset_name, labels)
    assert first.labeled_count / labels.size == pytest.approx(materializer.EXPECTED_FRACTIONS[dataset_name])
    assert protocol.split_content_sha256(first) == protocol.split_content_sha256(second)
    assert np.array_equal(first.labeled_ids, second.labeled_ids)
    assert np.array_equal(first.labeled_targets, second.labeled_targets)


def test_minimal_npz_is_binary_repeatable_and_missing_data_is_not_materialized(tmp_path: Path) -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    split = protocol.build_sparse_label_split("Toy", labels)
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    materializer.write_deterministic_npz(first, split.labeled_ids, split.labeled_targets)
    materializer.write_deterministic_npz(second, split.labeled_ids, split.labeled_targets)
    assert _sha256(first) == _sha256(second)
    with np.load(first, allow_pickle=False) as archive:
        assert set(archive.files) == {"labeled_ids", "labeled_targets"}
    with pytest.raises(ValueError, match="not a READY"):
        materializer.materialize_dataset("CCV", tmp_path)


def test_no_rng_or_model_training_dependency_and_native_hashes() -> None:
    source = Path(protocol.__file__).read_text(encoding="utf-8") + Path(materializer.__file__).read_text(encoding="utf-8")
    # Documentation may state the boundary, so guard executable imports/calls
    # rather than rejecting the explanatory words themselves.
    for forbidden in ("np.random", "random.", "import torch", "from torch", "DataLoader(", "optim.", "amv(", "BONE(", "KMeans(", "get_cluster_metrics("):
        assert forbidden not in source
    assert _sha256(REPO_ROOT / "method/bone.py") == ORIGINAL_BONE_SHA256
    assert _sha256(REPO_ROOT / "method/amv.py") == ORIGINAL_AMV_SHA256
