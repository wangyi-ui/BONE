"""Offline invariant tests for the opt-in P3-B1 Caltech engineering smoke."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import bone_p3_b0_generic_carrier as carrier
from experiments.utility_relation import run_bone_p3_b1_caltech_engineering_smoke as smoke


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_caltech_metadata_and_raw_feature_loading_without_label_request() -> None:
    assert (smoke.N, smoke.V, smoke.K, smoke.FEATURE_DIMS) == (1400, 5, 7, (40, 254, 1984, 512, 928))
    views = smoke.load_caltech_raw_views()
    assert [view.shape for view in views] == [(1400, dim) for dim in smoke.FEATURE_DIMS]
    assert all(view.dtype == np.float32 and np.isfinite(view).all() for view in views)
    source = Path(smoke.__file__).read_text(encoding="utf-8")
    assert 'variable_names=keys' in source
    assert 'mat["Y"]' not in source and "mat['Y']" not in source


def test_frozen_sparse_split_and_corruption_loading() -> None:
    labels, targets, mask, audit = smoke.load_and_audit_frozen_inputs()
    assert labels.shape == targets.shape == (14,)
    assert np.array_equal(np.bincount(targets, minlength=7), np.full(7, 2))
    assert mask.dtype == np.bool_ and mask.shape == (1400, 5)
    assert audit["row_histogram"] == {"2": 700, "3": 700}
    assert audit["per_view_counts"] == [700] * 5


def test_raw_id_mapping_and_sparse_anchor_mapping_are_exact() -> None:
    labels, _targets, mask, _audit = smoke.load_and_audit_frozen_inputs()
    adapter = carrier.SampleIdAdapter.from_native_shuffle(smoke.N)
    record = smoke.verify_sample_id_adapter(adapter, labels)
    assert record["unique_raw_ids"] == 1400
    assert record["exact_raw_id_set_0_to_1399"]
    assert np.array_equal(adapter.sample_ids[adapter.row_positions(labels)], labels)
    # The mask remains raw-row indexed; this explicit gather is the required bridge.
    assert np.array_equal(mask[adapter.sample_ids][adapter.row_positions(labels)], mask[labels])


def test_v5_action_space_and_exact_expected_u_t_shapes() -> None:
    actions, generator, verifier = carrier.action_arrays(5)
    assert len(actions) == generator.shape[0] == verifier.shape[0] == 20
    assert sorted({action.generator_size for action in actions}) == [2, 3]
    ids = carrier.native_shared_permutation(smoke.N)
    targets = np.repeat(np.arange(smoke.K, dtype=np.int64), 2)
    labeled = ids[np.asarray((0, 13, 101, 216, 399, 502, 603, 777, 810, 919, 1001, 1177, 1260, 1399))]
    q = np.full((smoke.N, smoke.V, smoke.K), 1.0 / smoke.K, dtype=np.float32)
    payload = carrier.build_generic_carrier(q, ids, labeled, targets, query_chunk_size=256)
    assert payload["p_gen"].shape == (1400, 20, 7)
    assert payload["p_ver"].shape == (1400, 20, 7)
    assert payload["U_cycle"].shape == (1400, 20)
    assert payload["Pi"].shape == (1386, 20, 7)
    assert payload["T_rel"].shape == (1386, 14, 20)
    assert np.isfinite(payload["U_cycle"]).all() and np.isfinite(payload["T_rel"]).all()


def test_smoke_is_opt_in_pre_gt_and_engineering_only() -> None:
    parser = smoke.build_parser()
    assert parser.parse_args([]).execute is False
    source = Path(smoke.__file__).read_text(encoding="utf-8")
    for forbidden in ("get_cluster_metrics", "accuracy_score", "normalized_mutual", "purity", "get_dataset("):
        assert forbidden not in source
    assert '"ENGINEERING_ONLY": True' in source
    assert '"FORMAL_RESULT": False' in source
    assert '"GT_loaded_before_seal": False' in source


def test_native_hash_protection_and_frozen_hashes_are_pinned() -> None:
    assert smoke.verify_native_hash_protection() == smoke.EXPECTED_NATIVE_HASHES
    hashes = smoke.frozen_input_hashes()
    assert hashes["caltech_data_source"] == _sha(smoke.DATA_PATH)
    assert hashes["p3_a3_split"] == _sha(smoke.SPARSE_SPLIT)
    assert hashes["p3_a4_mask"] == _sha(smoke.CORRUPTION_MASK)
    assert _sha(REPO_ROOT / "method/bone.py") == smoke.EXPECTED_NATIVE_HASHES["method/bone.py"]
    assert _sha(REPO_ROOT / "method/amv.py") == smoke.EXPECTED_NATIVE_HASHES["method/amv.py"]


def test_cuda_execution_fails_closed_when_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(smoke.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
        smoke.run_engineering_smoke(tmp_path)


def test_cuda_memory_stat_api_uses_current_device_without_device_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def record(name: str, result: int | None = None):
        def call(*args: object) -> int | None:
            calls.append((name, args))
            return result
        return call

    monkeypatch.setattr(smoke.torch.cuda, "set_device", record("set_device"))
    monkeypatch.setattr(smoke.torch.cuda, "empty_cache", record("empty_cache"))
    monkeypatch.setattr(smoke.torch.cuda, "reset_peak_memory_stats", record("reset_peak_memory_stats"))
    monkeypatch.setattr(smoke.torch.cuda, "max_memory_allocated", record("max_memory_allocated", 123))
    monkeypatch.setattr(smoke.torch.cuda, "max_memory_reserved", record("max_memory_reserved", 456))
    smoke.prepare_cuda_memory_stats()
    assert smoke.peak_cuda_memory_stats() == {"peak_allocated_bytes": 123, "peak_reserved_bytes": 456}
    assert calls == [
        ("set_device", (0,)),
        ("empty_cache", ()),
        ("reset_peak_memory_stats", ()),
        ("max_memory_allocated", ()),
        ("max_memory_reserved", ()),
    ]
