from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import amendment_a2, protocol, sparse_cv, trial_runner_a1
from .formal_tuning_context import CarrierContext
from .formal_tuning_context_a2 import (
    FormalTuningContextA2,
    build_formal_tuning_context_a2,
    build_formal_tuning_context_a2_with_audit,
)


def _carrier(tmp_path, dataset, seed, K, supported_ids=(0,)):
    labeled_ids = np.arange(2 * K, dtype=np.int64)
    targets = np.repeat(np.arange(K, dtype=np.int64), 2)
    sample_ids = np.arange(2 * K + 1, dtype=np.int64)
    utility = np.zeros((2 * K + 1, 1), dtype=float)
    utility[list(supported_ids), 0] = 1.0
    arrays = {
        "sample_ids": sample_ids,
        "labeled_ids": labeled_ids,
        "labeled_targets": targets,
        "U_cycle": utility,
        "p_gen": np.full((2 * K + 1, 1, K), 1.0 / K),
        "p_ver": np.full((2 * K + 1, 1, K), 1.0 / K),
    }
    folds = sparse_cv.build_sparse_label_folds(
        labeled_ids, targets, expected_class_ids=range(K)
    )
    return CarrierContext(
        dataset=dataset, dev_seed=seed, directory=tmp_path,
        state_path=tmp_path / f"state{seed}", pre_gt_path=tmp_path / f"pre{seed}",
        audit_path=tmp_path / f"audit{seed}", arrays=arrays,
        views=(np.zeros((2 * K + 1, 1)),),
        adapter=SimpleNamespace(), spec=SimpleNamespace(batch_size=1), k=K,
        folds=folds, state_file_sha256=f"s{seed}", pre_gt_sha256=f"p{seed}",
    )


def test_a2_context_bypasses_a1_feasibility_and_preserves_r3_numeric_order(tmp_path):
    carriers = {
        seed: _carrier(tmp_path, "COIL20", seed, 20, supported_ids=(39,))
        for seed in protocol.DEV_SEEDS
    }
    calls = []

    def loader(dataset, seed, root):
        calls.append((dataset, seed, root))
        return carriers[seed]

    context = build_formal_tuning_context_a2(
        "COIL20", tmp_path, carrier_loader=loader
    )
    assert isinstance(context, FormalTuningContextA2)
    assert context.active_protocol == amendment_a2.AMENDMENT_ID
    assert [call[1] for call in calls] == [11, 37]
    for seed in protocol.DEV_SEEDS:
        assert context.feasibility[seed]["A2_FEASIBILITY"] == "PASS"
        assert context.feasibility[seed]["abstained_class_count"] == 19
        assert context.carrier(seed).folds[0].train_targets == tuple(range(20))


def test_a2_context_all_zero_seed_still_fails_closed(tmp_path):
    carriers = {
        11: _carrier(tmp_path, "BDGP", 11, 5, supported_ids=()),
        37: _carrier(tmp_path, "BDGP", 37, 5, supported_ids=(0,)),
    }
    with pytest.raises(
        ValueError, match="NO_UTILITY_SUPPORTED_VALIDATION_EVIDENCE"
    ):
        build_formal_tuning_context_a2(
            "BDGP", tmp_path,
            carrier_loader=lambda dataset, seed, root: carriers[seed],
        )


def test_a2_context_copy_preserves_carrier_arrays_states_and_candidate_identity(tmp_path):
    carriers = {
        seed: _carrier(tmp_path, "BDGP", seed, 5, supported_ids=range(10))
        for seed in protocol.DEV_SEEDS
    }
    context, audit = build_formal_tuning_context_a2_with_audit(
        "BDGP", tmp_path,
        carrier_loader=lambda dataset, seed, root: carriers[seed],
    )
    for seed in protocol.DEV_SEEDS:
        assert context.carrier(seed).arrays is carriers[seed].arrays
        assert context.carrier(seed).state_path == carriers[seed].state_path
        assert context.carrier(seed).pre_gt_path == carriers[seed].pre_gt_path
        assert context.carrier(seed).state_file_sha256 == carriers[seed].state_file_sha256
        assert context.carrier(seed).pre_gt_sha256 == carriers[seed].pre_gt_sha256
    assert audit["amendment_id"] == "P3-C1-B1-A2"
    assert audit["full_gt_used"] is False
    assert audit["numeric_canonicalization"]["protocol_fingerprint_unchanged"]


def test_a2_context_remains_compatible_with_frozen_fold_semantics(tmp_path):
    carriers = {
        seed: _carrier(tmp_path, "BDGP", seed, 5, supported_ids=(0,))
        for seed in protocol.DEV_SEEDS
    }
    context = build_formal_tuning_context_a2(
        "BDGP", tmp_path,
        carrier_loader=lambda dataset, seed, root: carriers[seed],
    )
    carrier = context.carrier(11)
    carrier.adapter.sample_ids = carrier.arrays["sample_ids"]
    carrier.adapter.row_positions = lambda raw_ids: np.asarray(raw_ids, dtype=np.int64)
    payloads = trial_runner_a1.build_fold_semantics_a1(
        carrier.arrays, carrier.adapter, carrier.folds
    )
    assert set(payloads) == {"A", "B"}
    assert np.array_equal(payloads["A"]["class_signature"], np.full((5, 5), 0.2))


def test_a2_context_source_does_not_call_frozen_context_builder():
    source = Path(__import__("inspect").getsourcefile(build_formal_tuning_context_a2)).read_text(
        encoding="utf-8"
    )
    assert "frozen_context.build_formal_tuning_context(" not in source
    assert "frozen_context._carrier_context" in source

