from types import SimpleNamespace

import numpy as np
import pytest

from . import formal_candidate_evaluator_r2
from . import protocol
from . import trial_runner_a1
from .canonical_fold_order_r3 import canonicalize_sparse_folds
from .formal_tuning_context import CarrierContext, FormalTuningContext
from .formal_tuning_context_r3 import (
    build_formal_tuning_context_r3,
    canonicalize_formal_tuning_context_r3,
)
from .sparse_cv import SparseFold
from .study_schema import CandidateConfig


LEX_20 = (
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 3, 4, 5, 6, 7, 8, 9,
)


class _Adapter:
    def __init__(self, size):
        self.sample_ids = np.arange(size, dtype=np.int64)

    def row_positions(self, raw_ids):
        return np.asarray(raw_ids, dtype=np.int64)


def _folds(K, order=None):
    order = tuple(range(K)) if order is None else tuple(order)
    first = {target: target for target in range(K)}
    second = {target: K + target for target in range(K)}
    a_train = tuple(first[target] for target in order)
    a_val = tuple(second[target] for target in order)
    fold_a = SparseFold("A", a_train, order, a_val, order)
    fold_b = SparseFold("B", a_val, order, a_train, order)
    return (fold_a, fold_b)


def _context(tmp_path, K=20, order=LEX_20, dataset="COIL20"):
    arrays = {
        "p_gen": np.full((2 * K + 1, 1, K), 1.0 / K),
        "U_cycle": np.ones((2 * K + 1, 1)),
    }
    views = (np.zeros((2 * K + 1, 1)),)
    carriers = {
        seed: CarrierContext(
            dataset=dataset,
            dev_seed=seed,
            directory=tmp_path,
            state_path=tmp_path / f"state{seed}",
            pre_gt_path=tmp_path / f"pre{seed}",
            audit_path=tmp_path / f"audit{seed}",
            arrays=arrays,
            views=views,
            adapter=_Adapter(2 * K + 1),
            spec=SimpleNamespace(batch_size=1),
            k=K,
            folds=_folds(K, order),
            state_file_sha256=f"s{seed}",
            pre_gt_sha256=f"p{seed}",
        )
        for seed in protocol.DEV_SEEDS
    }
    return FormalTuningContext(
        dataset, tmp_path, carriers, {11: {}, 37: {}}, "fp"
    )


def test_context_is_a_read_only_copy_and_only_folds_change(tmp_path):
    frozen = _context(tmp_path)
    canonical, audit = canonicalize_formal_tuning_context_r3(frozen)
    assert canonical is not frozen
    assert frozen.carrier(11).folds[0].train_targets == LEX_20
    assert canonical.carrier(11).folds[0].train_targets == tuple(range(20))
    for seed in protocol.DEV_SEEDS:
        before = frozen.carrier(seed)
        after = canonical.carrier(seed)
        assert after is not before
        assert after.arrays is before.arrays
        assert after.views is before.views
        assert after.state_path == before.state_path
        assert after.pre_gt_path == before.pre_gt_path
        assert after.state_file_sha256 == before.state_file_sha256
        assert after.pre_gt_sha256 == before.pre_gt_sha256
        assert audit["carriers"][str(seed)]["folds"]["A"]["pair_preservation"]
    with pytest.raises(TypeError):
        canonical.carriers[99] = canonical.carrier(11)


def test_frozen_context_builder_runs_before_r3_canonicalization(tmp_path):
    frozen = _context(tmp_path)
    calls = []

    def builder(dataset, root):
        calls.append((dataset, root))
        return frozen

    result = build_formal_tuning_context_r3("COIL20", tmp_path, builder=builder)
    assert calls == [("COIL20", tmp_path)]
    assert result.carrier(11).folds[0].train_targets == tuple(range(20))


def test_frozen_contiguous_guard_is_preserved_and_passes_after_r3(tmp_path):
    frozen = _context(tmp_path)
    carrier = frozen.carrier(11)
    with pytest.raises(ValueError, match="FOLD_CLASS_ORDER_MUST_BE_CONTIGUOUS"):
        trial_runner_a1.build_fold_semantics_a1(
            carrier.arrays, carrier.adapter, carrier.folds
        )
    canonical, _ = canonicalize_formal_tuning_context_r3(frozen)
    payloads = trial_runner_a1.build_fold_semantics_a1(
        canonical.carrier(11).arrays,
        canonical.carrier(11).adapter,
        canonical.carrier(11).folds,
    )
    assert set(payloads) == {"A", "B"}
    assert "FOLD_CLASS_ORDER_MUST_BE_CONTIGUOUS" in open(
        trial_runner_a1.__file__, encoding="utf-8"
    ).read()


def test_r3_context_still_executes_all_four_frozen_evaluator_branches(tmp_path):
    frozen = _context(tmp_path, K=3, order=(0, 1, 2), dataset="BDGP")
    canonical, _ = canonicalize_formal_tuning_context_r3(frozen)
    candidate = CandidateConfig(1e-5, 3)
    calls = []

    def runner(carrier, fold, received, device, payload):
        calls.append((carrier.dev_seed, fold))
        assert np.array_equal(payload["validation_targets"], np.arange(3))
        return {
            "J_OURS": 1.0,
            "semantic_steps": received.semantic_exposure,
            "native_epochs": protocol.REFINEMENT_HORIZON,
            "initial_state_sha256": f"state-{carrier.dev_seed}",
            "audit": {},
        }

    results = formal_candidate_evaluator_r2._evaluate_with_runner(
        canonical, candidate, "cpu", runner
    )
    assert set(calls) == {(11, "A"), (11, "B"), (37, "A"), (37, "B")}
    assert set(results) == set(calls)


@pytest.mark.parametrize("dataset,K", [("HandWrite", 10), ("BDGP", 5), ("MSRC", 7)])
def test_r3_identity_parity_for_already_canonical_contexts(tmp_path, dataset, K):
    frozen = _context(tmp_path, K=K, order=tuple(range(K)), dataset=dataset)
    canonical, audit = canonicalize_formal_tuning_context_r3(frozen)
    for seed in protocol.DEV_SEEDS:
        for before, after in zip(
            frozen.carrier(seed).folds, canonical.carrier(seed).folds
        ):
            assert after is before
            assert after == before
        assert all(
            item["ordering_changed"] is False
            for item in audit["carriers"][str(seed)]["folds"].values()
        )


def test_canonicalize_sparse_folds_rejects_duplicate_fold_names():
    folds = _folds(3)
    duplicate = (folds[0], SparseFold("A", (3, 4, 5), (0, 1, 2), (0, 1, 2), (0, 1, 2)))
    with pytest.raises(ValueError, match="DUPLICATE_FOLD_NAME"):
        canonicalize_sparse_folds(duplicate, 3)

