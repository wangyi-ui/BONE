from collections import Counter

import numpy as np
import pytest

from .canonical_fold_order_r3 import (
    CLASS_COVERAGE_ERROR,
    canonicalize_sparse_fold,
)
from .sparse_cv import SparseFold


COIL20_BEFORE = (
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 3, 4, 5, 6, 7, 8, 9,
)


def _fold(order=COIL20_BEFORE):
    train_pairs = [(target, 1000 + target) for target in order]
    validation_pairs = [(target, 2000 + target) for target in order]
    return SparseFold(
        "A",
        tuple(raw for target, raw in train_pairs),
        tuple(target for target, raw in train_pairs),
        tuple(raw for target, raw in validation_pairs),
        tuple(target for target, raw in validation_pairs),
    )


def test_coil20_exact_lexicographic_to_numeric_order():
    before = _fold()
    after, audit = canonicalize_sparse_fold(before, 20)
    expected = tuple(range(20))
    assert before.train_targets == COIL20_BEFORE
    assert after.train_targets == expected
    assert after.val_targets == expected
    assert after.train_raw_row_ids == tuple(1000 + target for target in expected)
    assert after.val_raw_row_ids == tuple(2000 + target for target in expected)
    assert audit["before_order"]["train"] == list(COIL20_BEFORE)
    assert audit["after_order"]["train"] == list(expected)
    assert audit["ordering_changed"] is True


def test_pairs_labels_and_anchor_membership_are_exactly_preserved():
    before = _fold()
    after, audit = canonicalize_sparse_fold(before, 20)
    assert Counter(zip(before.train_targets, before.train_raw_row_ids)) == Counter(
        zip(after.train_targets, after.train_raw_row_ids)
    )
    assert Counter(zip(before.val_targets, before.val_raw_row_ids)) == Counter(
        zip(after.val_targets, after.val_raw_row_ids)
    )
    assert Counter(before.train_targets) == Counter(after.train_targets)
    assert Counter(before.val_targets) == Counter(after.val_targets)
    assert Counter(before.train_raw_row_ids) == Counter(after.train_raw_row_ids)
    assert Counter(before.val_raw_row_ids) == Counter(after.val_raw_row_ids)
    assert audit["pair_set_hash_before"] == audit["pair_set_hash_after"]
    assert audit["label_values_changed"] is False
    assert audit["anchor_membership_changed"] is False


@pytest.mark.parametrize("dataset,K", [("HandWrite", 10), ("BDGP", 5), ("MSRC", 7)])
def test_already_numeric_dataset_fold_is_exact_identity(dataset, K):
    del dataset
    order = tuple(range(K))
    before = SparseFold(
        "A", tuple(range(K)), order, tuple(range(K, 2 * K)), order
    )
    after, audit = canonicalize_sparse_fold(before, K)
    assert after is before
    assert after == before
    assert audit["ordering_changed"] is False
    assert audit["pair_set_hash_before"] == audit["pair_set_hash_after"]


@pytest.mark.parametrize(
    "targets,K",
    [((0, 1, 1), 3), ((0, 1), 3), ((0, 1, 3), 3), ((0.5, 1, 2), 3)],
)
def test_noncanonical_class_coverage_fails_closed(targets, K):
    fold = SparseFold(
        "A", tuple(range(len(targets))), targets,
        tuple(range(10, 10 + len(targets))), targets,
    )
    with pytest.raises(ValueError, match=CLASS_COVERAGE_ERROR):
        canonicalize_sparse_fold(fold, K)


def test_numpy_integer_labels_are_preserved_not_reencoded():
    targets = tuple(np.int64(value) for value in (0, 1, 2))
    fold = SparseFold("A", (3, 4, 5), targets, (6, 7, 8), targets)
    after, _ = canonicalize_sparse_fold(fold, 3)
    assert after is fold
    assert all(isinstance(value, np.int64) for value in after.train_targets)

