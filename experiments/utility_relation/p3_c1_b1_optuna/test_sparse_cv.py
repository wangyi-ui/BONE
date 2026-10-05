import pytest

from experiments.utility_relation.p3_c1_b1_optuna.sparse_cv import build_sparse_label_folds, split_audit_hash


def _folds(rows=(11, 10, 21, 20), targets=(0, 0, 1, 1)):
    return build_sparse_label_folds(rows, targets, expected_class_ids=(0, 1))


def test_exact_two_labels_and_deterministic_input_order_invariance():
    first = _folds()
    shuffled = _folds((20, 11, 10, 21), (1, 0, 0, 1))
    assert first == shuffled
    assert split_audit_hash(first) == split_audit_hash(shuffled)


def test_a_b_assignments_and_disjointness():
    fold_a, fold_b = _folds()
    assert fold_a.fold_name == "A"
    assert fold_a.train_raw_row_ids == (10, 20)
    assert fold_a.val_raw_row_ids == (11, 21)
    assert fold_b.fold_name == "B"
    assert fold_b.train_raw_row_ids == (11, 21)
    assert fold_b.val_raw_row_ids == (10, 20)
    assert not set(fold_a.train_raw_row_ids) & set(fold_a.val_raw_row_ids)
    assert set(fold_a.train_raw_row_ids + fold_b.train_raw_row_ids) == {10, 11, 20, 21}
    assert set(fold_a.val_raw_row_ids + fold_b.val_raw_row_ids) == {10, 11, 20, 21}
    assert fold_a.train_targets == fold_a.val_targets == (0, 1)


def test_duplicate_row_and_wrong_per_class_counts_fail_closed():
    with pytest.raises(ValueError, match="DUPLICATE"):
        build_sparse_label_folds((1, 1), (0, 0))
    with pytest.raises(ValueError, match="EXACTLY_TWO"):
        build_sparse_label_folds((1, 2, 3), (0, 0, 0))
    with pytest.raises(ValueError, match="EXACTLY_TWO"):
        build_sparse_label_folds((1, 2, 3), (0, 0, 1))


def test_shape_and_expected_class_mismatches_fail_closed():
    with pytest.raises(ValueError, match="SHAPE"):
        build_sparse_label_folds((1, 2), (0,))
    with pytest.raises(ValueError, match="MISSING"):
        build_sparse_label_folds((1, 2), (0, 0), expected_class_ids=(0, 1))
