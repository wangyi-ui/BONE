from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation.p3_c1_b1_optuna import amendment_a1, objective_a1, sparse_cv, trial_runner
from experiments.utility_relation.p3_c1_b1_optuna.trial_runner_a1 import build_fold_semantics_a1


CARRIER = Path(
    "outputs/bone_p3/p3_c1_b1_optuna/engineering_msrc_adapter_smoke/"
    "msrc/dev_seed11/carrier/carrier_pre_gt.npz"
)


def _real_payloads():
    with np.load(CARRIER, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in ("p_gen", "U_cycle")}
        sample_ids = np.asarray(archive["sample_ids"])
        labeled_ids = np.asarray(archive["labeled_ids"])
        targets = np.asarray(archive["labeled_targets"])
    adapter = b0.SampleIdAdapter(sample_ids)
    folds = sparse_cv.build_sparse_label_folds(labeled_ids, targets, expected_class_ids=range(7))
    return arrays, adapter, folds, build_fold_semantics_a1(arrays, adapter, folds), targets, labeled_ids


def test_real_fold_a_abstains_row175_but_continues_with_k7_denominator():
    _, _, _, payloads, _, _ = _real_payloads()
    fold_a = payloads["A"]
    assert fold_a["train_anchor_raw_ids"].tolist()[5] == 153
    assert fold_a["validation_raw_ids"].tolist()[5] == 175
    assert fold_a["validation_utility_mass"].tolist()[5] == 0.0
    pi = np.full((7, 6, 7), 1 / 7)
    score, audit = objective_a1.validation_j_with_audit(
        fold_a["U_validation"], pi, fold_a["validation_targets"]
    )
    assert score == pytest.approx(0.0)
    assert audit["anchor_scores"][5] == 0.0
    assert audit["fold_denominator"] == 7
    assert (audit["supported_anchor_count"], audit["abstained_anchor_count"]) == (6, 1)
    assert audit["coverage"] == pytest.approx(6 / 7)


def test_real_fold_b_retains_row175_unweighted_what_signature():
    arrays, adapter, _, payloads, _, _ = _real_payloads()
    fold_b = payloads["B"]
    assert fold_b["train_anchor_raw_ids"].tolist()[5] == 175
    assert fold_b["validation_raw_ids"].tolist()[5] == 153
    row175 = adapter.row_positions([175])[0]
    assert np.array_equal(fold_b["class_signature"][5], arrays["p_gen"][row175].mean(axis=0))
    assert fold_b["validation_utility_mass"][5] > 0
    pi = np.full((7, 6, 7), 1 / 7)
    _, audit = objective_a1.validation_j_with_audit(
        fold_b["U_validation"], pi, fold_b["validation_targets"]
    )
    assert (audit["supported_anchor_count"], audit["abstained_anchor_count"]) == (7, 0)
    assert audit["coverage"] == 1.0


def test_real_class_feasibility_positive_zero_passes_and_zero_zero_fails():
    arrays, adapter, _, _, targets, labeled_ids = _real_payloads()
    rows = adapter.row_positions(labeled_ids)
    masses = arrays["U_cycle"][rows].sum(axis=1)
    result = amendment_a1.validate_class_feasibility(targets, masses, expected_class_ids=range(7))
    assert result["per_class"]["5"]["anchor_utility_masses"][0] > 0
    assert result["per_class"]["5"]["anchor_utility_masses"][1] == 0
    bad = masses.copy(); bad[targets == 5] = 0
    with pytest.raises(ValueError, match="CLASS_ZERO_UTILITY_EVIDENCE"):
        amendment_a1.validate_class_feasibility(targets, bad, expected_class_ids=range(7))


def test_frozen_schedule_gives_exact_counts_initial_parity_and_isolation():
    class Loader:
        def __init__(self): self.epoch = 0
    models = []
    def load_branch():
        model = {"native": 0, "semantic": 0}; models.append(model); return model
    def native(model, _optimizer, loader):
        loader.epoch += 1; model["native"] += 1; return 1.0, f"order-{loader.epoch}"
    def semantic(model, _optimizer, _fold, _epoch):
        model["semantic"] += 1; return .5
    result = trial_runner.execute_isolated_schedule(
        load_branch, lambda _model: "reused-carrier", lambda _model: object(),
        lambda _model, _lr: object(), Loader, native, semantic, 1e-5, 5,
    )
    assert len({id(model) for model in models}) == 3
    assert len(set(result.initial_hashes.values())) == 1
    assert result.native_counts == {"base": 20, "A": 20, "B": 20}
    assert result.semantic_counts == {"A": 5, "B": 5}
