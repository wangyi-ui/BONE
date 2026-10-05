from pathlib import Path
import re

import numpy as np

from experiments.utility_relation import bone_p3_b0_generic_carrier as b0
from experiments.utility_relation.p3_c1_b1_optuna import objective
from experiments.utility_relation.p3_c1_b1_optuna.sparse_cv import build_sparse_label_folds
from experiments.utility_relation.p3_c1_b1_optuna.trial_runner import (
    build_fold_semantics, execute_isolated_schedule,
)


class Loader:
    def __init__(self):
        self.epoch = 0


def test_branch_schedule_counts_state_isolation_lr_and_frozen_u_contract():
    created_models, semantic_lrs, semantic_calls = [], [], []

    def load_branch():
        model = {"native": 0, "semantic": 0}
        created_models.append(model)
        return model

    def native_epoch(model, _optimizer, loader):
        loader.epoch += 1
        model["native"] += 1
        return 1.0, f"order-{loader.epoch}"

    def semantic_epoch(model, _optimizer, fold, epoch):
        model["semantic"] += 1
        semantic_calls.append((fold, epoch))
        return .5

    def semantic_optimizer(model, lr):
        semantic_lrs.append(lr)
        return object()

    result = execute_isolated_schedule(
        load_branch, lambda _model: "same-carrier-state", lambda _model: object(),
        semantic_optimizer, Loader, native_epoch, semantic_epoch, 1e-5, 5,
    )
    assert len({id(model) for model in created_models}) == 3
    assert len(set(result.initial_hashes.values())) == 1
    assert result.native_counts == {"base": 20, "A": 20, "B": 20}
    assert result.semantic_counts == {"A": 5, "B": 5}
    assert semantic_lrs == [1e-5, 1e-5]
    assert len(semantic_calls) == 10


def test_fold_semantics_excludes_validation_anchors_and_keeps_u_immutable():
    p_gen = np.array([
        [[.8, .2], [.7, .3]], [[.75, .25], [.65, .35]],
        [[.2, .8], [.3, .7]], [[.25, .75], [.35, .65]],
        [[.6, .4], [.4, .6]], [[.55, .45], [.45, .55]],
    ])
    utility = np.ones((6, 2))
    before = utility.copy()
    folds = build_sparse_label_folds((0, 1, 2, 3), (0, 0, 1, 1), expected_class_ids=(0, 1))
    payloads = build_fold_semantics({"p_gen": p_gen, "U_cycle": utility}, b0.SampleIdAdapter(np.arange(6)), folds)
    for payload in payloads.values():
        assert set(payload["query_rows"]) == {4, 5}
        assert not set(payload["validation_rows"]) & set(payload["query_rows"])
        assert payload["T_rel"].shape == (2, 2, 2)
    assert np.array_equal(utility, before)


def test_frozen_objective_is_imported_and_no_four_score_aggregation_is_faked():
    import experiments.utility_relation.p3_c1_b1_optuna.trial_runner as runner
    assert runner.objective.validation_j is objective.validation_j
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "aggregate_candidate_scores" not in source
    assert not re.search(r"\b(?:ACC|NMI|Purity|ARI)\b", source)
