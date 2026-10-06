import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import (
    bone_p3_b0_generic_carrier as b0,
)

from experiments.utility_relation import (
    p4_b0_early_stopping as es,
)


PREREG = Path(
    "/root/autodl-tmp/BONE/"
    "experiment_freeze/"
    "bone_p4_b0_utility_aligned_shared_early_stopping_definition_preregistered_20261006/"
    "PROTOCOL.json"
)


def random_q(
    n=24,
    v=4,
    k=3,
    seed=7,
):
    rng = np.random.default_rng(
        seed
    )

    x = rng.random(
        (n, v, k)
    )

    # [N,V,K]
    return (
        x
        / x.sum(
            axis=2,
            keepdims=True,
        )
    )


def test_protocol_constants_match_preregistration():
    p = json.loads(
        PREREG.read_text()
    )

    c = es.protocol_constants()

    assert c[
        "checkpoint_interval"
    ] == (
        p["checkpoint_schedule"]
        ["checkpoint_interval"]
    )

    assert c["h_max"] == (
        p["checkpoint_schedule"]
        ["maximum_refinement_h"]
    )

    assert c[
        "patience_checkpoints"
    ] == (
        p["stopping_rule"]
        ["patience_checkpoints"]
    )

    assert c[
        "patience_native_epochs"
    ] == 20

    assert abs(
        c["improvement_epsilon"]
        - p["stopping_rule"]
        ["strict_improvement_epsilon"]
    ) < 1e-15

    assert c["uses_gt"] is False
    assert c[
        "uses_sparse_labels"
    ] is False
    assert c["uses_t_rel"] is False
    assert c[
        "uses_semantic_loss"
    ] is False

    assert c[
        "arm_specific_formula"
    ] is False


def test_freeze_utility_is_read_only_copy():
    q = random_q(
        n=16,
        v=4,
        k=3,
    )

    actions, _, _ = (
        b0.action_arrays(4)
    )

    u = np.full(
        (
            16,
            len(actions),
        ),
        0.5,
        dtype=np.float32,
    )

    frozen = es.freeze_utility(
        u
    )

    assert frozen.shape == u.shape
    assert frozen.dtype == np.float64
    assert frozen.flags.writeable is False
    assert not np.shares_memory(
        frozen,
        u,
    )

    with pytest.raises(
        ValueError
    ):
        frozen[0, 0] = 0.2


def test_freeze_utility_rejects_invalid_values():
    with pytest.raises(
        ValueError
    ):
        es.freeze_utility(
            np.zeros(
                (3, 2),
                dtype=np.float64,
            )
        )

    with pytest.raises(
        ValueError
    ):
        es.freeze_utility(
            np.asarray(
                [[1.2, 0.0]]
            )
        )

    with pytest.raises(
        ValueError
    ):
        es.freeze_utility(
            np.asarray(
                [[np.nan, 0.2]]
            )
        )


def test_action_posteriors_use_frozen_action_count():
    q = random_q(
        n=10,
        v=4,
        k=5,
    )

    actions, _, _ = (
        b0.action_arrays(4)
    )

    p_gen, p_ver = (
        es.action_posteriors(
            q
        )
    )

    assert p_gen.shape == (
        10,
        len(actions),
        5,
    )

    assert p_ver.shape == (
        10,
        len(actions),
        5,
    )

    assert np.allclose(
        p_gen.sum(axis=2),
        1.0,
        atol=1e-12,
    )

    assert np.allclose(
        p_ver.sum(axis=2),
        1.0,
        atol=1e-12,
    )


def test_scv_equals_one_for_identical_view_posteriors():
    rng = np.random.default_rng(
        2
    )

    base = rng.random(
        (20, 4)
    )

    base = (
        base
        / base.sum(
            axis=1,
            keepdims=True,
        )
    )

    # [N,V,K] with every view identical.
    q = np.repeat(
        base[:, None, :],
        4,
        axis=1,
    )

    actions, _, _ = (
        b0.action_arrays(4)
    )

    u = es.freeze_utility(
        np.ones(
            (
                20,
                len(actions),
            )
        )
    )

    score = (
        es.utility_weighted_cross_view_score(
            q,
            u,
        )
    )

    assert score == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_scores_remain_in_unit_interval():
    q = random_q(
        n=32,
        v=4,
        k=6,
    )

    actions, _, _ = (
        b0.action_arrays(4)
    )

    u = es.freeze_utility(
        np.linspace(
            0.1,
            0.9,
            32 * len(actions),
        ).reshape(
            32,
            len(actions),
        )
    )

    pred0 = es.hard_prediction(
        q
    )

    # Perturb but keep [N,V,K] distributions.
    q2 = np.roll(
        q,
        shift=1,
        axis=0,
    )

    record = (
        es.evaluate_checkpoint(
            q2,
            u,
            pred0,
        )
    )

    for key in (
        "S_cv",
        "S_bal",
        "S_stab",
        "Q_stop",
    ):
        assert (
            0.0
            <= record[key]
            <= 1.0
        )

    assert record[
        "prediction"
    ].shape == (32,)


def test_cluster_balance_uniform_is_one():
    # 6 samples, 3 classes; consensus posterior perfectly balanced.
    q_bar = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # [N,V,K], identical two-view posterior.
    q = np.repeat(
        q_bar[:, None, :],
        2,
        axis=1,
    )

    assert (
        es.cluster_balance_score(
            q
        )
        == pytest.approx(
            1.0,
            abs=1e-10,
        )
    )


def test_temporal_stability_identical_predictions_is_one():
    y = np.asarray(
        [
            0,
            1,
            1,
            2,
            0,
            2,
        ],
        dtype=np.int64,
    )

    assert (
        es.temporal_stability_score(
            y,
            y.copy(),
        )
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )


def test_q_stop_is_exact_geometric_mean():
    got = es.q_stop_score(
        0.8,
        0.9,
        0.7,
    )

    expected = (
        0.8
        * 0.9
        * 0.7
    ) ** (
        1.0 / 3.0
    )

    assert got == pytest.approx(
        expected,
        abs=1e-15,
    )


def test_patience_stops_after_four_consecutive_non_improvements():
    state = (
        es.EarlyStoppingState()
    )

    scores = {
        5: 0.50,
        10: 0.60,
        15: 0.59,
        20: 0.58,
        25: 0.57,
        30: 0.56,
    }

    records = {}

    for h, score in scores.items():
        records[h] = (
            state.update(
                h,
                score,
            )
        )

        if records[h][
            "should_stop"
        ]:
            break

    assert state.stopped is True
    assert state.stop_h == 30
    assert state.best_h == 10

    assert records[30][
        "patience_triggered"
    ] is True

    assert records[30][
        "return_h"
    ] == 10


def test_improvement_resets_patience():
    state = (
        es.EarlyStoppingState()
    )

    sequence = [
        (5, 0.50),
        (10, 0.49),
        (15, 0.48),
        (20, 0.60),
        (25, 0.59),
        (30, 0.58),
        (35, 0.57),
        (40, 0.56),
    ]

    for h, score in sequence:
        record = state.update(
            h,
            score,
        )

        if h == 20:
            assert (
                record["bad_count"]
                == 0
            )

        if record[
            "should_stop"
        ]:
            break

    assert state.stop_h == 40
    assert state.best_h == 20


def test_tie_within_epsilon_keeps_earlier_checkpoint():
    state = (
        es.EarlyStoppingState()
    )

    first = state.update(
        5,
        0.5000000,
    )

    second = state.update(
        10,
        0.5000005,
    )

    assert first[
        "improved"
    ] is True

    assert second[
        "improved"
    ] is False

    assert state.best_h == 5


def test_hmax_returns_best_seen_checkpoint():
    state = (
        es.EarlyStoppingState(
            patience_checkpoints=100,
        )
    )

    best_h = 35

    for h in range(
        5,
        101,
        5,
    ):
        score = (
            0.9
            if h == best_h
            else 0.5
        )

        record = state.update(
            h,
            score,
        )

    assert state.stopped is True
    assert state.stop_h == 100
    assert state.best_h == 35
    assert record[
        "hmax_reached"
    ] is True
    assert record[
        "return_h"
    ] == 35


def test_base_and_ours_use_identical_state_machine_class():
    base = (
        es.EarlyStoppingState()
    )

    ours = (
        es.EarlyStoppingState()
    )

    assert type(base) is type(ours)

    assert (
        base.patience_checkpoints
        ==
        ours.patience_checkpoints
    )

    assert (
        base.improvement_epsilon
        ==
        ours.improvement_epsilon
    )

    assert (
        base.checkpoint_interval
        ==
        ours.checkpoint_interval
    )

    assert (
        base.h_max
        ==
        ours.h_max
    )


def test_different_trajectories_may_return_different_h():
    base = (
        es.EarlyStoppingState()
    )

    ours = (
        es.EarlyStoppingState()
    )

    base_scores = [
        0.70,
        0.75,
        0.74,
        0.73,
        0.72,
        0.71,
    ]

    ours_scores = [
        0.60,
        0.65,
        0.70,
        0.75,
        0.74,
        0.73,
        0.72,
        0.71,
    ]

    for idx, score in enumerate(
        base_scores,
        start=1,
    ):
        h = idx * 5

        r = base.update(
            h,
            score,
        )

        if r["should_stop"]:
            break

    for idx, score in enumerate(
        ours_scores,
        start=1,
    ):
        h = idx * 5

        r = ours.update(
            h,
            score,
        )

        if r["should_stop"]:
            break

    assert base.best_h == 10
    assert ours.best_h == 20

    # Same mechanism, different trajectory -> different selected H is legal.
    assert (
        base.patience_checkpoints
        ==
        ours.patience_checkpoints
        == 4
    )


def test_public_stopping_api_has_no_gt_or_semantic_inputs():
    targets = (
        es.utility_weighted_cross_view_score,
        es.cluster_balance_score,
        es.temporal_stability_score,
        es.q_stop_score,
        es.evaluate_checkpoint,
        es.EarlyStoppingState.update,
    )

    forbidden = (
        "ground_truth",
        "y_true",
        "t_rel",
        "semantic_loss",
        "sparse_target",
        "anchor_target",
    )

    for fn in targets:

        signature = str(
            inspect.signature(fn)
        ).lower()

        for token in forbidden:
            assert token not in signature


def test_shape_failure_is_fail_closed():
    q = random_q(
        n=12,
        v=4,
        k=3,
    )

    actions, _, _ = (
        b0.action_arrays(4)
    )

    wrong_u = np.ones(
        (
            11,
            len(actions),
        )
    )

    with pytest.raises(
        ValueError
    ):
        es.utility_weighted_cross_view_score(
            q,
            wrong_u,
        )

    with pytest.raises(
        ValueError
    ):
        es.temporal_stability_score(
            np.zeros(
                12,
                dtype=np.int64,
            ),
            np.zeros(
                11,
                dtype=np.int64,
            ),
        )


def test_invalid_checkpoint_schedule_fails_closed():
    state = (
        es.EarlyStoppingState()
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            6,
            0.5,
        )

    state.update(
        5,
        0.5,
    )

    with pytest.raises(
        ValueError
    ):
        state.update(
            15,
            0.6,
        )
