import inspect

import pytest

from experiments.utility_relation import (
    p4_b0_early_stopping as es,
)

from experiments.utility_relation import (
    run_bone_p4_b2_uases_confirmatory as runner,
)


def test_exact_confirmatory_seeds():
    assert runner.CONFIRMATORY_SEEDS == (
        113,
        127,
        139,
    )


def test_exact_confirmatory_datasets():
    assert runner.DATASETS == (
        "MSRC",
        "HandWrite",
        "MNIST-USPS",
    )


def test_protocol_is_confirmatory():
    p = runner.load_protocol()

    assert p["confirmatory"] is True
    assert p[
        "paper_formal_result"
    ] is True

    assert p[
        "method_development"
    ] is False


def test_frozen_uases_contract():
    p = runner.load_protocol()

    u = p["uases"]

    assert u[
        "checkpoint_interval"
    ] == 5

    assert u[
        "H_max_safety_cap"
    ] == 100

    assert u[
        "patience_checkpoints"
    ] == 4

    assert abs(
        u[
            "improvement_epsilon"
        ]
        - 1e-6
    ) < 1e-15


def test_msrc_config():
    c = runner.cell_config(
        "MSRC",
        113,
    )

    assert c.semantic_lr == 5e-4
    assert c.semantic_exposure == 1
    assert c.fixed_h == 5


def test_handwrite_config():
    c = runner.cell_config(
        "HandWrite",
        127,
    )

    assert c.semantic_lr == 5e-6
    assert c.semantic_exposure == 1
    assert c.fixed_h == 20


def test_mnist_config():
    c = runner.cell_config(
        "MNIST-USPS",
        139,
    )

    assert c.semantic_lr == 2e-4
    assert c.semantic_exposure == 5
    assert c.fixed_h == 40


def test_invalid_seed_fails_closed():
    with pytest.raises(
        ValueError,
        match="UNREGISTERED_CONFIRMATORY_SEED",
    ):
        runner.cell_config(
            "MSRC",
            20,
        )


def test_invalid_dataset_fails_closed():
    with pytest.raises(
        ValueError,
        match="UNREGISTERED_CONFIRMATORY_DATASET",
    ):
        runner.cell_config(
            "BDGP",
            113,
        )


def test_cell_end_requires_both_stops_and_fixed_h():
    base = (
        runner
        .b1_runner
        .VirtualStopController()
    )

    ours = (
        runner
        .b1_runner
        .VirtualStopController()
    )

    # No frozen decisions.
    assert runner.should_end_cell(
        40,
        20,
        base,
        ours,
    ) is False

    base.frozen = True
    base.selected_h = 10
    base.stop_h = 30

    # Only BASE stopped.
    assert runner.should_end_cell(
        40,
        20,
        base,
        ours,
    ) is False

    ours.frozen = True
    ours.selected_h = 20
    ours.stop_h = 40

    assert runner.should_end_cell(
        40,
        20,
        base,
        ours,
    ) is True


def test_cell_end_respects_fixed_h():
    base = (
        runner
        .b1_runner
        .VirtualStopController()
    )

    ours = (
        runner
        .b1_runner
        .VirtualStopController()
    )

    base.frozen = True
    base.selected_h = 5
    base.stop_h = 25

    ours.frozen = True
    ours.selected_h = 10
    ours.stop_h = 30

    # Fixed comparator H=40 not yet materialized.
    assert runner.should_end_cell(
        30,
        40,
        base,
        ours,
    ) is False

    assert runner.should_end_cell(
        40,
        40,
        base,
        ours,
    ) is True


def test_runner_has_no_gt_metric_import_path():
    source = inspect.getsource(
        runner
    ).lower()

    forbidden = (
        "scipy.io",
        "loadmat(",
        "compute_metrics(",
        "adjusted_rand_score",
        "normalized_mutual_info_score",
        "linear_sum_assignment",
        "evaluate_bone_p4_b1_uases_shadow_pilot.py",
    )

    for token in forbidden:
        assert token not in source


def test_output_namespace_is_p4_b2():
    assert (
        "p4_b2_uases_confirmatory"
        in str(
            runner.DEFAULT_OUTPUT_ROOT
        )
    )


def test_early_stopping_constants_unchanged():
    assert es.CHECKPOINT_INTERVAL == 5
    assert es.H_MAX == 100
    assert es.PATIENCE_CHECKPOINTS == 4

    assert (
        es.IMPROVEMENT_EPSILON
        == 1e-6
    )


def test_confirmatory_semantic_orders_support_unseen_seeds():
    import numpy as np

    for seed in runner.CONFIRMATORY_SEEDS:

        orders = (
            runner.confirmatory_semantic_orders(
                query_count=37,
                seed=seed,
                exposure=5,
            )
        )

        assert len(
            orders
        ) == 5

        for order in orders:

            assert order.shape == (
                37,
            )

            assert order.dtype == np.int64

            assert np.array_equal(
                np.sort(order),
                np.arange(
                    37,
                    dtype=np.int64,
                ),
            )


def test_confirmatory_semantic_orders_exact_historical_parity():
    import numpy as np

    from experiments.utility_relation import (
        bone_p3_c0_cross_dataset_formal_protocol
        as frozen,
    )

    for q in (
        17,
        196,
        1980,
    ):

        for seed in (
            20,
            30,
            50,
        ):

            old = np.asarray(
                frozen.semantic_orders(
                    q,
                    seed,
                ),
                dtype=np.int64,
            )

            new = np.stack(
                runner.confirmatory_semantic_orders(
                    query_count=q,
                    seed=seed,
                    exposure=old.shape[0],
                ),
                axis=0,
            )

            assert np.array_equal(
                old,
                new,
            )
