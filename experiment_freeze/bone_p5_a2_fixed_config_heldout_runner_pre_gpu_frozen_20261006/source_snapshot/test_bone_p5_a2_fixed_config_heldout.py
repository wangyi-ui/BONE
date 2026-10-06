import inspect

import numpy as np
import pytest

from experiments.utility_relation import (
    bone_p3_c0_cross_dataset_formal_protocol
    as frozen_protocol,
)

from experiments.utility_relation import (
    run_bone_p5_a2_fixed_config_heldout
    as runner,
)

from experiments.utility_relation import (
    run_bone_p4_b2_uases_confirmatory
    as p4b2,
)


def test_exact_seeds():
    assert runner.SEEDS == (
        149,
        163,
        179,
    )


def test_exact_datasets():
    assert runner.DATASETS == (
        "MSRC",
        "BDGP",
        "HandWrite",
        "Caltech-5V",
        "MNIST-USPS",
        "COIL20",
        "ALOI",
        "CCV",
    )


@pytest.mark.parametrize(
    "dataset,lr,e,h",
    [
        ("MSRC", 5e-4, 1, 5),
        ("BDGP", 2e-3, 1, 15),
        ("HandWrite", 5e-6, 1, 20),
        ("Caltech-5V", 1e-3, 10, 30),
        ("MNIST-USPS", 2e-4, 5, 40),
        ("COIL20", 3e-5, 1, 40),
        ("ALOI", 3e-4, 10, 50),
        ("CCV", 1e-4, 5, 65),
    ],
)
def test_frozen_configs(
    dataset,
    lr,
    e,
    h,
):
    cfg = runner.cell_config(
        dataset,
        149,
    )

    assert abs(
        cfg.semantic_lr - lr
    ) < 1e-15

    assert (
        cfg.semantic_exposure
        == e
    )

    assert (
        cfg.refinement_horizon
        == h
    )


def test_invalid_seed_fails():
    with pytest.raises(
        ValueError,
        match="UNREGISTERED_P5_SEED",
    ):
        runner.cell_config(
            "MSRC",
            20,
        )


def test_invalid_dataset_fails():
    with pytest.raises(
        ValueError,
        match="UNREGISTERED_P5_DATASET",
    ):
        runner.cell_config(
            "Fake",
            149,
        )


def test_schedule_extension_matches_historical_rule():
    for seed in (
        20,
        30,
        50,
    ):

        old = np.asarray(
            frozen_protocol.semantic_orders(
                41,
                seed,
            ),
            dtype=np.int64,
        )

        new = np.stack(
            p4b2.confirmatory_semantic_orders(
                41,
                seed,
                old.shape[0],
            ),
            axis=0,
        )

        assert np.array_equal(
            old,
            new,
        )


def test_heldout_schedules_deterministic():
    for seed in runner.SEEDS:

        a = (
            p4b2.confirmatory_semantic_orders(
                53,
                seed,
                10,
            )
        )

        b = (
            p4b2.confirmatory_semantic_orders(
                53,
                seed,
                10,
            )
        )

        assert all(
            np.array_equal(
                x,
                y,
            )
            for x, y
            in zip(a, b)
        )


def test_runner_has_no_full_gt_metric_path():

    source = inspect.getsource(
        runner
    ).lower()

    forbidden = (
        "loadmat(",
        "compute_metrics(",
        "adjusted_rand_score",
        "normalized_mutual_info_score",
        "linear_sum_assignment",
        "clusteringtest",
    )

    for token in forbidden:
        assert token not in source
