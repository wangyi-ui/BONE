import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation import (
    p4_b0_early_stopping as es,
)

from experiments.utility_relation import (
    run_bone_p4_b1_uases_shadow_pilot as runner,
)


def test_checkpoint_grid_matches_preregistration():
    assert runner.H_GRID == tuple(
        range(
            0,
            101,
            5,
        )
    )

    assert runner.SCORED_H_GRID == tuple(
        range(
            5,
            101,
            5,
        )
    )


def test_exact_three_preregistered_cells():
    cells = runner.pilot_cells()

    assert len(cells) == 3

    got = {
        (
            c.dataset,
            c.training_seed,
        ):
            (
                c.fixed_h,
                c.semantic_lr,
                c.semantic_exposure,
            )
        for c in cells
    }

    assert got[
        ("MSRC", 20)
    ] == (
        5,
        5e-4,
        1,
    )

    assert got[
        ("HandWrite", 20)
    ] == (
        20,
        5e-6,
        1,
    )

    assert got[
        ("MNIST-USPS", 50)
    ] == (
        40,
        2e-4,
        5,
    )


def test_unregistered_dataset_seed_pair_fails():
    with pytest.raises(
        ValueError,
        match="UNREGISTERED_P4_B1_CELL",
    ):
        runner.get_cell(
            "MSRC",
            50,
        )

    with pytest.raises(
        ValueError,
        match="UNREGISTERED_P4_B1_CELL",
    ):
        runner.get_cell(
            "HandWrite",
            50,
        )


def test_virtual_stop_freezes_first_decision():
    controller = (
        runner.VirtualStopController()
    )

    scores = {
        5: 0.50,
        10: 0.60,
        15: 0.59,
        20: 0.58,
        25: 0.57,
        30: 0.56,
    }

    for h, score in scores.items():
        out = controller.observe(
            h,
            score,
        )

    assert controller.frozen is True
    assert controller.selected_h == 10
    assert controller.stop_h == 30

    # A much better post-stop score must NOT alter selected H.
    later = controller.observe(
        35,
        1.0,
    )

    assert later[
        "state_updated"
    ] is False

    assert controller.selected_h == 10
    assert controller.stop_h == 30


def test_virtual_stop_hmax_case():
    controller = (
        runner.VirtualStopController(
            state=
                es.EarlyStoppingState(
                    patience_checkpoints=100
                )
        )
    )

    for h in range(
        5,
        101,
        5,
    ):
        controller.observe(
            h,
            0.9
            if h == 35
            else 0.5,
        )

    assert controller.frozen is True
    assert controller.selected_h == 35
    assert controller.stop_h == 100


def test_prediction_archive_exact_grid():
    n = 12

    sample_ids = np.arange(
        n,
        dtype=np.int64,
    )

    predictions = {
        h:
            np.full(
                n,
                h // 5,
                dtype=np.int64,
            )
        for h in runner.H_GRID
    }

    payload = (
        runner.prediction_archive_payload(
            sample_ids,
            predictions,
        )
    )

    assert payload[
        "sample_ids"
    ].shape == (n,)

    for h in runner.H_GRID:
        assert payload[
            f"pred_H{h}"
        ].shape == (n,)


def test_prediction_archive_missing_checkpoint_fails():
    predictions = {
        h:
            np.zeros(
                8,
                dtype=np.int64,
            )
        for h in runner.H_GRID
        if h != 50
    }

    with pytest.raises(
        ValueError,
        match="prediction checkpoints",
    ):
        runner.prediction_archive_payload(
            np.arange(
                8,
                dtype=np.int64,
            ),
            predictions,
        )


def test_prediction_archive_wrong_shape_fails():
    predictions = {
        h:
            np.zeros(
                8,
                dtype=np.int64,
            )
        for h in runner.H_GRID
    }

    predictions[25] = (
        np.zeros(
            7,
            dtype=np.int64,
        )
    )

    with pytest.raises(
        ValueError,
        match="prediction must",
    ):
        runner.prediction_archive_payload(
            np.arange(
                8,
                dtype=np.int64,
            ),
            predictions,
        )


def test_same_uases_class_used_for_both_arms():
    controllers = {
        arm:
            runner.VirtualStopController()
        for arm in runner.ARMS
    }

    assert type(
        controllers["BASE"].state
    ) is type(
        controllers["OURS"].state
    )

    assert (
        controllers[
            "BASE"
        ].state.patience_checkpoints
        ==
        controllers[
            "OURS"
        ].state.patience_checkpoints
        ==
        4
    )


def test_protocol_requires_shadow_mode():
    p = runner.load_protocol()

    assert p[
        "trajectory_policy"
    ]["mode"] == (
        "FULL_SHADOW_TRAJECTORY"
    )

    assert p[
        "trajectory_policy"
    ]["actual_training_horizon"] == 100

    assert p[
        "trajectory_policy"
    ]["UASES_virtual_stop"] is True


def test_protocol_gt_embargo():
    p = runner.load_protocol()

    assert p[
        "gt_embargo"
    ][
        "full_GT_read_before_global_pre_gt_seal"
    ] is False

    assert p[
        "gt_embargo"
    ][
        "GT_may_change_selected_h"
    ] is False

    assert p[
        "gt_embargo"
    ][
        "GT_may_change_UASES_formula"
    ] is False


def test_runner_has_no_gt_metric_imports():
    source = inspect.getsource(
        runner
    )

    forbidden = (
        "scipy.io",
        "loadmat(",
        "compute_metrics(",
        "adjusted_rand_score",
        "normalized_mutual_info_score",
        "linear_sum_assignment",
    )

    for token in forbidden:
        assert token not in source


def test_runner_no_evaluator_import():
    source = inspect.getsource(
        runner
    ).lower()

    assert (
        "evaluate_bone_p3"
        not in source
    )

    assert (
        "evaluate_final_formal"
        not in source
    )


def test_public_run_api_requires_registered_cell():
    signature = str(
        inspect.signature(
            runner.run_cell
        )
    )

    assert "dataset" in signature
    assert "training_seed" in signature


def test_ast_has_no_scipy_import():
    path = Path(
        runner.__file__
    )

    tree = ast.parse(
        path.read_text()
    )

    modules = []

    for node in ast.walk(tree):

        if isinstance(
            node,
            ast.Import,
        ):
            modules.extend(
                alias.name
                for alias
                in node.names
            )

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            modules.append(
                node.module
                or ""
            )

    assert all(
        not module.startswith(
            "scipy"
        )
        for module in modules
    )


def test_output_root_is_p4_b1_namespace():
    assert (
        "p4_b1_uases_shadow_pilot"
        in str(
            runner.DEFAULT_OUTPUT_ROOT
        )
    )
