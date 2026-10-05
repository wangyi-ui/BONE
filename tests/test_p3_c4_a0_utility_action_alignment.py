from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.utility_relation import run_bone_p3_c0_cross_dataset_formal as frozen
from experiments.utility_relation.p3_c4_a0_utility_action_alignment import alignment
from experiments.utility_relation.p3_c4_a0_utility_action_alignment.protocol import (
    DATASETS, FROZEN_SOURCE_HASHES, SEEDS, manifest_cells, verify_frozen_sources)
from experiments.utility_relation.p3_c4_a0_utility_action_alignment.run_diagnostic import diagnose_cell
from experiments.utility_relation.p3_c4_a0_utility_action_alignment.semantic_direction import (
    semantic_margins, utility_diagnostics)
from experiments.utility_relation.p3_c4_a0_utility_action_alignment.summarize_diagnostic import summarize
from method.bone import BONE


def _synthetic():
    torch.manual_seed(3)
    model = BONE([3, 4], k=2, a=1, v=2, layer=1, device="cpu", info=False)
    views = [torch.randn(8, 3), torch.randn(8, 4)]
    anchor = torch.tensor([0, 1]); query = torch.tensor([2, 3, 4, 5, 6, 7])
    # Every action has nonzero sparse-anchor class evidence and all classes exist.
    t = torch.tensor([[[.8, .2], [.2, .8]], [[.7, .3], [.3, .7]], [[.6, .4], [.4, .6]],
                      [[.4, .6], [.6, .4]], [[.3, .7], [.7, .3]], [[.2, .8], [.8, .2]]])
    full_u = torch.tensor([[.9, .1], [.1, .8], [.7, .2], [.2, .6], [.5, .4], [.3, .9], [.8, .1], [.1, .7]])
    return model, views, query, anchor, t, full_u, full_u[query], torch.tensor([0, 1])


def test_frozen_sources_and_target_freeze_unchanged():
    assert verify_frozen_sources() == FROZEN_SOURCE_HASHES


def test_true_loss_has_numeric_parity_with_frozen_current():
    model, views, query, anchor, t, _, u, _ = _synthetic()
    expected = alignment.current_semantic_loss(model, views, query, anchor, torch.arange(query.numel()), t, u, 2)
    # The frozen epoch has one batch for this synthetic input. lr=0 makes its
    # mandated frozen optimiser step observationally inert for parity only.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    actual = frozen._semantic_epoch(model, optimizer, views, query, anchor, t, u, np.arange(query.numel()), 2)
    assert actual == pytest.approx(float(expected.detach()), abs=1e-7)


def test_carrier_projection_shapes_and_shuffle_contract():
    _, _, query, anchor, t, full_u, u, _ = _synthetic()
    alignment.validate_carrier(query, anchor, t, full_u, u)
    shuffled = alignment.shuffle_u(u)
    assert torch.equal(shuffled.sum(1), u.sum(1))
    assert torch.equal(torch.sort(shuffled, 1).values, torch.sort(u, 1).values)
    assert not torch.equal(shuffled, u)


def test_autograd_grad_does_not_populate_parameter_grad_and_zero_norm_is_invalid():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    result = alignment.gradients(parameter * 0.0, (("p", parameter),))
    assert parameter.grad is None and result.norm == 0.0
    assert alignment.cosine(result, result) == (None, "ZERO_GRAD_NORM")


def test_entropy_and_semantic_margin_known_examples():
    report = utility_diagnostics(torch.tensor([[1., 0.], [.5, .5]]))
    assert report["entropy_mean"] == pytest.approx(.5)
    t = torch.tensor([[[.9], [.1]], [[.4], [.6]]])  # [Nq,A,S]
    margins = semantic_margins(t, torch.tensor([0, 1]), 2)
    assert torch.allclose(margins, torch.tensor([[.8], [.2]]))


def test_quintiles_are_deterministic_and_q_alignment_sign_examples():
    u = torch.tensor([[.1, .2, .3, .4, .5], [.6, .7, .8, .9, 1.]])
    first, second = alignment.quintile_masks(u), alignment.quintile_masks(u)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert alignment.spearman(range(1, 6), [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert alignment.spearman(range(1, 6), [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def _cells(positive_uniform=16, positive_shuffle=16, datasets_uniform=6, datasets_shuffle=6, q5=16, coupling=16):
    def selected(count, dataset_count):
        # At least two positive seed cells make a dataset mean positive.  Spread
        # the requested cell count over the requested datasets deterministically.
        if count < 2 * dataset_count: return set(range(count))
        picks = [dataset * 3 + seed for dataset in range(dataset_count) for seed in range(2)]
        for dataset in range(dataset_count):
            if len(picks) >= count: break
            picks.append(dataset * 3 + 2)
        return set(picks[:count])
    uniform_ids = selected(positive_uniform, datasets_uniform)
    shuffle_ids = selected(positive_shuffle, datasets_shuffle)
    cells = []
    for index, (dataset, seed) in enumerate(manifest_cells()):
        du = 1.0 if index in uniform_ids else -1.0
        ds = 1.0 if index in shuffle_ids else -1.0
        cells.append({"dataset": dataset, "training_seed": seed, "Delta_TRUE_UNIFORM": du,
                      "Delta_TRUE_SHUFFLE": ds, "DeltaAlignment_Q5_Q1": 1.0 if index < q5 else -1.0,
                      "Spearman_quintile_alignment": 1.0, "Spearman_U_semantic_margin": 1.0 if index < coupling else -1.0})
    return cells


def test_manifest_and_exact_summary_boundaries():
    assert len(DATASETS) == 8 and len(SEEDS) == 3 and len(manifest_cells()) == 24
    assert summarize(_cells(positive_uniform=15))["RELATIVE_U_ALIGNMENT_SUPPORTED"] is False
    assert summarize(_cells())["RELATIVE_U_ALIGNMENT_SUPPORTED"] is True
    assert summarize(_cells(datasets_uniform=5))["RELATIVE_U_ALIGNMENT_SUPPORTED"] is False
    assert summarize(_cells(datasets_uniform=6, datasets_shuffle=6))["RELATIVE_U_ALIGNMENT_SUPPORTED"] is True


def test_runner_has_no_gt_cli_metric_import_or_optimizer_step():
    import experiments.utility_relation.p3_c4_a0_utility_action_alignment.run_diagnostic as runner
    source = inspect.getsource(runner)
    parser_source = inspect.getsource(runner.main)
    assert "--gt" not in parser_source.lower() and "--label" not in parser_source.lower()
    assert "optimizer.step" not in source and "loss.backward" not in source
    assert "ACC" not in source and "NMI" not in source and "ARI" not in source


def test_full_read_only_diagnostic_cpu():
    model, views, query, anchor, t, full_u, u, labels = _synthetic()
    result = diagnose_cell(model, views, query, anchor, t, full_u, u, labels, query.clone(), batch_size=3)
    assert result["model_state_sha256_before_after"]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result["detach_contract"] == {"U_cycle": True, "T_rel": True, "anchor_posterior": True}
