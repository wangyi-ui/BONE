"""CPU contract tests for the P3-C3-A0 sealed dataset-validation pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation.p3_c3_a0_ucab import evaluate_confirmatory as evaluator
from experiments.utility_relation.p3_c3_a0_ucab import run_confirmatory as runner
from experiments.utility_relation.p3_c3_a0_ucab import summarize_confirmatory as summary
from experiments.utility_relation.p3_c3_a0_ucab import validation_io as io


def _metric_row(dataset: str, seed: int, true_acc: float, current_acc: float, mean_acc: float | None = None, shuffle_acc: float | None = None) -> dict:
    arms = io.arms_for_seed(seed)
    values = {"CURRENT": current_acc, "TRUE_BUDGET": true_acc}
    if mean_acc is not None:
        values["MEAN_BUDGET"] = mean_acc
        values["SHUFFLE_BUDGET"] = shuffle_acc if shuffle_acc is not None else mean_acc
    def metrics(value: float) -> dict:
        return {"ACC": value, "NMI": value, "ARI": value}
    return {"status": "REPORTING_ONLY", "dataset": dataset, "training_seed": seed, "seed_role": io.seed_role(seed), "arms": list(arms), **{arm: metrics(values[arm]) for arm in arms}}


def _write_all_metrics(root: Path, historical_value: tuple[float, float] = (0.1, 0.9), unseen_value: tuple[float, float, float, float] = (0.7, 0.5, 0.4, 0.3)) -> None:
    for dataset in ("ALOI", "CCV"):
        for seed in (20, 30, 50):
            io.write_json(io.run_root(root, dataset, seed) / "formal_metrics.json", _metric_row(dataset, seed, *historical_value))
        for seed in (140, 150, 170):
            io.write_json(io.run_root(root, dataset, seed) / "formal_metrics.json", _metric_row(dataset, seed, *unseen_value))


def test_parent_freezes_and_frozen_sources_are_intact() -> None:
    observed = io.verify_parent_freezes()
    assert set(observed) == {"cpu_implementation", "cpu_legacy", "smoke_source", "smoke_ucab", "smoke_legacy"}


@pytest.mark.parametrize("dataset", ("ALOI", "CCV"))
def test_only_registered_datasets_and_frozen_semantic_configs(dataset: str) -> None:
    config = io.semantic_config(dataset)
    assert config == ({"optimizer": "Adam", "semantic_lr": 1e-4, "semantic_exposure": 1} if dataset == "ALOI" else {"optimizer": "Adam", "semantic_lr": 1e-5, "semantic_exposure": 3})
    with pytest.raises(ValueError):
        io.validate_dataset_seed("MNIST-USPS", 20)


def test_only_registered_seeds_and_exact_arm_sets() -> None:
    with pytest.raises(ValueError):
        io.validate_dataset_seed("ALOI", 99)
    assert io.arms_for_seed(20) == ("CURRENT", "TRUE_BUDGET")
    assert io.arms_for_seed(140) == ("CURRENT", "TRUE_BUDGET", "MEAN_BUDGET", "SHUFFLE_BUDGET")


def test_runner_firewall_and_no_tunable_cli() -> None:
    source = inspect.getsource(runner)
    assert "evaluate_bone_p3_c0_cross_dataset_formal" not in source
    assert "compute_metrics" not in source
    options = {action.dest for action in runner.build_parser()._actions}
    assert "semantic_lr" not in options and "semantic_exposure" not in options and "optimizer" not in options
    assert "full_gt_path" not in options and "seed_replacement" not in options


def test_existing_or_partial_output_fails_closed(tmp_path: Path) -> None:
    root = io.run_root(tmp_path, "ALOI", 20)
    root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="PARTIAL"):
        runner.run_confirmatory("ALOI", 20, tmp_path)


def test_seal_verifier_rejects_unsealed_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io.verify_pre_gt_seal(io.run_root(tmp_path, "CCV", 140))


def test_evaluator_validates_seal_before_gt_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False
    def bad_seal(_root):
        raise RuntimeError("seal invalid")
    def gt_loader(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("GT must not load")
    monkeypatch.setattr(evaluator.io, "verify_pre_gt_seal", bad_seal)
    monkeypatch.setattr(evaluator.scipy.io, "loadmat", gt_loader)
    with pytest.raises(RuntimeError, match="seal invalid"):
        evaluator.evaluate_confirmatory(io.run_root(tmp_path, "ALOI", 20))
    assert not called


def test_summarizer_requires_exactly_all_12_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FORMAL_METRICS_MISSING"):
        summary.summarize_confirmatory(tmp_path)


def test_historical_results_do_not_affect_confirmatory_decision(tmp_path: Path) -> None:
    _write_all_metrics(tmp_path, historical_value=(-99.0, 99.0))
    result = summary.summarize_confirmatory(tmp_path)
    assert result["P3_C3_A0_UCAB_CONFIRMATORY"] == "PASS"
    assert result["historical_affects_confirmatory_decision"] is False


def test_primary_gate_math_fails_when_one_dataset_mean_is_nonpositive(tmp_path: Path) -> None:
    _write_all_metrics(tmp_path)
    for seed in (140, 150, 170):
        io.write_json(io.run_root(tmp_path, "CCV", seed) / "formal_metrics.json", _metric_row("CCV", seed, 0.4, 0.5, 0.2, 0.1))
    result = summary.summarize_confirmatory(tmp_path)
    assert result["PRIMARY_GATE"] == "FAIL"


def test_mechanism_gate_math_fails_without_true_budget_control_advantage(tmp_path: Path) -> None:
    _write_all_metrics(tmp_path, unseen_value=(0.7, 0.5, 0.8, 0.3))
    result = summary.summarize_confirmatory(tmp_path)
    assert result["PRIMARY_GATE"] == "PASS"
    assert result["MECHANISM_GATE"] == "FAIL"


def test_secondary_gate_math_requires_two_of_three_positive_metrics(tmp_path: Path) -> None:
    _write_all_metrics(tmp_path)
    for dataset in ("ALOI", "CCV"):
        for seed in (140, 150, 170):
            row = _metric_row(dataset, seed, 0.7, 0.5, 0.4, 0.3)
            row["TRUE_BUDGET"]["NMI"] = 0.1; row["CURRENT"]["NMI"] = 0.2
            row["TRUE_BUDGET"]["ARI"] = 0.1; row["CURRENT"]["ARI"] = 0.2
            io.write_json(io.run_root(tmp_path, dataset, seed) / "formal_metrics.json", row)
    result = summary.summarize_confirmatory(tmp_path)
    assert result["PRIMARY_GATE"] == "PASS"
    assert result["SECONDARY_GATE"] == "FAIL"


def _query_utility_payload() -> tuple[dict[str, np.ndarray], object]:
    n, anchor_count, query_count, actions = 8, 2, 6, 3
    adapter = runner.b0.SampleIdAdapter(np.arange(n, dtype=np.int64))
    query_rows = np.asarray([1, 2, 3, 5, 6, 7], dtype=np.int64)
    full_u_cycle = np.linspace(0.05, 0.95, n * actions, dtype=np.float64).reshape(n, actions)
    return {
        "query_rows": query_rows,
        "anchor_rows": np.asarray([0, 4], dtype=np.int64),
        "T_rel": np.zeros((query_count, anchor_count, actions), dtype=np.float64),
        "U_cycle": full_u_cycle,
        "query_U_cycle": full_u_cycle[query_rows].copy(),
    }, adapter


def test_query_utility_payload_uses_exact_full_row_projection() -> None:
    payload, adapter = _query_utility_payload()
    query_rows, anchor_rows, t_rel, returned_u = runner._validate_carrier_payload(payload, adapter, k=2)
    assert query_rows.shape == (6,) and anchor_rows.shape == (2,) and t_rel.shape == (6, 2, 3)
    assert returned_u.shape == (6, 3)
    assert np.array_equal(returned_u, payload["query_U_cycle"])
    assert payload["U_cycle"].shape[0] != query_rows.size  # Full-row U is intentionally not semantic U.


def test_query_utility_projection_mismatch_fails_closed() -> None:
    payload, adapter = _query_utility_payload()
    payload["query_U_cycle"][0, 0] += 0.01
    with pytest.raises(RuntimeError, match="QUERY_U_CYCLE_PROJECTION_MISMATCH"):
        runner._validate_carrier_payload(payload, adapter, k=2)


def test_wrong_query_utility_shape_fails_closed() -> None:
    payload, adapter = _query_utility_payload()
    payload["query_U_cycle"] = payload["query_U_cycle"][:-1]
    with pytest.raises(RuntimeError, match="T_REL_U_CYCLE_QUERY_ALIGNMENT_MISMATCH"):
        runner._validate_carrier_payload(payload, adapter, k=2)


def test_t_rel_query_action_dimension_mismatch_fails_closed() -> None:
    payload, adapter = _query_utility_payload()
    payload["T_rel"] = payload["T_rel"][:, :, :2]
    with pytest.raises(RuntimeError, match="T_REL_U_CYCLE_QUERY_ALIGNMENT_MISMATCH"):
        runner._validate_carrier_payload(payload, adapter, k=2)


def test_full_utility_adapter_cardinality_mismatch_fails_closed() -> None:
    payload, adapter = _query_utility_payload()
    payload["U_cycle"] = payload["U_cycle"][:-1]
    with pytest.raises(RuntimeError, match="T_REL_U_CYCLE_QUERY_ALIGNMENT_MISMATCH"):
        runner._validate_carrier_payload(payload, adapter, k=2)
