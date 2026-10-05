from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import context
from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import evaluate_confirmatory as evaluator
from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import protocol
from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import run_confirmatory as runner
from experiments.utility_relation.p3_c1_b3_msrc_confirmatory import summarize_confirmatory as summarizer


def _mock_audit(config: dict) -> dict:
    shared = "a" * 64
    return {
        "dataset": "MSRC",
        "training_seed": config["training_seed"],
        "arms": list(protocol.ARMS),
        "arm_parameters": protocol.ARM_PARAMETERS,
        "carrier_state_sha256": shared,
        "initial_BASE_state_sha256": shared,
        "initial_B1_state_sha256": shared,
        "initial_B2_state_sha256": shared,
        "ALL_ARMS_SHARED_CARRIER": True,
        "B1_B2_SAME_INITIALIZATION": True,
        "METHOD_CHANGED": False,
        "OPTIMIZER_CHANGED": False,
        "GT_loaded_before_seal": False,
        "full_gt_used_for_training": False,
        "full_gt_used_for_selection": False,
        "ACC_NMI_ARI_computed": False,
        "formal_20_30_50_results_read": False,
    }


def _mock_backend(config: dict, root: Path) -> dict:
    artifacts = []
    predictions = {
        "BASE": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "OURS_B1": np.asarray([0, 1, 1, 0], dtype=np.int64),
        "OURS_B2": np.asarray([0, 0, 1, 0], dtype=np.int64),
    }
    for arm in protocol.ARMS:
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "final_model_state.pt").write_bytes(arm.encode("ascii"))
        np.savez_compressed(
            directory / "pre_gt_predictions.npz",
            sample_ids=np.arange(4, dtype=np.int64),
            predictions=predictions[arm],
        )
        artifacts.extend(
            [f"{arm}/final_model_state.pt", f"{arm}/pre_gt_predictions.npz"]
        )
    return {"artifacts": artifacts, "audit": _mock_audit(config)}


@pytest.fixture
def preregistration(tmp_path):
    path = tmp_path / "preregistered"
    protocol.preregister(path)
    return path


def test_exact_dataset_seeds_arms_and_parameters():
    assert protocol.DATASET == "MSRC"
    assert protocol.CONFIRMATORY_SEEDS == (60, 70, 90)
    assert protocol.FORBIDDEN_SELECTION_SEEDS == (20, 30, 50)
    assert protocol.ARMS == ("BASE", "OURS_B1", "OURS_B2")
    assert protocol.ARM_PARAMETERS == {
        "BASE": {"semantic_lr": None, "semantic_exposure": 0},
        "OURS_B1": {"semantic_lr": 1e-3, "semantic_exposure": 1},
        "OURS_B2": {"semantic_lr": 2e-3, "semantic_exposure": 1},
    }
    with pytest.raises(ValueError, match="MSRC_ONLY"):
        context.resolve_training_inputs("ALOI")
    with pytest.raises(ValueError, match="UNREGISTERED"):
        protocol.run_root(protocol.OUTPUT_ROOT, 20)


def test_cli_has_no_parameter_or_training_rule_override():
    destinations = {action.dest for action in runner.build_parser()._actions}
    forbidden = {
        "semantic_lr",
        "semantic_exposure",
        "carrier_epochs",
        "refinement_epochs",
        "optimizer",
        "dataset",
        "arm",
    }
    assert not destinations.intersection(forbidden)
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--training-seed", "60", "--semantic-lr", "0.9", "--execute"]
        )


def test_parent_freeze_integrity_and_exact_b2_winner(monkeypatch):
    result = protocol.verify_parent_integrity()
    assert result["verified"] is True
    assert result["parent_source_modification"] is False
    altered = dict(protocol.PARENT_B2_FREEZE_SHA256)
    altered["STATUS.txt"] = "0" * 64
    monkeypatch.setattr(protocol, "PARENT_B2_FREEZE_SHA256", altered)
    with pytest.raises(RuntimeError, match="PARENT_FREEZE_HASH_MISMATCH"):
        protocol.verify_parent_integrity()


def test_preregistration_is_pre_run_and_source_bound(preregistration):
    verified = protocol.verify_preregistration(preregistration)
    assert verified["verified"] is True
    status = (preregistration / "STATUS.txt").read_text(encoding="utf-8")
    assert "SEEDS=60,70,90" in status
    assert "REAL_CONFIRMATORY_RUN=false" in status
    assert "B2_PARAMETER_ALREADY_FROZEN=true" in status
    assert "PARAMETER_RETUNING_ALLOWED=false" in status


def test_mock_run_has_shared_carrier_output_separation_and_is_idempotent(
    tmp_path, preregistration
):
    output = tmp_path / "outputs"
    result = runner.run_confirmatory(
        60,
        output_root=output,
        preregistration_freeze=preregistration,
        backend=_mock_backend,
    )
    assert result["status"] == "COMPLETE"
    root = output / "seed60"
    audit = json.loads((root / "PRE_GT_AUDIT.json").read_text(encoding="utf-8"))
    assert {
        audit["carrier_state_sha256"],
        audit["initial_BASE_state_sha256"],
        audit["initial_B1_state_sha256"],
        audit["initial_B2_state_sha256"],
    } == {"a" * 64}
    assert audit["full_gt_used_for_training"] is False
    assert audit["formal_20_30_50_results_read"] is False
    assert all((root / arm / "pre_gt_predictions.npz").is_file() for arm in protocol.ARMS)

    def forbidden_backend(_config, _root):
        raise AssertionError("sealed run must not retrain")

    again = runner.run_confirmatory(
        60,
        output_root=output,
        preregistration_freeze=preregistration,
        backend=forbidden_backend,
    )
    assert again["status"] == "ALREADY_COMPLETE_VERIFIED"


def test_gt_is_read_only_after_valid_pre_gt_seal(monkeypatch, tmp_path, preregistration):
    output = tmp_path / "outputs"
    runner.run_confirmatory(
        70,
        output_root=output,
        preregistration_freeze=preregistration,
        backend=_mock_backend,
    )
    calls = []

    def fake_loadmat(path, variable_names):
        calls.append((Path(path).name, tuple(variable_names)))
        return {"Y": np.asarray([[0], [0], [1], [1]], dtype=np.int64)}

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", fake_loadmat)
    result = evaluator.evaluate_confirmatory(70, output)
    assert calls == [("MSRC.mat", ("Y",))]
    assert set(result["BASE"]) == {"ACC", "NMI", "ARI"}
    assert set(result["OURS_B1"]) == {"ACC", "NMI", "ARI"}
    assert set(result["OURS_B2"]) == {"ACC", "NMI", "ARI"}
    assert result["metrics_used_for_parameter_selection"] is False
    assert result["parameter_retuning_allowed"] is False


def test_corrupt_seal_blocks_gt_read(monkeypatch, tmp_path, preregistration):
    output = tmp_path / "outputs"
    runner.run_confirmatory(
        90,
        output_root=output,
        preregistration_freeze=preregistration,
        backend=_mock_backend,
    )
    (output / "seed90/BASE/pre_gt_predictions.npz").write_bytes(b"corrupt")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("GT must not be read")

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", forbidden)
    with pytest.raises(RuntimeError, match="SEAL_INVALID"):
        evaluator.evaluate_confirmatory(90, output)
    assert called is False


def test_frozen_training_schedule_optimizer_and_order_algorithm():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert protocol.CARRIER_EPOCHS == 100
    assert protocol.REFINEMENT_EPOCHS == 20
    assert "if epoch == 0" in source
    assert 'for arm in ("OURS_B1", "OURS_B2")' in source
    assert "for arm in protocol.ARMS" in source
    assert "optim.Adam" in source
    assert "c0._semantic_epoch" in source
    assert "c0._native_epoch" in source
    assert "loadmat" not in source
    expected_rng = np.random.default_rng(60)
    expected = np.stack([expected_rng.permutation(17) for _ in range(20)])
    assert np.array_equal(runner._semantic_orders(17, 60), expected)


def test_three_seed_population_summary_and_result_freeze(
    monkeypatch, tmp_path, preregistration
):
    output = tmp_path / "outputs"
    labels = np.asarray([[0], [0], [1], [1]], dtype=np.int64)
    monkeypatch.setattr(
        evaluator.scipy.io,
        "loadmat",
        lambda *_args, **_kwargs: {"Y": labels},
    )
    for seed in protocol.CONFIRMATORY_SEEDS:
        runner.run_confirmatory(
            seed,
            output_root=output,
            preregistration_freeze=preregistration,
            backend=_mock_backend,
        )
        evaluator.evaluate_confirmatory(seed, output)
    result = summarizer.summarize_confirmatory(output)
    assert result["run_count"] == 9
    assert result["all_runs_complete"] is True
    assert result["population_std_ddof"] == 0
    for metric in ("ACC", "NMI", "ARI"):
        assert "B2_gt_B1_positive_seed_count" in result["metrics"][metric]
        assert "B2_gt_BASE_positive_seed_count" in result["metrics"][metric]
    frozen = summarizer.freeze_results(
        output,
        tmp_path / "results_freeze",
        preregistration_freeze=preregistration,
    )
    expected = {
        "STATUS.txt",
        "PER_SEED_RESULTS.csv",
        "THREE_SEED_SUMMARY.csv",
        "THREE_SEED_SUMMARY.json",
        "artifact_sha256.txt",
        "source_sha256.txt",
        "parent_freeze_sha256.txt",
        "PARENT_INTEGRITY.txt",
    }
    assert {path.name for path in frozen.iterdir()} == expected
    assert "RUN_COUNT=9" in (frozen / "STATUS.txt").read_text(encoding="utf-8")
